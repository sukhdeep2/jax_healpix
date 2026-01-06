#!/usr/bin/env python3
"""Profile CUDA map2alm implementation.

Usage:
    python profile_map2alm.py [nside]

    # For detailed GPU profiling with Nsight Systems:
    nsys profile -o profile_output python profile_map2alm.py [nside]

    # For Nsight Compute (kernel-level analysis):
    ncu --set full -o profile_kernels python profile_map2alm.py [nside]
"""

import numpy as np
import sys
import time

sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')

# ============================================================================
# CONFIGURATION - Edit these settings
# ============================================================================
STORAGE_PRECISION = 'float32'      # 'float32' or 'float64'
RECURRENCE_PRECISION = 'float32'   # 'float32' or 'float64'
PHASE1_METHOD = 'bluestein'        # 'dft', 'fft_equatorial', or 'bluestein'
N_ITERATIONS = 5                   # Number of iterations for timing
N_MAPS = 1                         # Number of maps to process
# ============================================================================

from spht_cuda import (SPHTCuda, set_phase1_method, get_phase1_method_name,
                       PHASE1_DFT, PHASE1_FFT_EQUATORIAL, PHASE1_BLUESTEIN)

def get_phase1_constant(method_name):
    """Convert method name to constant."""
    methods = {
        'dft': PHASE1_DFT,
        'fft_equatorial': PHASE1_FFT_EQUATORIAL,
        'bluestein': PHASE1_BLUESTEIN,
    }
    return methods.get(method_name.lower(), PHASE1_DFT)


def profile_map2alm(nside, storage_prec, recurrence_prec, phase1_method, n_iterations, n_maps):
    """Profile map2alm with detailed timing."""

    l_max = 3 * nside
    n_rings = 4 * nside - 1
    n_pix = 12 * nside * nside
    n_alm = (l_max + 1) * (l_max + 2) // 2

    # Set phase1 method
    set_phase1_method(get_phase1_constant(phase1_method))

    print("=" * 70)
    print("CUDA map2alm Profiling")
    print("=" * 70)
    print(f"  nside              = {nside}")
    print(f"  l_max              = {l_max}")
    print(f"  n_rings            = {n_rings}")
    print(f"  n_pixels           = {n_pix:,}")
    print(f"  n_alm              = {n_alm:,}")
    print(f"  n_maps             = {n_maps}")
    print(f"  storage_precision  = {storage_prec}")
    print(f"  recurrence_precision = {recurrence_prec}")
    print(f"  phase1_method      = {get_phase1_method_name()}")
    print(f"  iterations         = {n_iterations}")
    print("=" * 70)
    print()

    # Create input data
    dtype = np.float32 if storage_prec == 'float32' else np.float64
    np.random.seed(42)
    map_array = np.random.randn(n_maps, n_rings, 4 * nside).astype(dtype)

    # Create SPHT object
    print("Creating SPHTCuda object...")
    t0 = time.perf_counter()
    spht = SPHTCuda(nside, l_max, version='v6',
                    storage_precision=storage_prec,
                    recurrence_precision=recurrence_prec)
    t_init = time.perf_counter() - t0
    print(f"  Initialization time: {t_init*1000:.2f} ms")
    print()

    # Warmup run
    print("Warmup run...")
    t0 = time.perf_counter()
    _ = spht.map2alm({0: map_array}, spins=(0,))
    t_warmup = time.perf_counter() - t0
    print(f"  Warmup time: {t_warmup*1000:.2f} ms")
    print()

    # Profiling runs
    print(f"Profiling ({n_iterations} iterations)...")
    times = []
    for i in range(n_iterations):
        t0 = time.perf_counter()
        result = spht.map2alm({0: map_array}, spins=(0,))
        t_iter = time.perf_counter() - t0
        times.append(t_iter)
        print(f"  Iteration {i+1}: {t_iter*1000:.2f} ms")

    # Statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print()
    print("=" * 70)
    print("Timing Summary")
    print("=" * 70)
    print(f"  Mean time:     {mean_time*1000:.2f} ms")
    print(f"  Std dev:       {std_time*1000:.2f} ms")
    print(f"  Min time:      {min_time*1000:.2f} ms")
    print(f"  Max time:      {max_time*1000:.2f} ms")
    print()

    # Throughput analysis
    total_input_bytes = map_array.nbytes
    total_output_bytes = n_maps * (l_max + 1) * (l_max + 1) * 2 * (4 if dtype == np.float32 else 8)
    total_bytes = total_input_bytes + total_output_bytes

    # Compute operations estimate
    # Phase 1 (DFT): O(n_rings * n_phi * l_max) for DFT, O(n_rings * n_phi * log(n_phi)) for Bluestein
    # Phase 2: O(n_rings * l_max^2) for Ylm summation
    if phase1_method.lower() == 'bluestein':
        phase1_ops = n_rings * 4 * nside * np.log2(8 * nside) * 5  # ~5 ops per FFT element
    else:
        phase1_ops = n_rings * 4 * nside * l_max * 4  # 4 ops per DFT element (sin, cos, mul, add)

    phase2_ops = n_rings / 2 * l_max * l_max * 6  # ~6 ops per Ylm (recurrence + multiply + accumulate)
    total_ops = (phase1_ops + phase2_ops) * n_maps

    print("Throughput Analysis")
    print("=" * 70)
    print(f"  Input data:    {total_input_bytes / 1e6:.2f} MB")
    print(f"  Output data:   {total_output_bytes / 1e6:.2f} MB")
    print(f"  Total I/O:     {total_bytes / 1e6:.2f} MB")
    print(f"  Bandwidth:     {total_bytes / mean_time / 1e9:.2f} GB/s")
    print()
    print(f"  Est. Phase 1 ops: {phase1_ops / 1e9:.2f} GFLOP")
    print(f"  Est. Phase 2 ops: {phase2_ops / 1e9:.2f} GFLOP")
    print(f"  Est. Total ops:   {total_ops / 1e9:.2f} GFLOP")
    print(f"  Est. GFLOPS:      {total_ops / mean_time / 1e9:.2f} GFLOPS")
    print("=" * 70)
    print()

    # Verify result is valid
    alm = result[0][0]
    print("Result Validation")
    print("=" * 70)
    print(f"  Output shape: {alm.shape}")
    print(f"  Output dtype: {alm.dtype}")
    print(f"  All finite:   {np.all(np.isfinite(alm))}")
    print(f"  Max |alm|:    {np.max(np.abs(alm)):.6e}")
    print(f"  Mean |alm|:   {np.mean(np.abs(alm)):.6e}")
    print("=" * 70)

    return mean_time, std_time


def main():
    nside = int(sys.argv[1]) if len(sys.argv) > 1 else 256

    profile_map2alm(
        nside=nside,
        storage_prec=STORAGE_PRECISION,
        recurrence_prec=RECURRENCE_PRECISION,
        phase1_method=PHASE1_METHOD,
        n_iterations=N_ITERATIONS,
        n_maps=N_MAPS
    )


if __name__ == "__main__":
    main()

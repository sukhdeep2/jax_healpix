#!/usr/bin/env python3
"""Benchmark map2cl CUDA implementation.

Compares performance of map2cl pipeline between CUDA and JAX for various
configurations including spin-0, spin-2, and mixed maps.

Usage:
    python benchmark_map2cl.py [nside] [n_iterations]

    nside: HEALPix resolution parameter (default: 128)
    n_iterations: Number of iterations for timing (default: 5)
"""

import numpy as np
import sys
import time
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')

from spht_cuda import (set_phase1_method, PHASE1_DFT, PHASE1_BLUESTEIN)

# Default parameters
NSIDE = 128
N_ITERATIONS = 2
USE_BLUESTEIN = True  # Use Bluestein FFT for better performance


def benchmark_cuda_map2cl_spin0(nside, l_max, n_iterations, precision='float64'):
    """Benchmark CUDA map2cl for spin-0 only."""
    from spht_cuda import map2cl_cuda

    # Set phase1 method
    set_phase1_method(PHASE1_BLUESTEIN if USE_BLUESTEIN else PHASE1_DFT)

    dtype = np.float32 if precision == 'float32' else np.float64
    n_rings = 4 * nside - 1
    # Use matching recurrence precision for best performance
    recurrence = precision

    # Create random map
    np.random.seed(42)
    T_map = np.random.randn(n_rings, 4 * nside).astype(dtype)

    # Extended warmup (GPU power management can cause periodic slowdowns)
    for _ in range(5):
        _ = map2cl_cuda(nside, l_max, [T_map],
                        storage_precision=precision,
                        recurrence_precision=recurrence)

    # Benchmark with extra samples for robust statistics
    times = []
    for _ in range(n_iterations * 3):  # 3x samples for filtering
        start = time.perf_counter()
        result = map2cl_cuda(nside, l_max, [T_map],
                            storage_precision=precision,
                            recurrence_precision=recurrence)
        end = time.perf_counter()
        times.append(end - start)

    # Use median (robust to GPU throttling spikes)
    return np.median(times), np.std(times)


def benchmark_cuda_map2cl_spin2(nside, l_max, n_iterations, precision='float64'):
    """Benchmark CUDA map2cl for spin-2 only."""
    from spht_cuda import map2cl_cuda

    # Set phase1 method
    set_phase1_method(PHASE1_BLUESTEIN if USE_BLUESTEIN else PHASE1_DFT)

    dtype = np.float32 if precision == 'float32' else np.float64
    n_rings = 4 * nside - 1
    # Use matching recurrence precision for best performance
    recurrence = precision

    # Create random Q, U maps
    np.random.seed(42)
    Q_map = np.random.randn(n_rings, 4 * nside).astype(dtype)
    U_map = np.random.randn(n_rings, 4 * nside).astype(dtype)
    QU_map = np.stack([Q_map, U_map], axis=0)

    # Extended warmup
    for _ in range(5):
        _ = map2cl_cuda(nside, l_max, [QU_map],
                        storage_precision=precision,
                        recurrence_precision=recurrence)

    # Benchmark with extra samples
    times = []
    for _ in range(n_iterations * 3):
        start = time.perf_counter()
        result = map2cl_cuda(nside, l_max, [QU_map],
                            storage_precision=precision,
                            recurrence_precision=recurrence)
        end = time.perf_counter()
        times.append(end - start)

    return np.median(times), np.std(times)


def benchmark_cuda_map2cl_mixed(nside, l_max, n_iterations, precision='float64'):
    """Benchmark CUDA map2cl for mixed T + QU."""
    from spht_cuda import map2cl_cuda

    # Set phase1 method
    set_phase1_method(PHASE1_BLUESTEIN if USE_BLUESTEIN else PHASE1_DFT)

    dtype = np.float32 if precision == 'float32' else np.float64
    n_rings = 4 * nside - 1
    # Use matching recurrence precision for best performance
    recurrence = precision

    # Create random T, Q, U maps
    np.random.seed(42)
    T_map = np.random.randn(n_rings, 4 * nside).astype(dtype)
    Q_map = np.random.randn(n_rings, 4 * nside).astype(dtype)
    U_map = np.random.randn(n_rings, 4 * nside).astype(dtype)
    QU_map = np.stack([Q_map, U_map], axis=0)

    # Extended warmup
    for _ in range(5):
        _ = map2cl_cuda(nside, l_max, [T_map, QU_map],
                        storage_precision=precision,
                        recurrence_precision=recurrence)

    # Benchmark with extra samples
    times = []
    for _ in range(n_iterations * 3):
        start = time.perf_counter()
        result = map2cl_cuda(nside, l_max, [T_map, QU_map],
                            storage_precision=precision,
                            recurrence_precision=recurrence)
        end = time.perf_counter()
        times.append(end - start)

    return np.median(times), np.std(times)


def benchmark_cuda_map2cl_two_sets(nside, l_max, n_iterations, precision='float64'):
    """Benchmark CUDA map2cl for two TQU sets."""
    from spht_cuda import map2cl_cuda

    # Set phase1 method
    set_phase1_method(PHASE1_BLUESTEIN if USE_BLUESTEIN else PHASE1_DFT)

    dtype = np.float32 if precision == 'float32' else np.float64
    n_rings = 4 * nside - 1
    # Use matching recurrence precision for best performance
    recurrence = precision

    # Create two TQU sets
    np.random.seed(42)
    T1 = np.random.randn(n_rings, 4 * nside).astype(dtype)
    Q1 = np.random.randn(n_rings, 4 * nside).astype(dtype)
    U1 = np.random.randn(n_rings, 4 * nside).astype(dtype)
    QU1 = np.stack([Q1, U1], axis=0)

    np.random.seed(123)
    T2 = np.random.randn(n_rings, 4 * nside).astype(dtype)
    Q2 = np.random.randn(n_rings, 4 * nside).astype(dtype)
    U2 = np.random.randn(n_rings, 4 * nside).astype(dtype)
    QU2 = np.stack([Q2, U2], axis=0)

    # Extended warmup
    for _ in range(5):
        _ = map2cl_cuda(nside, l_max, [T1, QU1, T2, QU2],
                        storage_precision=precision,
                        recurrence_precision=recurrence)

    # Benchmark with extra samples
    times = []
    for _ in range(n_iterations * 3):
        start = time.perf_counter()
        result = map2cl_cuda(nside, l_max, [T1, QU1, T2, QU2],
                            storage_precision=precision,
                            recurrence_precision=recurrence)
        end = time.perf_counter()
        times.append(end - start)

    return np.median(times), np.std(times)


def benchmark_jax_spin0(nside, l_max, n_iterations, precision='float64'):
    """Benchmark JAX map2alm + alm2cl for spin-0."""
    import jax
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2cl as jax_alm2cl

    if precision == 'float64':
        jax.config.update("jax_enable_x64", True)
        dtype = np.float64
    else:
        jax.config.update("jax_enable_x64", False)
        dtype = np.float32

    n_rings = 4 * nside - 1

    np.random.seed(42)
    T_map = np.random.randn(n_rings, 4 * nside).astype(dtype)
    maps_jax = {0: jnp.array(T_map[None, :, :])}

    # Warmup
    alm = jax_map2alm(nside, l_max, (0,), maps_jax)
    cl = jax_alm2cl(l_max, alm[0])
    jax.block_until_ready(cl)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        alm = jax_map2alm(nside, l_max, (0,), maps_jax)
        cl = jax_alm2cl(l_max, alm[0])
        jax.block_until_ready(cl)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def benchmark_jax_spin2(nside, l_max, n_iterations, precision='float64'):
    """Benchmark JAX map2alm + alm2cl for spin-2."""
    import jax
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2cl as jax_alm2cl

    if precision == 'float64':
        jax.config.update("jax_enable_x64", True)
        dtype = np.float64
    else:
        jax.config.update("jax_enable_x64", False)
        dtype = np.float32

    n_rings = 4 * nside - 1

    np.random.seed(42)
    Q_map = np.random.randn(n_rings, 4 * nside).astype(dtype)
    U_map = np.random.randn(n_rings, 4 * nside).astype(dtype)
    maps_jax = {
        2: jnp.array(Q_map[None, :, :]),
        -2: jnp.array(U_map[None, :, :])
    }

    # Warmup
    alm = jax_map2alm(nside, l_max, (2,), maps_jax)
    cl_EE = jax_alm2cl(l_max, alm[2])
    cl_BB = jax_alm2cl(l_max, alm[-2])
    cl_EB = jax_alm2cl(l_max, alm[2], alm[-2])
    jax.block_until_ready((cl_EE, cl_BB, cl_EB))

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        alm = jax_map2alm(nside, l_max, (2,), maps_jax)
        cl_EE = jax_alm2cl(l_max, alm[2])
        cl_BB = jax_alm2cl(l_max, alm[-2])
        cl_EB = jax_alm2cl(l_max, alm[2], alm[-2])
        jax.block_until_ready((cl_EE, cl_BB, cl_EB))
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def benchmark_jax_mixed(nside, l_max, n_iterations, precision='float64'):
    """Benchmark JAX for T + QU (all 6 spectra)."""
    import jax
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2cl as jax_alm2cl

    if precision == 'float64':
        jax.config.update("jax_enable_x64", True)
        dtype = np.float64
    else:
        jax.config.update("jax_enable_x64", False)
        dtype = np.float32

    n_rings = 4 * nside - 1

    np.random.seed(42)
    T_map = np.random.randn(n_rings, 4 * nside).astype(dtype)
    Q_map = np.random.randn(n_rings, 4 * nside).astype(dtype)
    U_map = np.random.randn(n_rings, 4 * nside).astype(dtype)

    maps_T = {0: jnp.array(T_map[None, :, :])}
    maps_QU = {
        2: jnp.array(Q_map[None, :, :]),
        -2: jnp.array(U_map[None, :, :])
    }

    # Warmup
    alm_T = jax_map2alm(nside, l_max, (0,), maps_T)
    alm_EB = jax_map2alm(nside, l_max, (2,), maps_QU)
    jax.block_until_ready((alm_T, alm_EB))

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        alm_T = jax_map2alm(nside, l_max, (0,), maps_T)
        alm_EB = jax_map2alm(nside, l_max, (2,), maps_QU)
        # All 6 spectra: TT, TE, TB, EE, EB, BB
        cl_TT = jax_alm2cl(l_max, alm_T[0])
        cl_EE = jax_alm2cl(l_max, alm_EB[2])
        cl_BB = jax_alm2cl(l_max, alm_EB[-2])
        cl_TE = jax_alm2cl(l_max, alm_T[0], alm_EB[2])
        cl_TB = jax_alm2cl(l_max, alm_T[0], alm_EB[-2])
        cl_EB = jax_alm2cl(l_max, alm_EB[2], alm_EB[-2])
        jax.block_until_ready((cl_TT, cl_EE, cl_BB, cl_TE, cl_TB, cl_EB))
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def run_benchmark(nside, n_iterations):
    """Run full benchmark suite."""
    l_max = 3 * nside
    n_pix = 12 * nside * nside
    n_alm = (l_max + 1) * (l_max + 2) // 2

    print("=" * 80)
    print("map2cl CUDA Benchmark")
    print("=" * 80)
    print(f"  nside      = {nside}")
    print(f"  l_max      = {l_max}")
    print(f"  n_pixels   = {n_pix:,}")
    print(f"  n_alm      = {n_alm:,}")
    print(f"  iterations = {n_iterations}")
    print(f"  Phase1     = {'BLUESTEIN' if USE_BLUESTEIN else 'DFT'}")
    print("=" * 80)
    print()

    results = {}

    # CUDA benchmarks
    print("Running CUDA benchmarks...")

    for precision in ['float64', 'float32']:
        # Spin-0
        name = f'cuda_spin0_{precision}'
        print(f"  {name}...")
        try:
            mean, std = benchmark_cuda_map2cl_spin0(nside, l_max, n_iterations, precision)
            results[name] = (mean, std)
            print(f"    {mean*1000:.2f} +/- {std*1000:.2f} ms")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[name] = None

        # Spin-2
        name = f'cuda_spin2_{precision}'
        print(f"  {name}...")
        try:
            mean, std = benchmark_cuda_map2cl_spin2(nside, l_max, n_iterations, precision)
            results[name] = (mean, std)
            print(f"    {mean*1000:.2f} +/- {std*1000:.2f} ms")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[name] = None

        # Mixed T+QU
        name = f'cuda_mixed_{precision}'
        print(f"  {name}...")
        try:
            mean, std = benchmark_cuda_map2cl_mixed(nside, l_max, n_iterations, precision)
            results[name] = (mean, std)
            print(f"    {mean*1000:.2f} +/- {std*1000:.2f} ms")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[name] = None

        # Two TQU sets
        name = f'cuda_two_sets_{precision}'
        print(f"  {name}...")
        try:
            mean, std = benchmark_cuda_map2cl_two_sets(nside, l_max, n_iterations, precision)
            results[name] = (mean, std)
            print(f"    {mean*1000:.2f} +/- {std*1000:.2f} ms")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[name] = None

    print()

    # JAX benchmarks
    print("Running JAX benchmarks...")

    for precision in ['float64', 'float32']:
        # Spin-0
        name = f'jax_spin0_{precision}'
        print(f"  {name}...")
        try:
            mean, std = benchmark_jax_spin0(nside, l_max, n_iterations, precision)
            results[name] = (mean, std)
            print(f"    {mean*1000:.2f} +/- {std*1000:.2f} ms")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[name] = None

        # Spin-2
        name = f'jax_spin2_{precision}'
        print(f"  {name}...")
        try:
            mean, std = benchmark_jax_spin2(nside, l_max, n_iterations, precision)
            results[name] = (mean, std)
            print(f"    {mean*1000:.2f} +/- {std*1000:.2f} ms")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[name] = None

        # Mixed T+QU
        name = f'jax_mixed_{precision}'
        print(f"  {name}...")
        try:
            mean, std = benchmark_jax_mixed(nside, l_max, n_iterations, precision)
            results[name] = (mean, std)
            print(f"    {mean*1000:.2f} +/- {std*1000:.2f} ms")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[name] = None

    print()

    # Summary table
    print("=" * 80)
    print("Summary (CUDA uses median time for robustness to GPU throttling)")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Time (ms)':<18} {'Speedup vs JAX f64':<20}")
    print("-" * 80)

    # Group by test type
    test_types = ['spin0', 'spin2', 'mixed', 'two_sets']

    for test_type in test_types:
        # Use JAX float64 as baseline for this test type
        jax_baseline = results.get(f'jax_{test_type}_float64')
        baseline_time = jax_baseline[0] if jax_baseline else None

        # Collect all results for this test type and sort by time
        test_results = []

        # CUDA results
        for precision in ['float64', 'float32']:
            cuda_key = f'cuda_{test_type}_{precision}'
            if cuda_key in results and results[cuda_key] is not None:
                test_results.append((cuda_key, results[cuda_key]))

        # JAX results
        for precision in ['float64', 'float32']:
            jax_key = f'jax_{test_type}_{precision}'
            if jax_key in results and results[jax_key] is not None:
                test_results.append((jax_key, results[jax_key]))

        # Sort by time (fastest first)
        test_results.sort(key=lambda x: x[1][0])

        for name, (mean, std) in test_results:
            time_str = f"{mean*1000:.2f} +/- {std*1000:.2f}"
            if baseline_time:
                speedup = baseline_time / mean
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "-"
            print(f"{name:<30} {time_str:<18} {speedup_str:<20}")

        print()

    print("=" * 80)

    return results


def main():
    nside = int(sys.argv[1]) if len(sys.argv) > 1 else NSIDE
    n_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else N_ITERATIONS

    run_benchmark(nside, n_iterations)


if __name__ == "__main__":
    main()

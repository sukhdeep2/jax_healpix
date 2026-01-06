#!/usr/bin/env python3
"""Benchmark CUDA vs JAX spin-2 SPHT implementations across different precisions.

Usage:
    python benchmark_spin2.py [nside] [n_iterations]

    nside: HEALPix resolution parameter (default: 64)
    n_iterations: Number of iterations for timing (default: 2)
"""

import numpy as np
import sys
import time
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')

import jax
from spht_cuda import (SPHTCuda, set_phase1_method, PHASE1_DFT, PHASE1_BLUESTEIN)

# Default parameters
NSIDE = 64
N_ITERATIONS = 2


def create_random_qu_maps(nside, n_maps, dtype):
    """Create random Q and U polarization maps."""
    n_rings = 4 * nside - 1
    max_pix = 4 * nside

    np.random.seed(42)
    Q_map = np.random.randn(n_maps, n_rings, max_pix).astype(dtype)
    U_map = np.random.randn(n_maps, n_rings, max_pix).astype(dtype)

    return Q_map, U_map


def benchmark_jax_map2alm_spin2(nside, l_max, n_iterations, dtype_name):
    """Benchmark JAX spin-2 map2alm with specified precision."""
    if dtype_name == 'float64':
        jax.config.update("jax_enable_x64", True)
        dtype = np.float64
    elif dtype_name == 'float32':
        jax.config.update("jax_enable_x64", False)
        dtype = np.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype_name}")

    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm

    Q_map, U_map = create_random_qu_maps(nside, 1, dtype)
    maps_jax = {2: jnp.array(Q_map), -2: jnp.array(U_map)}

    # Warmup
    _ = jax_map2alm(nside, l_max, (2,), maps_jax)
    jax.block_until_ready(_)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = jax_map2alm(nside, l_max, (2,), maps_jax)
        jax.block_until_ready(result)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def benchmark_jax_alm2map_spin2(nside, l_max, n_iterations, dtype_name):
    """Benchmark JAX spin-2 alm2map with specified precision."""
    if dtype_name == 'float64':
        jax.config.update("jax_enable_x64", True)
        dtype = np.float64
    elif dtype_name == 'float32':
        jax.config.update("jax_enable_x64", False)
        dtype = np.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype_name}")

    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2map as jax_alm2map

    # Get realistic E, B alm from random Q, U maps
    Q_map, U_map = create_random_qu_maps(nside, 1, dtype)
    maps_jax = {2: jnp.array(Q_map), -2: jnp.array(U_map)}
    alm_jax = jax_map2alm(nside, l_max, (2,), maps_jax)
    jax.block_until_ready(alm_jax)

    # Warmup
    _ = jax_alm2map(nside, l_max, (2,), alm_jax)
    jax.block_until_ready(_)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = jax_alm2map(nside, l_max, (2,), alm_jax)
        jax.block_until_ready(result)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def benchmark_cuda_map2alm_spin2(nside, l_max, n_iterations, storage_precision='float64',
                                  recurrence_precision='float64', phase1_method=None):
    """Benchmark CUDA spin-2 map2alm with specified precision."""
    if phase1_method is not None:
        set_phase1_method(phase1_method)

    spht_cuda = SPHTCuda(nside, l_max, version='v6',
                         storage_precision=storage_precision,
                         recurrence_precision=recurrence_precision)

    dtype = np.float64 if storage_precision == 'float64' else np.float32
    Q_map, U_map = create_random_qu_maps(nside, 1, dtype)
    maps_cuda = np.stack([Q_map, U_map], axis=-1)

    # Warmup
    _ = spht_cuda.map2alm({2: maps_cuda}, spins=(2,))

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = spht_cuda.map2alm({2: maps_cuda}, spins=(2,))
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def benchmark_cuda_alm2map_spin2(nside, l_max, n_iterations, storage_precision='float64',
                                  recurrence_precision='float64', phase1_method=None):
    """Benchmark CUDA spin-2 alm2map with specified precision."""
    if phase1_method is not None:
        set_phase1_method(phase1_method)

    spht_cuda = SPHTCuda(nside, l_max, version='v6',
                         storage_precision=storage_precision,
                         recurrence_precision=recurrence_precision)

    dtype = np.float64 if storage_precision == 'float64' else np.float32
    Q_map, U_map = create_random_qu_maps(nside, 1, dtype)
    maps_cuda = np.stack([Q_map, U_map], axis=-1)
    alm_cuda = spht_cuda.map2alm({2: maps_cuda}, spins=(2,))

    # Warmup
    _ = spht_cuda.alm2map(alm_cuda, spins=(2,))

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = spht_cuda.alm2map(alm_cuda, spins=(2,))
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def run_benchmark(nside, n_iterations):
    """Run full benchmark suite for spin-2 transforms."""
    l_max = 3 * nside
    n_pix = 12 * nside * nside
    n_alm = (l_max + 1) * (l_max + 2) // 2

    print("=" * 80)
    print("CUDA vs JAX Spin-2 SPHT Benchmark")
    print("=" * 80)
    print(f"  nside      = {nside}")
    print(f"  l_max      = {l_max}")
    print(f"  n_pixels   = {n_pix:,}")
    print(f"  n_alm      = {n_alm:,}")
    print(f"  iterations = {n_iterations}")
    print("=" * 80)
    print()

    results_map2alm = {}
    results_alm2map = {}

    # CUDA configurations: (name, storage, recurrence, phase1_method)
    cuda_configs = [
        # DFT (default)
        ('v6_f64_f64', 'float64', 'float64', PHASE1_DFT),
        ('v6_f32_f64', 'float32', 'float64', PHASE1_DFT),
        ('v6_f32_f32', 'float32', 'float32', PHASE1_DFT),
        # Bluestein FFT
        ('v6_f64_f64_fft', 'float64', 'float64', PHASE1_BLUESTEIN),
        ('v6_f32_f32_fft', 'float32', 'float32', PHASE1_BLUESTEIN),
    ]

    # ===== map2alm benchmarks =====
    print("=== map2alm (Q,U -> E,B) ===")
    print()

    for name, storage, recurrence, phase1_method in cuda_configs:
        method_str = " + FFT" if phase1_method == PHASE1_BLUESTEIN else ""
        print(f"Benchmarking CUDA {name} ({storage} storage, {recurrence} recurrence{method_str})...")
        try:
            mean, std = benchmark_cuda_map2alm_spin2(nside, l_max, n_iterations,
                                                      storage_precision=storage,
                                                      recurrence_precision=recurrence,
                                                      phase1_method=phase1_method)
            results_map2alm[name] = (mean, std)
            print(f"  {name}: {mean*1000:.2f} ± {std*1000:.2f} ms")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            results_map2alm[name] = None

    # JAX map2alm benchmarks
    for dtype_name in ['float32', 'float64']:
        print(f"Benchmarking JAX map2alm spin-2 ({dtype_name})...")
        try:
            mean, std = benchmark_jax_map2alm_spin2(nside, l_max, n_iterations, dtype_name)
            results_map2alm[f'jax_{dtype_name}'] = (mean, std)
            print(f"  JAX {dtype_name}: {mean*1000:.2f} ± {std*1000:.2f} ms")
        except Exception as e:
            print(f"  JAX {dtype_name}: FAILED ({e})")
            results_map2alm[f'jax_{dtype_name}'] = None

    print()

    # ===== alm2map benchmarks =====
    print("=== alm2map (E,B -> Q,U) ===")
    print()

    for name, storage, recurrence, phase1_method in cuda_configs:
        method_str = " + FFT" if phase1_method == PHASE1_BLUESTEIN else ""
        print(f"Benchmarking CUDA {name} ({storage} storage, {recurrence} recurrence{method_str})...")
        try:
            mean, std = benchmark_cuda_alm2map_spin2(nside, l_max, n_iterations,
                                                      storage_precision=storage,
                                                      recurrence_precision=recurrence,
                                                      phase1_method=phase1_method)
            results_alm2map[name] = (mean, std)
            print(f"  {name}: {mean*1000:.2f} ± {std*1000:.2f} ms")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            results_alm2map[name] = None

    # JAX alm2map benchmarks
    for dtype_name in ['float32', 'float64']:
        print(f"Benchmarking JAX alm2map spin-2 ({dtype_name})...")
        try:
            mean, std = benchmark_jax_alm2map_spin2(nside, l_max, n_iterations, dtype_name)
            results_alm2map[f'jax_{dtype_name}'] = (mean, std)
            print(f"  JAX {dtype_name}: {mean*1000:.2f} ± {std*1000:.2f} ms")
        except Exception as e:
            print(f"  JAX {dtype_name}: FAILED ({e})")
            results_alm2map[f'jax_{dtype_name}'] = None

    # ===== Print summary tables =====
    print()
    print("=" * 80)
    print("Summary - map2alm Spin-2 (Q,U -> E,B)")
    print("=" * 80)
    print(f"{'Implementation':<25} {'Time (ms)':<18} {'Speedup vs JAX f64':<20}")
    print("-" * 80)

    baseline = results_map2alm.get('jax_float64')
    baseline_time = baseline[0] if baseline else None

    sorted_results = sorted(
        [(k, v) for k, v in results_map2alm.items() if v is not None],
        key=lambda x: x[1][0]
    )

    for name, result in sorted_results:
        mean, std = result
        time_str = f"{mean*1000:.2f} ± {std*1000:.2f}"
        if baseline_time:
            speedup = baseline_time / mean
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "-"
        print(f"{name:<25} {time_str:<18} {speedup_str:<20}")

    print()
    print("=" * 80)
    print("Summary - alm2map Spin-2 (E,B -> Q,U)")
    print("=" * 80)
    print(f"{'Implementation':<25} {'Time (ms)':<18} {'Speedup vs JAX f64':<20}")
    print("-" * 80)

    baseline = results_alm2map.get('jax_float64')
    baseline_time = baseline[0] if baseline else None

    sorted_results = sorted(
        [(k, v) for k, v in results_alm2map.items() if v is not None],
        key=lambda x: x[1][0]
    )

    for name, result in sorted_results:
        mean, std = result
        time_str = f"{mean*1000:.2f} ± {std*1000:.2f}"
        if baseline_time:
            speedup = baseline_time / mean
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "-"
        print(f"{name:<25} {time_str:<18} {speedup_str:<20}")

    print("=" * 80)

    # Print precision notes
    print()
    print("Precision configurations:")
    print("  v6_f64_f64:     Double precision storage + recurrence (highest accuracy)")
    print("  v6_f32_f64:     Float32 storage, double recurrence (balanced)")
    print("  v6_f32_f32:     Float32 storage + recurrence (fastest, lower accuracy)")
    print("  *_fft:          Uses Bluestein FFT for Phase 1 (faster for large nside)")
    print()

    return results_map2alm, results_alm2map


def main():
    nside = int(sys.argv[1]) if len(sys.argv) > 1 else NSIDE
    n_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else N_ITERATIONS

    run_benchmark(nside, n_iterations)


if __name__ == "__main__":
    main()

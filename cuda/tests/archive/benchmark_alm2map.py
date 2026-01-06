#!/usr/bin/env python3
"""Benchmark CUDA alm2map_v6 vs JAX alm2map across different precisions.

Usage:
    python benchmark_alm2map.py [nside] [n_iterations]

    nside: HEALPix resolution parameter (default: 64)
    n_iterations: Number of iterations for timing (default: 10)
"""

import numpy as np
import sys
import time
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')

import jax
from spht_cuda import SPHTCuda

# Default parameters
NSIDE = 64
N_ITERATIONS = 10


def create_test_alm(nside, l_max, dtype=np.float64):
    """Create test alm coefficients from a random map using JAX."""
    if dtype == np.float64:
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_enable_x64", False)

    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm

    n_rings = 4 * nside - 1

    np.random.seed(42)
    map_data = np.random.randn(1, n_rings, 4 * nside).astype(dtype)

    map_jax = {0: jnp.array(map_data)}
    alm_jax = jax_map2alm(nside, l_max, (0,), map_jax)
    jax.block_until_ready(alm_jax)

    return np.array(alm_jax[0])


def benchmark_jax_alm2map(nside, l_max, n_iterations, dtype_name):
    """Benchmark JAX alm2map with specified precision."""
    if dtype_name == 'float64':
        jax.config.update("jax_enable_x64", True)
        dtype = np.float64
    elif dtype_name == 'float32':
        jax.config.update("jax_enable_x64", False)
        dtype = np.float32
    elif dtype_name == 'bfloat16':
        jax.config.update("jax_enable_x64", False)
        dtype = np.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype_name}")

    import jax.numpy as jnp
    from SPHT_jax import alm2map as jax_alm2map

    # Create test alm
    alm_data = create_test_alm(nside, l_max, dtype)

    if dtype_name == 'bfloat16':
        alm_jax = {0: jnp.array(alm_data, dtype=jnp.bfloat16)}
    else:
        alm_jax = {0: jnp.array(alm_data)}

    # Warmup
    _ = jax_alm2map(nside, l_max, (0,), alm_jax)
    jax.block_until_ready(_)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = jax_alm2map(nside, l_max, (0,), alm_jax)
        jax.block_until_ready(result)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def benchmark_cuda_alm2map(nside, l_max, n_iterations, storage_precision='float64',
                            recurrence_precision='float64'):
    """Benchmark CUDA alm2map with specified precision."""
    spht_cuda = SPHTCuda(nside, l_max, version='v6',
                         storage_precision=storage_precision,
                         recurrence_precision=recurrence_precision)

    # Create test alm with appropriate dtype
    dtype = np.float64 if storage_precision == 'float64' else np.float32
    alm_data = create_test_alm(nside, l_max, dtype)

    if storage_precision == 'float32':
        alm_data = alm_data.astype(np.complex64)

    alm_in = {0: alm_data}

    # Warmup
    _ = spht_cuda.alm2map(alm_in, spins=(0,))

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = spht_cuda.alm2map(alm_in, spins=(0,))
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def run_benchmark(nside, n_iterations):
    """Run full benchmark suite."""
    l_max = 3 * nside
    n_pix = 12 * nside * nside
    n_alm = (l_max + 1) * (l_max + 2) // 2

    print("=" * 80)
    print("CUDA alm2map_v6 vs JAX alm2map Benchmark")
    print("=" * 80)
    print(f"  nside      = {nside}")
    print(f"  l_max      = {l_max}")
    print(f"  n_pixels   = {n_pix:,}")
    print(f"  n_alm      = {n_alm:,}")
    print(f"  iterations = {n_iterations}")
    print("=" * 80)
    print()

    results = {}

    # CUDA benchmarks with different precision configurations
    cuda_configs = [
        ('cuda_f64_f64', 'float64', 'float64'),  # Full float64
        ('cuda_f64_f32', 'float64', 'float32'),  # f64 storage, f32 recurrence
        ('cuda_f32_f64', 'float32', 'float64'),  # f32 storage, f64 recurrence
        ('cuda_f32_f32', 'float32', 'float32'),  # Full float32
    ]

    for name, storage, recurrence in cuda_configs:
        print(f"Benchmarking CUDA alm2map ({storage} storage, {recurrence} recurrence)...")
        try:
            mean, std = benchmark_cuda_alm2map(nside, l_max, n_iterations,
                                                storage_precision=storage,
                                                recurrence_precision=recurrence)
            results[name] = (mean, std)
            print(f"  {name}: {mean*1000:.2f} ± {std*1000:.2f} ms")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            results[name] = None

    # JAX benchmarks
    for dtype_name in ['bfloat16', 'float32', 'float64']:
        print(f"Benchmarking JAX alm2map ({dtype_name})...")
        try:
            jax_mean, jax_std = benchmark_jax_alm2map(nside, l_max, n_iterations, dtype_name)
            results[f'jax_{dtype_name}'] = (jax_mean, jax_std)
            print(f"  JAX {dtype_name}: {jax_mean*1000:.2f} ± {jax_std*1000:.2f} ms")
        except Exception as e:
            print(f"  JAX {dtype_name}: FAILED ({e})")
            results[f'jax_{dtype_name}'] = None

    # Print summary table
    print()
    print("=" * 80)
    print("Summary - alm2map Performance")
    print("=" * 80)
    print(f"{'Implementation':<20} {'Time (ms)':<18} {'Speedup vs JAX f64':<20}")
    print("-" * 80)

    # Get JAX float64 as baseline
    baseline = results.get('jax_float64')
    baseline_time = baseline[0] if baseline else None

    # Sort by time for nice display
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v is not None],
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
        print(f"{name:<20} {time_str:<18} {speedup_str:<20}")

    # Print failed ones at the end
    for name, result in results.items():
        if result is None:
            print(f"{name:<20} {'FAILED':<18} {'-':<20}")

    print("=" * 80)

    # Print precision notes
    print()
    print("Precision configurations:")
    print("  cuda_f64_f64: Highest accuracy (double for everything)")
    print("  cuda_f64_f32: Double storage, float32 Ylm recurrence")
    print("  cuda_f32_f64: Float32 storage, double Ylm recurrence (balanced)")
    print("  cuda_f32_f32: Fastest, lowest accuracy (may have issues at high l)")
    print()

    return results


def run_scaling_benchmark(n_iterations=5):
    """Run benchmark across different nside values to show scaling."""
    print("=" * 80)
    print("Scaling Benchmark - alm2map across different nside")
    print("=" * 80)

    nside_values = [16, 32, 64, 128, 256]

    print(f"\n{'nside':<8} {'CUDA f64_f64 (ms)':<20} {'JAX f64 (ms)':<20} {'Speedup':<10}")
    print("-" * 60)

    for nside in nside_values:
        l_max = 3 * nside

        try:
            cuda_mean, cuda_std = benchmark_cuda_alm2map(nside, l_max, n_iterations,
                                                          storage_precision='float64',
                                                          recurrence_precision='float64')
        except Exception as e:
            print(f"{nside:<8} CUDA FAILED: {e}")
            continue

        try:
            jax_mean, jax_std = benchmark_jax_alm2map(nside, l_max, n_iterations, 'float64')
        except Exception as e:
            print(f"{nside:<8} JAX FAILED: {e}")
            continue

        speedup = jax_mean / cuda_mean
        print(f"{nside:<8} {cuda_mean*1000:>8.2f} ± {cuda_std*1000:<8.2f} "
              f"{jax_mean*1000:>8.2f} ± {jax_std*1000:<8.2f} {speedup:>8.2f}x")

    print("=" * 80)


def main():
    nside = int(sys.argv[1]) if len(sys.argv) > 1 else NSIDE
    n_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else N_ITERATIONS

    run_benchmark(nside, n_iterations)

    if len(sys.argv) <= 1:
        # Also run scaling benchmark if using defaults
        print("\n")
        run_scaling_benchmark()


if __name__ == "__main__":
    main()

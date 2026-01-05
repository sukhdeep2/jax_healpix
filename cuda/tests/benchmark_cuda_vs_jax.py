#!/usr/bin/env python3
"""Benchmark CUDA vs JAX SPHT implementations across different precisions.

Usage:
    python benchmark_cuda_vs_jax.py [nside] [n_iterations]

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


def benchmark_jax(nside, l_max, n_iterations, dtype_name):
    """Benchmark JAX map2alm with specified precision."""
    # Set JAX precision
    if dtype_name == 'float64':
        jax.config.update("jax_enable_x64", True)
        dtype = np.float64
    elif dtype_name == 'float32':
        jax.config.update("jax_enable_x64", False)
        dtype = np.float32
    elif dtype_name == 'bfloat16':
        jax.config.update("jax_enable_x64", False)
        dtype = np.float32  # Input as float32, will convert to bfloat16
    else:
        raise ValueError(f"Unknown dtype: {dtype_name}")

    # Import after setting config
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm

    n_rings = 4 * nside - 1

    # Create random map
    np.random.seed(42)
    map_data = np.random.randn(1, n_rings, 4 * nside).astype(dtype)

    if dtype_name == 'bfloat16':
        map_jax = {0: jnp.array(map_data, dtype=jnp.bfloat16)}
    else:
        map_jax = {0: jnp.array(map_data)}

    # Warmup
    _ = jax_map2alm(nside, l_max, (0,), map_jax)
    jax.block_until_ready(_)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = jax_map2alm(nside, l_max, (0,), map_jax)
        jax.block_until_ready(result)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def benchmark_cuda(nside, l_max, n_iterations):
    """Benchmark CUDA map2alm (always float64)."""
    spht_cuda = SPHTCuda(nside, l_max)

    n_rings = 4 * nside - 1

    # Create random map
    np.random.seed(42)
    map_data = np.random.randn(1, n_rings, 4 * nside).astype(np.float64)

    # Warmup
    _ = spht_cuda.map2alm({0: map_data}, spins=(0,))

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = spht_cuda.map2alm({0: map_data}, spins=(0,))
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def run_benchmark(nside, n_iterations):
    """Run full benchmark suite."""
    l_max = 3 * nside
    n_pix = 12 * nside * nside
    n_alm = (l_max + 1) * (l_max + 2) // 2

    print("=" * 70)
    print("CUDA vs JAX SPHT Benchmark")
    print("=" * 70)
    print(f"  nside      = {nside}")
    print(f"  l_max      = {l_max}")
    print(f"  n_pixels   = {n_pix:,}")
    print(f"  n_alm      = {n_alm:,}")
    print(f"  iterations = {n_iterations}")
    print("=" * 70)
    print()

    results = {}

    # CUDA benchmark (float64 only)
    print("Benchmarking CUDA (float64)...")
    try:
        cuda_mean, cuda_std = benchmark_cuda(nside, l_max, n_iterations)
        results['cuda_f64'] = (cuda_mean, cuda_std)
        print(f"  CUDA float64: {cuda_mean*1000:.2f} ± {cuda_std*1000:.2f} ms")
    except Exception as e:
        print(f"  CUDA float64: FAILED ({e})")
        results['cuda_f64'] = None

    # JAX benchmarks
    for dtype_name in ['bfloat16', 'float32', 'float64']:
        print(f"Benchmarking JAX ({dtype_name})...")
        try:
            jax_mean, jax_std = benchmark_jax(nside, l_max, n_iterations, dtype_name)
            results[f'jax_{dtype_name}'] = (jax_mean, jax_std)
            print(f"  JAX {dtype_name}: {jax_mean*1000:.2f} ± {jax_std*1000:.2f} ms")
        except Exception as e:
            print(f"  JAX {dtype_name}: FAILED ({e})")
            results[f'jax_{dtype_name}'] = None

    # Print summary table
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Implementation':<20} {'Time (ms)':<15} {'Speedup vs JAX f64':<20}")
    print("-" * 70)

    # Get JAX float64 as baseline
    baseline = results.get('jax_float64')
    baseline_time = baseline[0] if baseline else None

    for name, result in results.items():
        if result is None:
            print(f"{name:<20} {'FAILED':<15} {'-':<20}")
        else:
            mean, std = result
            time_str = f"{mean*1000:.2f} ± {std*1000:.2f}"
            if baseline_time:
                speedup = baseline_time / mean
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "-"
            print(f"{name:<20} {time_str:<15} {speedup_str:<20}")

    print("=" * 70)

    return results


def main():
    nside = int(sys.argv[1]) if len(sys.argv) > 1 else NSIDE
    n_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else N_ITERATIONS

    run_benchmark(nside, n_iterations)


if __name__ == "__main__":
    main()

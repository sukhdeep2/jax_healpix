#!/usr/bin/env python3
"""Benchmark multi-map performance for CUDA SPHT v6.

This benchmark measures the performance benefit of processing multiple maps
in parallel, where Ylm is computed once and reused across all maps.

Usage:
    python benchmark_multimap.py [nside] [n_iterations]

    nside: HEALPix resolution parameter (default: 256)
    n_iterations: Number of iterations for timing (default: 5)
"""

import numpy as np
import sys
import time
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')

from spht_cuda import SPHTCuda


def benchmark_multimap(nside, l_max, n_maps, n_iterations, precision='float64'):
    """Benchmark CUDA v6 map2alm with multiple maps.

    Args:
        nside: HEALPix nside parameter
        l_max: Maximum multipole
        n_maps: Number of maps to process in parallel
        n_iterations: Number of timing iterations
        precision: 'float64' or 'float32'

    Returns:
        (total_mean, total_std, per_map_mean): Timing results in seconds
    """
    storage = precision
    recurrence = 'float64'  # Always use f64 recurrence for accuracy

    spht_cuda = SPHTCuda(nside, l_max, version='v6',
                         storage_precision=storage,
                         recurrence_precision=recurrence)

    n_rings = 4 * nside - 1
    dtype = np.float32 if precision == 'float32' else np.float64

    # Create random map array [n_maps, n_rings, 4*nside]
    np.random.seed(42)
    map_array = np.random.randn(n_maps, n_rings, 4 * nside).astype(dtype)

    # Warmup
    _ = spht_cuda.map2alm({0: map_array}, spins=(0,))

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = spht_cuda.map2alm({0: map_array}, spins=(0,))
        end = time.perf_counter()
        times.append(end - start)

    total_mean = np.mean(times)
    total_std = np.std(times)
    per_map_mean = total_mean / n_maps

    return total_mean, total_std, per_map_mean


def run_multimap_benchmark(nside, n_iterations):
    """Run multi-map benchmark suite."""
    l_max = 3 * nside
    n_pix = 12 * nside * nside

    print("=" * 70)
    print("Multi-Map Performance Benchmark (CUDA v6)")
    print("=" * 70)
    print(f"  nside      = {nside}")
    print(f"  l_max      = {l_max}")
    print(f"  n_pixels   = {n_pix:,}")
    print(f"  iterations = {n_iterations}")
    print()
    print("Ylm is computed once and reused for all maps in parallel.")
    print("Max parallel maps: 5 (f64) or 11 (f32) due to shared memory limits.")
    print("=" * 70)
    print()

    # Test configurations
    map_counts = [1, 2, 3, 5, 10, 20]

    # Float64 benchmark
    print("Float64 storage, Float64 recurrence:")
    print("-" * 70)
    print(f"{'n_maps':>8} {'Total (ms)':>14} {'Per-map (ms)':>14} {'Speedup':>10}")
    print("-" * 70)

    baseline_per_map = None
    for n_maps in map_counts:
        try:
            total, std, per_map = benchmark_multimap(nside, l_max, n_maps,
                                                      n_iterations, 'float64')
            if baseline_per_map is None:
                baseline_per_map = per_map
            speedup = baseline_per_map / per_map

            print(f"{n_maps:>8} {total*1000:>10.2f} +/- {std*1000:>4.2f} "
                  f"{per_map*1000:>14.2f} {speedup:>10.2f}x")
        except Exception as e:
            print(f"{n_maps:>8} FAILED: {e}")

    print()

    # Float32 benchmark
    print("Float32 storage, Float64 recurrence:")
    print("-" * 70)
    print(f"{'n_maps':>8} {'Total (ms)':>14} {'Per-map (ms)':>14} {'Speedup':>10}")
    print("-" * 70)

    baseline_per_map = None
    for n_maps in map_counts:
        try:
            total, std, per_map = benchmark_multimap(nside, l_max, n_maps,
                                                      n_iterations, 'float32')
            if baseline_per_map is None:
                baseline_per_map = per_map
            speedup = baseline_per_map / per_map

            print(f"{n_maps:>8} {total*1000:>10.2f} +/- {std*1000:>4.2f} "
                  f"{per_map*1000:>14.2f} {speedup:>10.2f}x")
        except Exception as e:
            print(f"{n_maps:>8} FAILED: {e}")

    print()
    print("=" * 70)
    print("Notes:")
    print("  - Speedup is relative to single-map (n_maps=1) per-map time")
    print("  - For n_maps > max_parallel, maps are processed in batches")
    print("  - Each batch still benefits from Ylm reuse within the batch")
    print("=" * 70)


def main():
    nside = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    n_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    run_multimap_benchmark(nside, n_iterations)


if __name__ == "__main__":
    main()

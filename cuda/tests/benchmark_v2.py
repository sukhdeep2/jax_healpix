#!/usr/bin/env python3
"""
Benchmark script comparing map2alm v1 (original) vs v2 (cuFFT/cuBLAS optimized)
"""

import sys
import os
import time
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from spht_cuda import SPHTCuda


def benchmark_map2alm(nside: int, l_max: int = None, n_warmup: int = 2, n_runs: int = 5):
    """
    Benchmark map2alm for both v1 and v2 implementations.

    Args:
        nside: HEALPix nside parameter
        l_max: Maximum l value (default: 3*nside)
        n_warmup: Number of warmup runs
        n_runs: Number of timed runs

    Returns:
        Tuple of (time_v1, time_v2) in milliseconds
    """
    if l_max is None:
        l_max = 3 * nside

    n_rings = 4 * nside - 1

    # Create random test map
    np.random.seed(42)
    test_map = np.random.randn(1, n_rings, 4 * nside).astype(np.float64)

    # Initialize both versions
    spht_v1 = SPHTCuda(nside, l_max, use_v2=False)
    spht_v2 = SPHTCuda(nside, l_max, use_v2=True)

    # Warmup v1
    for _ in range(n_warmup):
        _ = spht_v1.map2alm({0: test_map}, spins=(0,))

    # Time v1
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = spht_v1.map2alm({0: test_map}, spins=(0,))
    end = time.perf_counter()
    time_v1 = (end - start) / n_runs * 1000  # ms

    # Warmup v2
    for _ in range(n_warmup):
        _ = spht_v2.map2alm({0: test_map}, spins=(0,))

    # Time v2
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = spht_v2.map2alm({0: test_map}, spins=(0,))
    end = time.perf_counter()
    time_v2 = (end - start) / n_runs * 1000  # ms

    return time_v1, time_v2


def verify_accuracy(nside: int = 64, l_max: int = None):
    """
    Verify that v1 and v2 produce the same results.

    Args:
        nside: HEALPix nside parameter
        l_max: Maximum l value

    Returns:
        True if results match, False otherwise
    """
    if l_max is None:
        l_max = 3 * nside

    n_rings = 4 * nside - 1

    # Create random test map
    np.random.seed(42)
    test_map = np.random.randn(1, n_rings, 4 * nside).astype(np.float64)

    # Run both versions
    spht_v1 = SPHTCuda(nside, l_max, use_v2=False)
    spht_v2 = SPHTCuda(nside, l_max, use_v2=True)

    alm_v1 = spht_v1.map2alm({0: test_map}, spins=(0,))[0]
    alm_v2 = spht_v2.map2alm({0: test_map}, spins=(0,))[0]

    # Compare
    max_abs_error = np.max(np.abs(alm_v1 - alm_v2))
    max_rel_error = np.max(np.abs(alm_v1 - alm_v2) / (np.abs(alm_v1) + 1e-15))

    print(f"  Max absolute error: {max_abs_error:.6e}")
    print(f"  Max relative error: {max_rel_error:.6e}")

    return max_rel_error < 0.01


def main():
    print("=" * 60)
    print("map2alm Benchmark: v1 (original) vs v2 (cuFFT/cuBLAS)")
    print("=" * 60)

    # Test configurations
    nsides = [64, 128, 256, 512]

    # Verify accuracy first
    print("\nVerifying accuracy (nside=64)...")
    if verify_accuracy(64):
        print("  PASSED: v1 and v2 produce matching results")
    else:
        print("  FAILED: Results do not match!")
        return 1

    # Run benchmarks
    print("\nBenchmark results:")
    print("-" * 60)
    print(f"{'nside':>8} | {'l_max':>8} | {'v1 (ms)':>10} | {'v2 (ms)':>10} | {'Speedup':>10}")
    print("-" * 60)

    for nside in nsides:
        l_max = 3 * nside

        try:
            time_v1, time_v2 = benchmark_map2alm(nside, l_max)
            speedup = time_v1 / time_v2

            print(f"{nside:>8} | {l_max:>8} | {time_v1:>10.2f} | {time_v2:>10.2f} | {speedup:>9.2f}x")
        except Exception as e:
            print(f"{nside:>8} | {l_max:>8} | ERROR: {e}")

    print("-" * 60)
    print("\nBenchmark complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Benchmark CUDA vs JAX spin-0 alm2map implementations.

Usage:
    python benchmark_alm2map_spin0.py [nside] [n_iterations]

    nside: HEALPix resolution parameter (default: 512)
    n_iterations: Number of iterations for timing (default: 10)
"""

import numpy as np
import sys
import time
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')

import jax
from spht_cuda import SPHTCuda, set_phase1_method, PHASE1_DFT, PHASE1_FFT_EQUATORIAL, PHASE1_BLUESTEIN

# Default parameters
NSIDE = 512
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
    """Benchmark JAX spin-0 alm2map."""
    if dtype_name == 'float64':
        jax.config.update("jax_enable_x64", True)
        dtype = np.float64
    else:
        jax.config.update("jax_enable_x64", False)
        dtype = np.float32

    import jax.numpy as jnp
    from SPHT_jax import alm2map as jax_alm2map

    alm_data = create_test_alm(nside, l_max, dtype)
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
        times.append(time.perf_counter() - start)

    return np.median(times), np.std(times)


def benchmark_cuda_alm2map(nside, l_max, n_iterations, storage='float64',
                           recurrence='float64', phase1_method=PHASE1_DFT):
    """Benchmark CUDA spin-0 alm2map."""
    set_phase1_method(phase1_method)

    spht = SPHTCuda(nside, l_max, version='v6',
                    storage_precision=storage,
                    recurrence_precision=recurrence)

    dtype = np.float64 if storage == 'float64' else np.float32
    alm_data = create_test_alm(nside, l_max, dtype)
    if storage == 'float32':
        alm_data = alm_data.astype(np.complex64)
    alm_in = {0: alm_data}

    # Warmup
    _ = spht.alm2map(alm_in, spins=(0,))

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = spht.alm2map(alm_in, spins=(0,))
        times.append(time.perf_counter() - start)

    return np.median(times), np.std(times)


def run_benchmark(nside, n_iterations):
    """Run full spin-0 alm2map benchmark suite."""
    l_max = 3 * nside
    n_pix = 12 * nside * nside
    n_alm = (l_max + 1) * (l_max + 2) // 2

    print("=" * 80)
    print("Spin-0 alm2map Benchmark: CUDA vs JAX")
    print("=" * 80)
    print(f"  nside      = {nside}")
    print(f"  l_max      = {l_max}")
    print(f"  n_pixels   = {n_pix:,}")
    print(f"  n_alm      = {n_alm:,}")
    print(f"  iterations = {n_iterations}")
    print("=" * 80)
    print()

    results = {}

    # CUDA benchmarks
    # Note: For alm2map (inverse transform), only DFT and BLUESTEIN apply.
    # FFT_EQUATORIAL is only for map2alm forward transforms.
    cuda_configs = [
        ('cuda_f64_dft', 'float64', 'float64', PHASE1_DFT),
        ('cuda_f64_bluestein', 'float64', 'float64', PHASE1_BLUESTEIN),
        ('cuda_f32_dft', 'float32', 'float32', PHASE1_DFT),
        ('cuda_f32_bluestein', 'float32', 'float32', PHASE1_BLUESTEIN),
    ]

    print("Running CUDA benchmarks...")
    for name, storage, recurrence, phase1 in cuda_configs:
        try:
            mean, std = benchmark_cuda_alm2map(nside, l_max, n_iterations,
                                               storage, recurrence, phase1)
            results[name] = (mean, std)
            print(f"  {name}: {mean*1000:.2f} +/- {std*1000:.2f} ms")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            results[name] = None

    # JAX benchmarks
    print("\nRunning JAX benchmarks...")
    for dtype_name in ['float32', 'float64']:
        try:
            mean, std = benchmark_jax_alm2map(nside, l_max, n_iterations, dtype_name)
            results[f'jax_{dtype_name}'] = (mean, std)
            print(f"  jax_{dtype_name}: {mean*1000:.2f} +/- {std*1000:.2f} ms")
        except Exception as e:
            print(f"  jax_{dtype_name}: FAILED ({e})")
            results[f'jax_{dtype_name}'] = None

    # Print summary
    print()
    print("=" * 80)
    print("Summary - Spin-0 alm2map")
    print("=" * 80)
    print(f"{'Configuration':<25} {'Time (ms)':<20} {'Speedup vs JAX f64':<20}")
    print("-" * 80)

    baseline = results.get('jax_float64')
    baseline_time = baseline[0] if baseline else None

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v is not None],
        key=lambda x: x[1][0]
    )

    for name, (mean, std) in sorted_results:
        time_str = f"{mean*1000:.2f} +/- {std*1000:.2f}"
        speedup_str = f"{baseline_time/mean:.2f}x" if baseline_time else "-"
        print(f"{name:<25} {time_str:<20} {speedup_str:<20}")

    print("=" * 80)
    return results


def run_scaling_benchmark(n_iterations=5):
    """Run benchmark across different nside values."""
    print("\n" + "=" * 80)
    print("Scaling Benchmark - Spin-0 alm2map")
    print("=" * 80)

    nside_values = [64, 128, 256, 512, 1024]

    print(f"\n{'nside':<8} {'CUDA f64 (ms)':<18} {'CUDA f32 (ms)':<18} {'JAX f64 (ms)':<18} {'Speedup f64':<12}")
    print("-" * 80)

    for nside in nside_values:
        l_max = 3 * nside
        row = f"{nside:<8} "

        try:
            cuda_f64, _ = benchmark_cuda_alm2map(nside, l_max, n_iterations, 'float64', 'float64', PHASE1_DFT)
            row += f"{cuda_f64*1000:>8.2f}          "
        except:
            row += f"{'FAILED':<18}"
            cuda_f64 = None

        try:
            cuda_f32, _ = benchmark_cuda_alm2map(nside, l_max, n_iterations, 'float32', 'float32', PHASE1_DFT)
            row += f"{cuda_f32*1000:>8.2f}          "
        except:
            row += f"{'FAILED':<18}"

        try:
            jax_f64, _ = benchmark_jax_alm2map(nside, l_max, n_iterations, 'float64')
            row += f"{jax_f64*1000:>8.2f}          "
            if cuda_f64:
                row += f"{jax_f64/cuda_f64:>8.2f}x"
        except:
            row += f"{'FAILED':<18}"

        print(row)

    print("=" * 80)


def main():
    nside = int(sys.argv[1]) if len(sys.argv) > 1 else NSIDE
    n_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else N_ITERATIONS

    run_benchmark(nside, n_iterations)

    if '--scaling' in sys.argv or len(sys.argv) <= 1:
        run_scaling_benchmark()


if __name__ == "__main__":
    main()

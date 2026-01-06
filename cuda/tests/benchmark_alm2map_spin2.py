#!/usr/bin/env python3
"""Benchmark CUDA vs JAX spin-2 alm2map implementations.

Usage:
    python benchmark_alm2map_spin2.py [nside] [n_iterations]

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


def create_test_eb_alm(nside, l_max, dtype=np.float64):
    """Create test E,B alm coefficients from random Q,U maps using JAX.

    Returns:
        alm_packed: Shape [n_maps, l_max+1, l_max+1, 2] for CUDA (E,B in last dim)
        alm_E, alm_B: Separate arrays for JAX
    """
    if dtype == np.float64:
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_enable_x64", False)

    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm

    n_rings = 4 * nside - 1
    np.random.seed(42)
    Q_map = np.random.randn(1, n_rings, 4 * nside).astype(dtype)
    U_map = np.random.randn(1, n_rings, 4 * nside).astype(dtype)

    maps_jax = {2: jnp.array(Q_map), -2: jnp.array(U_map)}
    alm_jax = jax_map2alm(nside, l_max, (2,), maps_jax)
    jax.block_until_ready(alm_jax)

    alm_E = np.array(alm_jax[2])
    alm_B = np.array(alm_jax[-2])
    # Pack E,B into last dimension for CUDA: [n_maps, l_max+1, l_max+1, 2]
    alm_packed = np.stack([alm_E, alm_B], axis=-1)

    return alm_packed, alm_E, alm_B


def benchmark_jax_alm2map_spin2(nside, l_max, n_iterations, dtype_name):
    """Benchmark JAX spin-2 alm2map."""
    if dtype_name == 'float64':
        jax.config.update("jax_enable_x64", True)
        dtype = np.float64
    else:
        jax.config.update("jax_enable_x64", False)
        dtype = np.float32

    import jax.numpy as jnp
    from SPHT_jax import alm2map as jax_alm2map

    _, alm_E, alm_B = create_test_eb_alm(nside, l_max, dtype)
    alm_jax = {2: jnp.array(alm_E), -2: jnp.array(alm_B)}

    # Warmup
    _ = jax_alm2map(nside, l_max, (2,), alm_jax)
    jax.block_until_ready(_)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = jax_alm2map(nside, l_max, (2,), alm_jax)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)

    return np.median(times), np.std(times)


def benchmark_cuda_alm2map_spin2(nside, l_max, n_iterations, storage='float64',
                                  recurrence='float64', phase1_method=PHASE1_DFT):
    """Benchmark CUDA spin-2 alm2map."""
    set_phase1_method(phase1_method)

    spht = SPHTCuda(nside, l_max, version='v6',
                    storage_precision=storage,
                    recurrence_precision=recurrence)

    dtype = np.float64 if storage == 'float64' else np.float32
    alm_packed, _, _ = create_test_eb_alm(nside, l_max, dtype)
    if storage == 'float32':
        alm_packed = alm_packed.astype(np.complex64)
    # CUDA expects {2: array} with shape [n_maps, l_max+1, l_max+1, 2]
    alm_in = {2: alm_packed}

    # Warmup
    _ = spht.alm2map(alm_in, spins=(2,))

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = spht.alm2map(alm_in, spins=(2,))
        times.append(time.perf_counter() - start)

    return np.median(times), np.std(times)


def run_benchmark(nside, n_iterations):
    """Run full spin-2 alm2map benchmark suite."""
    l_max = 3 * nside
    n_pix = 12 * nside * nside
    n_alm = (l_max + 1) * (l_max + 2) // 2

    print("=" * 80)
    print("Spin-2 alm2map Benchmark: CUDA vs JAX")
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
            mean, std = benchmark_cuda_alm2map_spin2(nside, l_max, n_iterations,
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
            mean, std = benchmark_jax_alm2map_spin2(nside, l_max, n_iterations, dtype_name)
            results[f'jax_{dtype_name}'] = (mean, std)
            print(f"  jax_{dtype_name}: {mean*1000:.2f} +/- {std*1000:.2f} ms")
        except Exception as e:
            print(f"  jax_{dtype_name}: FAILED ({e})")
            results[f'jax_{dtype_name}'] = None

    # Print summary
    print()
    print("=" * 80)
    print("Summary - Spin-2 alm2map")
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
    print("Scaling Benchmark - Spin-2 alm2map")
    print("=" * 80)

    nside_values = [64, 128, 256, 512, 1024]

    print(f"\n{'nside':<8} {'CUDA f64 (ms)':<18} {'CUDA f32 (ms)':<18} {'JAX f64 (ms)':<18} {'Speedup f64':<12}")
    print("-" * 80)

    for nside in nside_values:
        l_max = 3 * nside
        row = f"{nside:<8} "

        try:
            cuda_f64, _ = benchmark_cuda_alm2map_spin2(nside, l_max, n_iterations, 'float64', 'float64', PHASE1_DFT)
            row += f"{cuda_f64*1000:>8.2f}          "
        except:
            row += f"{'FAILED':<18}"
            cuda_f64 = None

        try:
            cuda_f32, _ = benchmark_cuda_alm2map_spin2(nside, l_max, n_iterations, 'float32', 'float32', PHASE1_DFT)
            row += f"{cuda_f32*1000:>8.2f}          "
        except:
            row += f"{'FAILED':<18}"

        try:
            jax_f64, _ = benchmark_jax_alm2map_spin2(nside, l_max, n_iterations, 'float64')
            row += f"{jax_f64*1000:>8.2f}          "
            if cuda_f64:
                row += f"{jax_f64/cuda_f64:>8.2f}x"
        except:
            row += f"{'FAILED':<18}"

        print(row)

    print("=" * 80)


def run_comparison_with_spin0(nside, n_iterations):
    """Compare spin-2 vs spin-0 performance."""
    from benchmark_alm2map_spin0 import benchmark_cuda_alm2map

    l_max = 3 * nside

    print("\n" + "=" * 80)
    print("Spin-2 vs Spin-0 Comparison - alm2map")
    print("=" * 80)
    print(f"  nside = {nside}, l_max = {l_max}")
    print("-" * 80)

    for precision in ['float32', 'float64']:
        print(f"\n{precision}:")
        try:
            spin0_time, _ = benchmark_cuda_alm2map(nside, l_max, n_iterations, precision, precision, PHASE1_DFT)
            spin2_time, _ = benchmark_cuda_alm2map_spin2(nside, l_max, n_iterations, precision, precision, PHASE1_DFT)
            ratio = spin2_time / spin0_time
            print(f"  Spin-0: {spin0_time*1000:.2f} ms")
            print(f"  Spin-2: {spin2_time*1000:.2f} ms")
            print(f"  Ratio (spin2/spin0): {ratio:.2f}x")
        except Exception as e:
            print(f"  Error: {e}")

    print("=" * 80)


def main():
    nside = int(sys.argv[1]) if len(sys.argv) > 1 else NSIDE
    n_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else N_ITERATIONS

    run_benchmark(nside, n_iterations)

    if '--scaling' in sys.argv or len(sys.argv) <= 1:
        run_scaling_benchmark()

    if '--compare' in sys.argv:
        run_comparison_with_spin0(nside, n_iterations)


if __name__ == "__main__":
    main()

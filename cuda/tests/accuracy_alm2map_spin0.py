#!/usr/bin/env python3
"""Accuracy test for CUDA spin-0 alm2map vs JAX reference.

Usage:
    python accuracy_alm2map_spin0.py [nside] [l_max]
"""

import numpy as np
import sys
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from SPHT_jax import map2alm as jax_map2alm, alm2map as jax_alm2map
from spht_cuda import SPHTCuda, set_phase1_method, PHASE1_DFT, PHASE1_BLUESTEIN

# Default parameters
NSIDE = 64
THRESHOLD = 1e-10


def create_test_alm(nside, l_max, dtype=np.float64):
    """Create synthetic bandlimited test alm.

    Creates random alm only for l <= nside to avoid aliasing and ensure
    accurate comparison. Uses complex128 for generation, returns in dtype.
    """
    np.random.seed(42)
    # Create alm array with shape [n_maps, l_max+1, l_max+1]
    alm = np.zeros((1, l_max + 1, l_max + 1), dtype=np.complex128)

    # Fill only l <= nside with random values
    for l in range(nside + 1):
        # m=0 is real
        alm[0, l, 0] = np.random.randn()
        # m>0 are complex
        for m in range(1, l + 1):
            alm[0, l, m] = np.random.randn() + 1j * np.random.randn()

    return alm


def compute_relative_diff(cuda_map, jax_map):
    """Compute relative difference between CUDA and JAX maps."""
    diff = np.abs(cuda_map - jax_map)
    ref = np.abs(jax_map)
    mask = ref > 1e-15
    rel_diff = np.zeros_like(diff)
    rel_diff[mask] = diff[mask] / ref[mask]
    return np.max(rel_diff[mask]), np.mean(rel_diff[mask])


def compute_regional_diff(cuda_map, jax_map, nside):
    """Compute relative difference by region (polar vs equatorial).

    Returns dict with errors for:
    - north_polar: rings 0 to nside-1
    - equatorial: rings nside to 3*nside-2
    - south_polar: rings 3*nside-1 to end
    """
    n_rings = 4 * nside - 1
    diff = np.abs(cuda_map - jax_map)
    ref = np.abs(jax_map)

    regions = {
        'north_polar': (0, nside),
        'equatorial': (nside, 3 * nside - 1),
        'south_polar': (3 * nside - 1, n_rings)
    }

    results = {}
    for name, (start, end) in regions.items():
        region_diff = diff[start:end]
        region_ref = ref[start:end]
        mask = region_ref > 1e-15
        if mask.any():
            rel_diff = region_diff[mask] / region_ref[mask]
            results[name] = (np.max(rel_diff), np.mean(rel_diff))
        else:
            results[name] = (0, 0)

    return results


def test_cuda_vs_jax(nside, l_max, storage='float64', recurrence='float64',
                     phase1_method=PHASE1_DFT):
    """Test CUDA alm2map against JAX reference."""
    set_phase1_method(phase1_method)

    # Create test alm
    alm_data = create_test_alm(nside, l_max)

    # JAX computation
    alm_jax = {0: jnp.array(alm_data)}
    map_jax = jax_alm2map(nside, l_max, (0,), alm_jax)
    jax.block_until_ready(map_jax)
    map_jax_np = np.array(map_jax[0][0])

    # CUDA computation
    spht = SPHTCuda(nside, l_max, version='v6',
                    storage_precision=storage,
                    recurrence_precision=recurrence)

    dtype = np.float64 if storage == 'float64' else np.float32
    alm_cuda_in = {0: alm_data.astype(np.complex128 if storage == 'float64' else np.complex64)}
    map_cuda = spht.alm2map(alm_cuda_in, spins=(0,))
    map_cuda_np = map_cuda[0][0]

    # Compute difference (overall and by region)
    max_diff, mean_diff = compute_relative_diff(map_cuda_np, map_jax_np)
    regional = compute_regional_diff(map_cuda_np, map_jax_np, nside)

    return max_diff, mean_diff, regional, map_cuda_np, map_jax_np


def run_accuracy_tests(nside, l_max):
    """Run full accuracy test suite."""
    print("=" * 80)
    print("Spin-0 alm2map Accuracy Test: CUDA vs JAX")
    print("=" * 80)
    print(f"  nside     = {nside}")
    print(f"  l_max     = {l_max}")
    print(f"  threshold = {THRESHOLD:.0e}")
    print("=" * 80)
    print()

    # Note: For alm2map (inverse transform), only DFT and BLUESTEIN apply.
    # FFT_EQUATORIAL is only for map2alm forward transforms.
    configs = [
        ('f64_f64_dft', 'float64', 'float64', PHASE1_DFT),
        ('f64_f64_bluestein', 'float64', 'float64', PHASE1_BLUESTEIN),
        ('f32_f32_dft', 'float32', 'float32', PHASE1_DFT),
        ('f32_f32_bluestein', 'float32', 'float32', PHASE1_BLUESTEIN),
    ]

    all_pass = True

    # Overall accuracy table
    print(f"{'Configuration':<25} {'Max Rel Diff':<15} {'Mean Rel Diff':<15} {'Status':<10}")
    print("-" * 80)

    for name, storage, recurrence, phase1 in configs:
        try:
            max_diff, mean_diff, regional, cuda_map, jax_map = test_cuda_vs_jax(
                nside, l_max, storage, recurrence, phase1
            )

            threshold = THRESHOLD if storage == 'float64' else 1e-5
            passed = max_diff < threshold
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False

            print(f"{name:<25} {max_diff:<15.2e} {mean_diff:<15.2e} {status:<10}")

        except Exception as e:
            print(f"{name:<25} {'ERROR':<15} {str(e)[:20]:<15} {'FAIL':<10}")
            all_pass = False

    # Regional accuracy table
    print()
    print("Regional Accuracy (Max Rel Diff by region):")
    print("-" * 100)
    print(f"{'Configuration':<25} {'North Polar':<15} {'Equatorial':<15} {'South Polar':<15} {'Eq Status':<10}")
    print("-" * 100)

    for name, storage, recurrence, phase1 in configs:
        try:
            max_diff, mean_diff, regional, cuda_map, jax_map = test_cuda_vs_jax(
                nside, l_max, storage, recurrence, phase1
            )

            north_max = regional['north_polar'][0]
            eq_max = regional['equatorial'][0]
            south_max = regional['south_polar'][0]

            # Check equatorial region for pass/fail (poles may have numerical issues)
            threshold = THRESHOLD if storage == 'float64' else 1e-5
            eq_passed = eq_max < threshold
            eq_status = "PASS" if eq_passed else "FAIL"

            print(f"{name:<25} {north_max:<15.2e} {eq_max:<15.2e} {south_max:<15.2e} {eq_status:<10}")

        except Exception as e:
            print(f"{name:<25} {'ERROR':<15} {'':<15} {'':<15} {'FAIL':<10}")

    # Sample values
    print()
    print("Sample map values (different regions):")
    print("-" * 60)

    max_diff, mean_diff, regional, cuda_map, jax_map = test_cuda_vs_jax(
        nside, l_max, 'float64', 'float64', PHASE1_DFT
    )

    sample_rings = [0, nside//2, nside, 2*nside, 3*nside-1]
    region_names = ['North pole', 'North polar', 'Equator start', 'Equator center', 'South polar']
    for ring, region in zip(sample_rings, region_names):
        if ring < jax_map.shape[0]:
            print(f"  Ring {ring} ({region}):")
            print(f"    JAX:  {jax_map[ring, :4]}")
            print(f"    CUDA: {cuda_map[ring, :4]}")

    print()
    print("=" * 80)
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 80)

    return all_pass


def main():
    nside = int(sys.argv[1]) if len(sys.argv) > 1 else NSIDE
    l_max = int(sys.argv[2]) if len(sys.argv) > 2 else 3 * nside

    success = run_accuracy_tests(nside, l_max)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

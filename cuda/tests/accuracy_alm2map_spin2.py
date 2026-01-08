#!/usr/bin/env python3
"""Accuracy test for CUDA spin-2 alm2map vs JAX reference.

Usage:
    python accuracy_alm2map_spin2.py [nside] [l_max]
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


def create_test_eb_alm(nside, l_max, dtype=np.float64):
    """Create synthetic bandlimited test E,B alm.

    Creates random alm only for l <= nside (starting at l=2 for spin-2)
    to avoid aliasing and ensure accurate comparison.

    Returns:
        alm_packed: Shape [n_maps, l_max+1, l_max+1, 2] for CUDA (E,B in last dim)
        alm_E, alm_B: Separate arrays for JAX
    """
    np.random.seed(42)
    # Create alm arrays with shape [n_maps, l_max+1, l_max+1]
    alm_E = np.zeros((1, l_max + 1, l_max + 1), dtype=np.complex128)
    alm_B = np.zeros((1, l_max + 1, l_max + 1), dtype=np.complex128)

    # Fill only 2 <= l <= nside with random values (spin-2 starts at l=2)
    for l in range(2, nside + 1):
        # m=0 is real
        alm_E[0, l, 0] = np.random.randn()
        alm_B[0, l, 0] = np.random.randn()
        # m>0 are complex
        for m in range(1, l + 1):
            alm_E[0, l, m] = np.random.randn() + 1j * np.random.randn()
            alm_B[0, l, m] = np.random.randn() + 1j * np.random.randn()

    # Pack E,B into last dimension for CUDA
    alm_packed = np.stack([alm_E, alm_B], axis=-1)

    return alm_packed, alm_E, alm_B


def compute_relative_diff(cuda_map, jax_map):
    """Compute relative difference between CUDA and JAX maps."""
    diff = np.abs(cuda_map - jax_map)
    ref = np.abs(jax_map)
    mask = ref > 1e-15
    rel_diff = np.zeros_like(diff)
    rel_diff[mask] = diff[mask] / ref[mask]
    return np.max(rel_diff[mask]) if mask.any() else 0, np.mean(rel_diff[mask]) if mask.any() else 0


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
    """Test CUDA spin-2 alm2map against JAX reference."""
    set_phase1_method(phase1_method)

    # Create test alm
    alm_packed, alm_E, alm_B = create_test_eb_alm(nside, l_max)

    # JAX computation (uses separate E, B)
    alm_jax = {2: jnp.array(alm_E), -2: jnp.array(alm_B)}
    map_jax = jax_alm2map(nside, l_max, (2,), alm_jax)
    jax.block_until_ready(map_jax)
    # JAX returns complex maps due to convention: maps[2] *= -1, maps[-2] *= 1j
    # CUDA computes Q and U that match:
    #   Q = -Re(maps[2])
    #   U = Im(maps[-2])
    Q_jax = -np.array(map_jax[2][0]).real
    U_jax = np.array(map_jax[-2][0]).imag

    # CUDA computation (uses packed [E, B] in last dim)
    spht = SPHTCuda(nside, l_max, version='v6',
                    storage_precision=storage,
                    recurrence_precision=recurrence)

    complex_dtype = np.complex128 if storage == 'float64' else np.complex64
    # CUDA expects {2: array} with shape [n_maps, l_max+1, l_max+1, 2]
    alm_cuda_in = {2: alm_packed.astype(complex_dtype)}
    map_cuda = spht.alm2map(alm_cuda_in, spins=(2,))
    # Output is [n_maps, n_rings, 4*nside, 2] where last dim is [Q, U]
    Q_cuda = map_cuda[2][0, :, :, 0]
    U_cuda = map_cuda[2][0, :, :, 1]

    # Compute differences (overall and by region)
    Q_max_diff, Q_mean_diff = compute_relative_diff(Q_cuda, Q_jax)
    U_max_diff, U_mean_diff = compute_relative_diff(U_cuda, U_jax)
    Q_regional = compute_regional_diff(Q_cuda, Q_jax, nside)
    U_regional = compute_regional_diff(U_cuda, U_jax, nside)

    return {
        'Q': (Q_max_diff, Q_mean_diff, Q_regional, Q_cuda, Q_jax),
        'U': (U_max_diff, U_mean_diff, U_regional, U_cuda, U_jax)
    }


def run_accuracy_tests(nside, l_max):
    """Run full accuracy test suite."""
    print("=" * 80)
    print("Spin-2 alm2map Accuracy Test: CUDA vs JAX")
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
    print(f"{'Configuration':<20} {'Map':<5} {'Max Rel Diff':<15} {'Mean Rel Diff':<15} {'Status':<10}")
    print("-" * 80)

    for name, storage, recurrence, phase1 in configs:
        try:
            result = test_cuda_vs_jax(nside, l_max, storage, recurrence, phase1)

            threshold = THRESHOLD if storage == 'float64' else 1e-5

            for map_type in ['Q', 'U']:
                max_diff, mean_diff, regional, cuda_map, jax_map = result[map_type]
                passed = max_diff < threshold
                status = "PASS" if passed else "FAIL"
                if not passed:
                    all_pass = False

                print(f"{name:<20} {map_type:<5} {max_diff:<15.2e} {mean_diff:<15.2e} {status:<10}")

        except Exception as e:
            print(f"{name:<20} {'Q/U':<5} {'ERROR':<15} {str(e)[:20]:<15} {'FAIL':<10}")
            all_pass = False

    # Regional accuracy table
    print()
    print("Regional Accuracy (Max Rel Diff by region):")
    print("-" * 110)
    print(f"{'Configuration':<20} {'Map':<5} {'North Polar':<15} {'Equatorial':<15} {'South Polar':<15} {'Eq Status':<10}")
    print("-" * 110)

    for name, storage, recurrence, phase1 in configs:
        try:
            result = test_cuda_vs_jax(nside, l_max, storage, recurrence, phase1)

            threshold = THRESHOLD if storage == 'float64' else 1e-5

            for map_type in ['Q', 'U']:
                max_diff, mean_diff, regional, cuda_map, jax_map = result[map_type]

                north_max = regional['north_polar'][0]
                eq_max = regional['equatorial'][0]
                south_max = regional['south_polar'][0]

                # Check equatorial region for pass/fail (poles may have numerical issues)
                eq_passed = eq_max < threshold
                eq_status = "PASS" if eq_passed else "FAIL"

                print(f"{name:<20} {map_type:<5} {north_max:<15.2e} {eq_max:<15.2e} {south_max:<15.2e} {eq_status:<10}")

        except Exception as e:
            print(f"{name:<20} {'Q/U':<5} {'ERROR':<15} {'':<15} {'':<15} {'FAIL':<10}")

    # Sample values
    print()
    print("Sample Q map values (different regions):")
    print("-" * 60)

    result = test_cuda_vs_jax(nside, l_max, 'float64', 'float64', PHASE1_DFT)
    Q_cuda = result['Q'][3]
    Q_jax = result['Q'][4]

    sample_rings = [0, nside//2, nside, 2*nside, 3*nside-1]
    region_names = ['North pole', 'North polar', 'Equator start', 'Equator center', 'South polar']
    for ring, region in zip(sample_rings, region_names):
        if ring < Q_jax.shape[0]:
            print(f"  Ring {ring} ({region}):")
            print(f"    JAX:  {Q_jax[ring, :4]}")
            print(f"    CUDA: {Q_cuda[ring, :4]}")

    print()
    print("Sample U map values (different regions):")
    print("-" * 60)

    U_cuda = result['U'][3]
    U_jax = result['U'][4]

    for ring, region in zip(sample_rings, region_names):
        if ring < U_jax.shape[0]:
            print(f"  Ring {ring} ({region}):")
            print(f"    JAX:  {U_jax[ring, :4]}")
            print(f"    CUDA: {U_cuda[ring, :4]}")

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

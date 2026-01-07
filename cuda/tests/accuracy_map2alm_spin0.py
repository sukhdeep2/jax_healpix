#!/usr/bin/env python3
"""Accuracy test for CUDA spin-0 map2alm vs JAX reference.

Usage:
    python accuracy_map2alm_spin0.py [options]

Options:
    --nside=N       HEALPix resolution parameter (default: 64)
    --lmax=N        Maximum multipole (default: 3*nside)
    --linear        Use LINEAR accumulation mode (default)
    --log           Use LOG accumulation mode
    --help          Show this help

Examples:
    python accuracy_map2alm_spin0.py                    # Default: nside=64, LINEAR mode
    python accuracy_map2alm_spin0.py --nside=128        # nside=128, LINEAR mode
    python accuracy_map2alm_spin0.py --log              # nside=64, LOG mode
    python accuracy_map2alm_spin0.py --log --nside=32   # nside=32, LOG mode
"""

import numpy as np
import sys
import argparse
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from SPHT_jax import map2alm as jax_map2alm
from spht_cuda import SPHTCuda, set_phase1_method, set_precision, PHASE1_DFT, PHASE1_FFT_EQUATORIAL, PHASE1_BLUESTEIN

# Default parameters
NSIDE = 64
THRESHOLD = 1e-10  # Relative error threshold
ACCUMULATION_MODE = "linear"  # Default accumulation mode


def create_random_map(nside, dtype=np.float64):
    """Create random temperature map."""
    n_rings = 4 * nside - 1
    max_pix = 4 * nside
    np.random.seed(42)
    return np.random.randn(1, n_rings, max_pix).astype(dtype)


def compute_relative_diff(cuda_alm, jax_alm, nside):
    """Compute relative difference between CUDA and JAX results.

    Only compares l <= nside where HEALPix accuracy is reliable.
    """
    diff = np.abs(cuda_alm - jax_alm)
    ref = np.abs(jax_alm)

    # Only compare l <= nside and ref > threshold
    mask = np.zeros_like(ref, dtype=bool)
    for l in range(min(nside + 1, cuda_alm.shape[0])):
        for m in range(l + 1):
            if ref[l, m] > 1e-15:
                mask[l, m] = True

    rel_diff = np.zeros_like(diff)
    rel_diff[mask] = diff[mask] / ref[mask]

    return np.max(rel_diff[mask]) if mask.any() else 0, np.mean(rel_diff[mask]) if mask.any() else 0


def test_cuda_vs_jax(nside, l_max, storage='float64', recurrence='float64',
                     phase1_method=PHASE1_DFT, phase1_name='DFT',
                     accumulation_mode='linear'):
    """Test CUDA map2alm against JAX reference."""
    set_phase1_method(phase1_method)

    # Create test map
    map_data = create_random_map(nside)

    # JAX computation
    maps_jax = {0: jnp.array(map_data)}
    alm_jax = jax_map2alm(nside, l_max, (0,), maps_jax)
    jax.block_until_ready(alm_jax)
    alm_jax_np = np.array(alm_jax[0][0])

    # CUDA computation
    dtype = np.float64 if storage == 'float64' else np.float32
    spht = SPHTCuda(nside, l_max, version='v6',
                    storage_precision=storage,
                    recurrence_precision=recurrence,
                    accumulation_mode=accumulation_mode)

    maps_cuda = {0: map_data.astype(dtype)}
    alm_cuda = spht.map2alm(maps_cuda, spins=(0,))
    alm_cuda_np = alm_cuda[0][0]

    # Compute difference (only for l <= nside)
    max_diff, mean_diff = compute_relative_diff(alm_cuda_np, alm_jax_np, nside)

    return max_diff, mean_diff, alm_cuda_np, alm_jax_np


def run_accuracy_tests(nside, l_max, accumulation_mode='linear'):
    """Run full accuracy test suite."""
    print("=" * 80)
    print("Spin-0 map2alm Accuracy Test: CUDA vs JAX")
    print("=" * 80)
    print(f"  nside     = {nside}")
    print(f"  l_max     = {l_max}")
    print(f"  mode      = {accumulation_mode.upper()}")
    print(f"  threshold = {THRESHOLD:.0e}")
    print("=" * 80)
    print()

    configs = [
        ('f64_f64_dft', 'float64', 'float64', PHASE1_DFT, 'DFT'),
        ('f64_f64_fft', 'float64', 'float64', PHASE1_FFT_EQUATORIAL, 'FFT'),
        ('f64_f64_bluestein', 'float64', 'float64', PHASE1_BLUESTEIN, 'Bluestein'),
        ('f32_f32_dft', 'float32', 'float32', PHASE1_DFT, 'DFT'),
        ('f32_f32_bluestein', 'float32', 'float32', PHASE1_BLUESTEIN, 'Bluestein'),
    ]

    results = {}
    all_pass = True

    print(f"{'Configuration':<25} {'Max Rel Diff':<15} {'Mean Rel Diff':<15} {'Status':<10}")
    print("-" * 80)

    for name, storage, recurrence, phase1, phase1_name in configs:
        try:
            max_diff, mean_diff, cuda_alm, jax_alm = test_cuda_vs_jax(
                nside, l_max, storage, recurrence, phase1, phase1_name,
                accumulation_mode=accumulation_mode
            )

            # Use looser threshold for f32
            threshold = THRESHOLD if storage == 'float64' else 1e-5
            passed = max_diff < threshold
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False

            results[name] = (max_diff, mean_diff, passed)
            print(f"{name:<25} {max_diff:<15.2e} {mean_diff:<15.2e} {status:<10}")

        except Exception as e:
            print(f"{name:<25} {'ERROR':<15} {str(e)[:20]:<15} {'FAIL':<10}")
            results[name] = (None, None, False)
            all_pass = False

    # Print sample values
    print()
    print("Sample ALM values (l=2, m=0,1,2):")
    print("-" * 60)

    # Get last successful result for sample values
    max_diff, mean_diff, cuda_alm, jax_alm = test_cuda_vs_jax(
        nside, l_max, 'float64', 'float64', PHASE1_DFT, 'DFT',
        accumulation_mode=accumulation_mode
    )

    for l in [2, 5, 10]:
        print(f"  l={l}:")
        for m in range(min(3, l+1)):
            jax_val = jax_alm[l, m]
            cuda_val = cuda_alm[l, m]
            print(f"    m={m}: JAX={jax_val:.6e}  CUDA={cuda_val:.6e}")

    # Summary
    print()
    print("=" * 80)
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 80)

    return all_pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Accuracy test for CUDA spin-0 map2alm vs JAX reference.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python accuracy_map2alm_spin0.py                    # Default: nside=64, LINEAR mode
    python accuracy_map2alm_spin0.py --nside=128        # nside=128, LINEAR mode
    python accuracy_map2alm_spin0.py --log              # nside=64, LOG mode
    python accuracy_map2alm_spin0.py --log --nside=32   # nside=32, LOG mode
        """
    )
    parser.add_argument('--nside', type=int, default=NSIDE,
                        help=f'HEALPix resolution parameter (default: {NSIDE})')
    parser.add_argument('--lmax', type=int, default=None,
                        help='Maximum multipole (default: 3*nside)')
    parser.add_argument('--linear', action='store_true', default=True,
                        help='Use LINEAR accumulation mode (default)')
    parser.add_argument('--log', action='store_true',
                        help='Use LOG accumulation mode')
    return parser.parse_args()


def main():
    args = parse_args()

    nside = args.nside
    l_max = args.lmax if args.lmax else 3 * nside
    accumulation_mode = 'log' if args.log else 'linear'

    success = run_accuracy_tests(nside, l_max, accumulation_mode)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

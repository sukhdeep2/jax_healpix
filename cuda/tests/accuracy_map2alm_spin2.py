#!/usr/bin/env python3
"""Accuracy test for CUDA spin-2 map2alm vs JAX reference.

Usage:
    python accuracy_map2alm_spin2.py [options]

Options:
    --nside=N       HEALPix resolution parameter (default: 64)
    --lmax=N        Maximum multipole (default: 3*nside)
    --linear        Use LINEAR accumulation mode (default)
    --log           Use LOG accumulation mode
    --help          Show this help

Examples:
    python accuracy_map2alm_spin2.py                    # Default: nside=64, LINEAR mode
    python accuracy_map2alm_spin2.py --nside=128        # nside=128, LINEAR mode
    python accuracy_map2alm_spin2.py --log              # nside=64, LOG mode
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
from spht_cuda import SPHTCuda, set_phase1_method, set_precision, PHASE1_DFT, PHASE1_BLUESTEIN

# Default parameters
NSIDE = 64
THRESHOLD = 1e-10  # Relative error threshold
ACCUMULATION_MODE = "linear"  # Default accumulation mode


def create_random_qu_maps(nside, dtype=np.float64):
    """Create random Q and U polarization maps.

    Returns:
        QU_packed: Shape [n_maps, n_rings, 4*nside, 2] for CUDA
        Q_map, U_map: Separate arrays for JAX
    """
    n_rings = 4 * nside - 1
    max_pix = 4 * nside
    np.random.seed(42)
    Q_map = np.random.randn(1, n_rings, max_pix).astype(dtype)
    U_map = np.random.randn(1, n_rings, max_pix).astype(dtype)
    # Pack Q,U into last dimension for CUDA
    QU_packed = np.stack([Q_map, U_map], axis=-1)
    return QU_packed, Q_map, U_map


def compute_relative_diff(cuda_alm, jax_alm, nside, l_min=2):
    """Compute relative difference between CUDA and JAX results.

    Only compares l_min <= l <= nside where HEALPix accuracy is reliable.
    """
    diff = np.abs(cuda_alm - jax_alm)
    ref = np.abs(jax_alm)

    # Mask out l < l_min (spin-2 starts at l=2) and l > nside
    mask = np.zeros_like(ref, dtype=bool)
    for l in range(l_min, min(nside + 1, cuda_alm.shape[0])):
        for m in range(l + 1):
            if ref[l, m] > 1e-15:
                mask[l, m] = True

    rel_diff = np.zeros_like(diff)
    rel_diff[mask] = diff[mask] / ref[mask]

    return np.max(rel_diff[mask]) if mask.any() else 0, np.mean(rel_diff[mask]) if mask.any() else 0


def test_cuda_vs_jax(nside, l_max, storage='float64', recurrence='float64',
                     phase1_method=PHASE1_DFT, phase1_name='DFT',
                     accumulation_mode='linear'):
    """Test CUDA spin-2 map2alm against JAX reference."""
    set_phase1_method(phase1_method)

    # Create test maps
    QU_packed, Q_map, U_map = create_random_qu_maps(nside)

    # JAX computation (uses separate Q, U)
    maps_jax = {2: jnp.array(Q_map), -2: jnp.array(U_map)}
    alm_jax = jax_map2alm(nside, l_max, (2,), maps_jax)
    jax.block_until_ready(alm_jax)
    alm_E_jax = np.array(alm_jax[2][0])
    alm_B_jax = np.array(alm_jax[-2][0])

    # CUDA computation (uses packed [Q, U] in last dim)
    dtype = np.float64 if storage == 'float64' else np.float32
    spht = SPHTCuda(nside, l_max, version='v6',
                    storage_precision=storage,
                    recurrence_precision=recurrence,
                    accumulation_mode=accumulation_mode)

    # CUDA expects {2: array} with shape [n_maps, n_rings, 4*nside, 2]
    maps_cuda = {2: QU_packed.astype(dtype)}
    alm_cuda = spht.map2alm(maps_cuda, spins=(2,))
    # Output is [n_maps, l_max+1, l_max+1, 2] where last dim is [E, B]
    alm_E_cuda = alm_cuda[2][0, :, :, 0]
    alm_B_cuda = alm_cuda[2][0, :, :, 1]

    # Compute differences (only for l <= nside)
    E_max_diff, E_mean_diff = compute_relative_diff(alm_E_cuda, alm_E_jax, nside)
    B_max_diff, B_mean_diff = compute_relative_diff(alm_B_cuda, alm_B_jax, nside)

    return {
        'E': (E_max_diff, E_mean_diff, alm_E_cuda, alm_E_jax),
        'B': (B_max_diff, B_mean_diff, alm_B_cuda, alm_B_jax)
    }


def run_accuracy_tests(nside, l_max, accumulation_mode='linear'):
    """Run full accuracy test suite."""
    print("=" * 80)
    print("Spin-2 map2alm Accuracy Test: CUDA vs JAX")
    print("=" * 80)
    print(f"  nside     = {nside}")
    print(f"  l_max     = {l_max}")
    print(f"  mode      = {accumulation_mode.upper()}")
    print(f"  threshold = {THRESHOLD:.0e}")
    print("=" * 80)
    print()

    configs = [
        ('f64_f64_dft', 'float64', 'float64', PHASE1_DFT, 'DFT'),
        ('f64_f64_bluestein', 'float64', 'float64', PHASE1_BLUESTEIN, 'Bluestein'),
        ('f32_f32_dft', 'float32', 'float32', PHASE1_DFT, 'DFT'),
        ('f32_f32_bluestein', 'float32', 'float32', PHASE1_BLUESTEIN, 'Bluestein'),
    ]

    results = {}
    all_pass = True

    print(f"{'Configuration':<20} {'Mode':<5} {'Max Rel Diff':<15} {'Mean Rel Diff':<15} {'Status':<10}")
    print("-" * 80)

    for name, storage, recurrence, phase1, phase1_name in configs:
        try:
            result = test_cuda_vs_jax(nside, l_max, storage, recurrence, phase1, phase1_name,
                                       accumulation_mode=accumulation_mode)

            # Use looser threshold for f32
            threshold = THRESHOLD if storage == 'float64' else 1e-5

            for mode in ['E', 'B']:
                max_diff, mean_diff, cuda_alm, jax_alm = result[mode]
                passed = max_diff < threshold
                status = "PASS" if passed else "FAIL"
                if not passed:
                    all_pass = False

                results[f"{name}_{mode}"] = (max_diff, mean_diff, passed)
                print(f"{name:<20} {mode:<5} {max_diff:<15.2e} {mean_diff:<15.2e} {status:<10}")

        except Exception as e:
            print(f"{name:<20} {'E/B':<5} {'ERROR':<15} {str(e)[:20]:<15} {'FAIL':<10}")
            results[name] = (None, None, False)
            all_pass = False

    # Print sample values
    print()
    print("Sample E-mode ALM values (l=2,5,10, m=0,1,2):")
    print("-" * 60)

    result = test_cuda_vs_jax(nside, l_max, 'float64', 'float64', PHASE1_DFT, 'DFT',
                               accumulation_mode=accumulation_mode)
    cuda_E = result['E'][2]
    jax_E = result['E'][3]

    for l in [2, 5, 10]:
        print(f"  l={l}:")
        for m in range(min(3, l+1)):
            jax_val = jax_E[l, m]
            cuda_val = cuda_E[l, m]
            print(f"    m={m}: JAX={jax_val:.6e}  CUDA={cuda_val:.6e}")

    print()
    print("Sample B-mode ALM values (l=2,5,10, m=0,1,2):")
    print("-" * 60)

    cuda_B = result['B'][2]
    jax_B = result['B'][3]

    for l in [2, 5, 10]:
        print(f"  l={l}:")
        for m in range(min(3, l+1)):
            jax_val = jax_B[l, m]
            cuda_val = cuda_B[l, m]
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
        description='Accuracy test for CUDA spin-2 map2alm vs JAX reference.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python accuracy_map2alm_spin2.py                    # Default: nside=64, LINEAR mode
    python accuracy_map2alm_spin2.py --nside=128        # nside=128, LINEAR mode
    python accuracy_map2alm_spin2.py --log              # nside=64, LOG mode
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

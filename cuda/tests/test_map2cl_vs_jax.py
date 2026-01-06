#!/usr/bin/env python3
"""Test map2cl CUDA implementation against JAX reference.

This test validates:
1. alm2cl computation (auto and cross spectra)
2. map2cl end-to-end pipeline
3. Mixed spin-0 and spin-2 maps
4. Multiple map cross-correlations

Usage:
    python test_map2cl_vs_jax.py [nside] [precision]

    nside: HEALPix resolution parameter (default: 64)
    precision: 'float64' or 'float32' (default: float64)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')

# Default parameters
NSIDE = 64
PRECISION = 'float64'


def setup_precision(precision):
    """Configure JAX precision settings."""
    import jax

    if precision == 'float64':
        jax.config.update("jax_enable_x64", True)
        np_dtype = np.float64
        np_complex = np.complex128
    else:
        jax.config.update("jax_enable_x64", False)
        np_dtype = np.float32
        np_complex = np.complex64

    return np_dtype, np_complex


def test_alm2cl_auto(nside, precision):
    """Test auto power spectrum computation."""
    from spht_cuda import SPHTCuda, alm2cl_cuda
    import jax
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2cl as jax_alm2cl

    np_dtype, np_complex = setup_precision(precision)

    print(f"\n=== Test alm2cl auto-spectrum: nside={nside}, precision={precision} ===\n")

    l_max = 3 * nside
    n_rings = 4 * nside - 1
    n_maps = 2  # Test with multiple maps

    # Create random maps
    np.random.seed(42)
    maps = np.random.randn(n_maps, n_rings, 4 * nside).astype(np_dtype)

    # CUDA: compute alm
    spht = SPHTCuda(nside, l_max, version='v6',
                    storage_precision=precision,
                    recurrence_precision='float64')
    alm_split = spht.map2alm({0: maps}, spins=(0,), return_split=True)
    alm_real, alm_imag = alm_split[0]

    # CUDA: compute C_l
    cl_cuda = alm2cl_cuda(l_max, alm_real, alm_imag)

    # JAX: compute alm and C_l
    map_jax = {0: jnp.array(maps)}
    alm_jax = jax_map2alm(nside, l_max, (0,), map_jax)
    jax.block_until_ready(alm_jax)
    cl_jax = np.array(jax_alm2cl(l_max, alm_jax[0]))

    # Compare
    diff = np.abs(cl_cuda - cl_jax)
    rel_diff = diff / (np.abs(cl_jax) + 1e-15)

    # Only compare reliable range
    l_reliable = nside
    max_rel = np.max(rel_diff[:, :l_reliable+1])
    mean_rel = np.mean(rel_diff[:, :l_reliable+1])

    print(f"C_l comparison (l <= {l_reliable}):")
    print(f"  Max relative diff: {max_rel:.6e}")
    print(f"  Mean relative diff: {mean_rel:.6e}")

    # Show sample values
    sample_ells = [0, 2, 10, nside//4, nside//2, nside]
    sample_ells = [l for l in sample_ells if l <= l_max]

    print(f"\nSample C_l values (map 0):")
    print(f"  l     |    JAX          |    CUDA")
    print("-" * 50)
    for l in sample_ells:
        print(f"  {l:4d} | {cl_jax[0, l]:14.6e} | {cl_cuda[0, l]:14.6e}")

    threshold = 1e-4 if precision == 'float64' else 1e-2
    passed = max_rel < threshold
    print(f"\nTest: {'PASS' if passed else 'FAIL'} (threshold: {threshold:.0e})")

    return passed


def test_alm2cl_cross(nside, precision):
    """Test cross power spectrum computation."""
    from spht_cuda import SPHTCuda, alm2cl_cuda
    import jax
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2cl as jax_alm2cl

    np_dtype, np_complex = setup_precision(precision)

    print(f"\n=== Test alm2cl cross-spectrum: nside={nside}, precision={precision} ===\n")

    l_max = 3 * nside
    n_rings = 4 * nside - 1

    # Create two random maps
    np.random.seed(42)
    map1 = np.random.randn(1, n_rings, 4 * nside).astype(np_dtype)
    np.random.seed(123)
    map2 = np.random.randn(1, n_rings, 4 * nside).astype(np_dtype)

    # CUDA: compute alm for both maps
    spht = SPHTCuda(nside, l_max, version='v6',
                    storage_precision=precision,
                    recurrence_precision='float64')
    alm1_split = spht.map2alm({0: map1}, spins=(0,), return_split=True)
    alm2_split = spht.map2alm({0: map2}, spins=(0,), return_split=True)
    alm1_real, alm1_imag = alm1_split[0]
    alm2_real, alm2_imag = alm2_split[0]

    # CUDA: compute cross-C_l
    cl_cuda = alm2cl_cuda(l_max, alm1_real, alm1_imag, alm2_real, alm2_imag)

    # JAX: compute cross-C_l
    alm1_jax = jax_map2alm(nside, l_max, (0,), {0: jnp.array(map1)})
    alm2_jax = jax_map2alm(nside, l_max, (0,), {0: jnp.array(map2)})
    jax.block_until_ready(alm1_jax)
    jax.block_until_ready(alm2_jax)
    cl_jax = np.array(jax_alm2cl(l_max, alm1_jax[0], alm2_jax[0]))

    # Compare
    diff = np.abs(cl_cuda - cl_jax)
    rel_diff = diff / (np.abs(cl_jax) + 1e-15)

    l_reliable = nside
    max_rel = np.max(rel_diff[:, :l_reliable+1])
    mean_rel = np.mean(rel_diff[:, :l_reliable+1])

    print(f"Cross C_l comparison (l <= {l_reliable}):")
    print(f"  Max relative diff: {max_rel:.6e}")
    print(f"  Mean relative diff: {mean_rel:.6e}")

    threshold = 1e-4 if precision == 'float64' else 1e-2
    passed = max_rel < threshold
    print(f"\nTest: {'PASS' if passed else 'FAIL'} (threshold: {threshold:.0e})")

    return passed


def test_map2cl_spin0(nside, precision):
    """Test map2cl with spin-0 (temperature) maps only."""
    from spht_cuda import map2cl_cuda, SPHTCuda
    import jax
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2cl as jax_alm2cl

    np_dtype, np_complex = setup_precision(precision)

    print(f"\n=== Test map2cl spin-0: nside={nside}, precision={precision} ===\n")

    l_max = 3 * nside
    n_rings = 4 * nside - 1

    # Create random T map
    np.random.seed(42)
    T_map = np.random.randn(n_rings, 4 * nside).astype(np_dtype)

    # CUDA map2cl
    cl_cuda = map2cl_cuda(nside, l_max, [T_map],
                          storage_precision=precision,
                          recurrence_precision='float64')

    # JAX reference
    alm_jax = jax_map2alm(nside, l_max, (0,), {0: jnp.array(T_map[None, :, :])})
    jax.block_until_ready(alm_jax)
    cl_jax = np.array(jax_alm2cl(l_max, alm_jax[0]))[0]

    print(f"CUDA map2cl returned keys: {list(cl_cuda.keys())}")

    # Compare TT spectrum
    diff = np.abs(cl_cuda['TT'] - cl_jax)
    rel_diff = diff / (np.abs(cl_jax) + 1e-15)

    l_reliable = nside
    max_rel = np.max(rel_diff[:l_reliable+1])
    mean_rel = np.mean(rel_diff[:l_reliable+1])

    print(f"\nTT spectrum comparison (l <= {l_reliable}):")
    print(f"  Max relative diff: {max_rel:.6e}")
    print(f"  Mean relative diff: {mean_rel:.6e}")

    threshold = 1e-4 if precision == 'float64' else 1e-2
    passed = max_rel < threshold
    print(f"\nTest: {'PASS' if passed else 'FAIL'} (threshold: {threshold:.0e})")

    return passed


def test_map2cl_spin2(nside, precision):
    """Test map2cl with spin-2 (polarization) maps only."""
    from spht_cuda import map2cl_cuda
    import jax
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2cl as jax_alm2cl

    np_dtype, np_complex = setup_precision(precision)

    print(f"\n=== Test map2cl spin-2: nside={nside}, precision={precision} ===\n")

    l_max = 3 * nside
    n_rings = 4 * nside - 1

    # Create random Q, U maps
    np.random.seed(42)
    Q_map = np.random.randn(n_rings, 4 * nside).astype(np_dtype)
    U_map = np.random.randn(n_rings, 4 * nside).astype(np_dtype)

    # CUDA map2cl - spin-2 input format: [2, n_rings, 4*nside]
    QU_map = np.stack([Q_map, U_map], axis=0)
    cl_cuda = map2cl_cuda(nside, l_max, [QU_map],
                          storage_precision=precision,
                          recurrence_precision='float64')

    # JAX reference
    maps_jax = {
        2: jnp.array(Q_map[None, :, :]),
        -2: jnp.array(U_map[None, :, :])
    }
    alm_jax = jax_map2alm(nside, l_max, (2,), maps_jax)
    jax.block_until_ready(alm_jax)

    # JAX computes EE, BB spectra
    cl_EE_jax = np.array(jax_alm2cl(l_max, alm_jax[2]))[0]
    cl_BB_jax = np.array(jax_alm2cl(l_max, alm_jax[-2]))[0]
    # EB cross
    cl_EB_jax = np.array(jax_alm2cl(l_max, alm_jax[2], alm_jax[-2]))[0]

    print(f"CUDA map2cl returned keys: {list(cl_cuda.keys())}")

    passed = True
    l_reliable = nside

    # Compare EE
    diff_EE = np.abs(cl_cuda['EE'] - cl_EE_jax)
    rel_diff_EE = diff_EE / (np.abs(cl_EE_jax) + 1e-15)
    max_rel_EE = np.max(rel_diff_EE[:l_reliable+1])
    print(f"\nEE spectrum: Max relative diff = {max_rel_EE:.6e}")

    # Compare BB
    diff_BB = np.abs(cl_cuda['BB'] - cl_BB_jax)
    rel_diff_BB = diff_BB / (np.abs(cl_BB_jax) + 1e-15)
    max_rel_BB = np.max(rel_diff_BB[:l_reliable+1])
    print(f"BB spectrum: Max relative diff = {max_rel_BB:.6e}")

    # Compare EB
    diff_EB = np.abs(cl_cuda['EB'] - cl_EB_jax)
    rel_diff_EB = diff_EB / (np.abs(cl_EB_jax) + 1e-15)
    max_rel_EB = np.max(rel_diff_EB[:l_reliable+1])
    print(f"EB spectrum: Max relative diff = {max_rel_EB:.6e}")

    threshold = 1e-4 if precision == 'float64' else 1e-2
    passed = (max_rel_EE < threshold and max_rel_BB < threshold and max_rel_EB < threshold)
    print(f"\nTest: {'PASS' if passed else 'FAIL'} (threshold: {threshold:.0e})")

    return passed


def test_map2cl_mixed(nside, precision):
    """Test map2cl with mixed spin-0 and spin-2 maps (T + QU)."""
    from spht_cuda import map2cl_cuda
    import jax
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2cl as jax_alm2cl

    np_dtype, np_complex = setup_precision(precision)

    print(f"\n=== Test map2cl mixed T+QU: nside={nside}, precision={precision} ===\n")

    l_max = 3 * nside
    n_rings = 4 * nside - 1

    # Create random T, Q, U maps
    np.random.seed(42)
    T_map = np.random.randn(n_rings, 4 * nside).astype(np_dtype)
    Q_map = np.random.randn(n_rings, 4 * nside).astype(np_dtype)
    U_map = np.random.randn(n_rings, 4 * nside).astype(np_dtype)

    # CUDA map2cl - mixed format: [T, [Q, U]]
    QU_map = np.stack([Q_map, U_map], axis=0)
    cl_cuda = map2cl_cuda(nside, l_max, [T_map, QU_map],
                          storage_precision=precision,
                          recurrence_precision='float64')

    print(f"CUDA map2cl returned keys: {list(cl_cuda.keys())}")

    # JAX reference for T
    alm_T_jax = jax_map2alm(nside, l_max, (0,), {0: jnp.array(T_map[None, :, :])})
    jax.block_until_ready(alm_T_jax)
    cl_TT_jax = np.array(jax_alm2cl(l_max, alm_T_jax[0]))[0]

    # JAX reference for E, B
    maps_jax = {
        2: jnp.array(Q_map[None, :, :]),
        -2: jnp.array(U_map[None, :, :])
    }
    alm_EB_jax = jax_map2alm(nside, l_max, (2,), maps_jax)
    jax.block_until_ready(alm_EB_jax)
    cl_EE_jax = np.array(jax_alm2cl(l_max, alm_EB_jax[2]))[0]
    cl_BB_jax = np.array(jax_alm2cl(l_max, alm_EB_jax[-2]))[0]

    # TE cross
    cl_TE_jax = np.array(jax_alm2cl(l_max, alm_T_jax[0], alm_EB_jax[2]))[0]
    # TB cross
    cl_TB_jax = np.array(jax_alm2cl(l_max, alm_T_jax[0], alm_EB_jax[-2]))[0]

    l_reliable = nside
    threshold = 1e-4 if precision == 'float64' else 1e-2

    results = {}

    # Compare TT
    if 'TT' in cl_cuda:
        diff = np.abs(cl_cuda['TT'] - cl_TT_jax)
        rel_diff = diff / (np.abs(cl_TT_jax) + 1e-15)
        results['TT'] = np.max(rel_diff[:l_reliable+1])
        print(f"TT: Max relative diff = {results['TT']:.6e}")

    # Compare EE
    if 'EE' in cl_cuda:
        diff = np.abs(cl_cuda['EE'] - cl_EE_jax)
        rel_diff = diff / (np.abs(cl_EE_jax) + 1e-15)
        results['EE'] = np.max(rel_diff[:l_reliable+1])
        print(f"EE: Max relative diff = {results['EE']:.6e}")

    # Compare BB
    if 'BB' in cl_cuda:
        diff = np.abs(cl_cuda['BB'] - cl_BB_jax)
        rel_diff = diff / (np.abs(cl_BB_jax) + 1e-15)
        results['BB'] = np.max(rel_diff[:l_reliable+1])
        print(f"BB: Max relative diff = {results['BB']:.6e}")

    # Compare TE
    if 'TE' in cl_cuda:
        diff = np.abs(cl_cuda['TE'] - cl_TE_jax)
        rel_diff = diff / (np.abs(cl_TE_jax) + 1e-15)
        results['TE'] = np.max(rel_diff[:l_reliable+1])
        print(f"TE: Max relative diff = {results['TE']:.6e}")

    # Compare TB
    if 'TB' in cl_cuda:
        diff = np.abs(cl_cuda['TB'] - cl_TB_jax)
        rel_diff = diff / (np.abs(cl_TB_jax) + 1e-15)
        results['TB'] = np.max(rel_diff[:l_reliable+1])
        print(f"TB: Max relative diff = {results['TB']:.6e}")

    # Check all pass
    passed = all(v < threshold for v in results.values())
    print(f"\nTest: {'PASS' if passed else 'FAIL'} (threshold: {threshold:.0e})")

    return passed


def test_map2cl_multiple_maps(nside, precision):
    """Test map2cl with multiple map sets for cross-correlations."""
    from spht_cuda import map2cl_cuda

    np_dtype, _ = setup_precision(precision)

    print(f"\n=== Test map2cl multiple maps: nside={nside}, precision={precision} ===\n")

    l_max = 3 * nside
    n_rings = 4 * nside - 1

    # Create two T maps
    np.random.seed(42)
    T1 = np.random.randn(n_rings, 4 * nside).astype(np_dtype)
    np.random.seed(123)
    T2 = np.random.randn(n_rings, 4 * nside).astype(np_dtype)

    # CUDA map2cl
    cl_cuda = map2cl_cuda(nside, l_max, [T1, T2],
                          storage_precision=precision,
                          recurrence_precision='float64')

    print(f"CUDA map2cl returned keys: {list(cl_cuda.keys())}")

    # Should have: TT (T1 auto), T1T2 (cross), TT_2 (T2 auto)
    expected_keys = ['TT', 'T1T2', 'TT_2']
    has_all_keys = all(k in cl_cuda for k in expected_keys)

    print(f"Expected keys: {expected_keys}")
    print(f"Has all expected keys: {has_all_keys}")

    # Basic sanity checks
    # Cross should be different from auto
    if 'TT' in cl_cuda and 'T1T2' in cl_cuda:
        auto_sum = np.sum(cl_cuda['TT'][:nside])
        cross_sum = np.sum(cl_cuda['T1T2'][:nside])
        different = abs(auto_sum - cross_sum) > 1e-10
        print(f"Auto (sum) = {auto_sum:.6e}, Cross (sum) = {cross_sum:.6e}")
        print(f"Auto != Cross: {different}")

    passed = has_all_keys
    print(f"\nTest: {'PASS' if passed else 'FAIL'}")

    return passed


def test_map2cl_two_sets(nside, precision):
    """Test map2cl with two complete TQU sets for full cross-correlations."""
    from spht_cuda import map2cl_cuda

    np_dtype, _ = setup_precision(precision)

    print(f"\n=== Test map2cl two TQU sets: nside={nside}, precision={precision} ===\n")

    l_max = 3 * nside
    n_rings = 4 * nside - 1

    # Create two TQU sets
    np.random.seed(42)
    T1 = np.random.randn(n_rings, 4 * nside).astype(np_dtype)
    Q1 = np.random.randn(n_rings, 4 * nside).astype(np_dtype)
    U1 = np.random.randn(n_rings, 4 * nside).astype(np_dtype)

    np.random.seed(123)
    T2 = np.random.randn(n_rings, 4 * nside).astype(np_dtype)
    Q2 = np.random.randn(n_rings, 4 * nside).astype(np_dtype)
    U2 = np.random.randn(n_rings, 4 * nside).astype(np_dtype)

    QU1 = np.stack([Q1, U1], axis=0)
    QU2 = np.stack([Q2, U2], axis=0)

    # CUDA map2cl with two TQU sets
    cl_cuda = map2cl_cuda(nside, l_max, [T1, QU1, T2, QU2],
                          storage_precision=precision,
                          recurrence_precision='float64')

    print(f"CUDA map2cl returned {len(cl_cuda)} spectra:")
    for key in sorted(cl_cuda.keys()):
        print(f"  {key}: shape {cl_cuda[key].shape}")

    # For 6 fields (T1, E1, B1, T2, E2, B2), expect 6*7/2 = 21 spectra
    n_expected = 21
    print(f"\nExpected {n_expected} spectra, got {len(cl_cuda)}")

    passed = len(cl_cuda) == n_expected
    print(f"\nTest: {'PASS' if passed else 'FAIL'}")

    return passed


def main(nside=None, precision=None):
    """Run all map2cl tests."""
    if nside is None:
        nside = NSIDE
    if precision is None:
        precision = PRECISION

    print("=" * 70)
    print("map2cl CUDA vs JAX Reference Tests")
    print("=" * 70)
    print(f"  nside     = {nside}")
    print(f"  l_max     = {3 * nside}")
    print(f"  precision = {precision}")
    print("=" * 70)

    results = {}

    results['alm2cl_auto'] = test_alm2cl_auto(nside, precision)
    results['alm2cl_cross'] = test_alm2cl_cross(nside, precision)
    results['map2cl_spin0'] = test_map2cl_spin0(nside, precision)
    results['map2cl_spin2'] = test_map2cl_spin2(nside, precision)
    results['map2cl_mixed'] = test_map2cl_mixed(nside, precision)
    results['map2cl_multiple'] = test_map2cl_multiple_maps(nside, precision)
    results['map2cl_two_sets'] = test_map2cl_two_sets(nside, precision)

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    for test_name, passed in results.items():
        status = 'PASS' if passed else 'FAIL'
        print(f"  {test_name}: {status}")
    print("=" * 70)

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    nside = int(sys.argv[1]) if len(sys.argv) > 1 else NSIDE
    precision = sys.argv[2] if len(sys.argv) > 2 else PRECISION

    success = main(nside, precision)
    sys.exit(0 if success else 1)

import pytest
import jax
import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True) # Enable float64 for precision
import numpy as np # For healpy and test data
import healpy as hp

from jax_healpix.transform import HealpixTransformer
# from jax_healpix.grid import HealpixGrid # Not directly needed if transformer instantiates it

# Test parameters
NSIDE_TEST_SMALL = 2  # For quick tests, npix=48
LMAX_TEST_SMALL = 2 * NSIDE_TEST_SMALL # lmax for map2alm is often 3*nside-1 or 2*nside. For testing, 2*nside.
NSIDE_TEST_MEDIUM = 4 # npix=192
LMAX_TEST_MEDIUM = 2 * NSIDE_TEST_MEDIUM

# Helper to generate a simple 1D test map
def make_simple_1d_map(nside, peak_pix_idx=0, peak_val=1.0):
    npix = hp.nside2npix(nside)
    m = np.zeros(npix, dtype=np.float64)
    if 0 <= peak_pix_idx < npix:
        m[peak_pix_idx] = peak_val
    return jnp.array(m)

# Helper to generate simple 1D alms (healpy order, m-primary)
def make_simple_1d_alm(lmax, l_peak=1, m_peak=0, val=1.0+0.5j):
    alm_size = hp.Alm.getsize(lmax)
    alms_1d = np.zeros(alm_size, dtype=np.complex128)
    if m_peak < 0 or m_peak > l_peak or l_peak > lmax : # Basic check
        pass # invalid index, return zeros
    else:
        idx = hp.Alm.getidx(lmax, l_peak, m_peak)
        alms_1d[idx] = val
    return jnp.array(alms_1d)

# Helper to generate simple 2D alms (L,M order)
def make_simple_2d_alm(lmax, l_peak=1, m_peak=0, val=1.0+0.5j):
    alms_2d = jnp.zeros((lmax+1, lmax+1), dtype=jnp.complex128)
    if 0 <= m_peak <= l_peak <= lmax:
        alms_2d = alms_2d.at[l_peak, m_peak].set(val)
    return alms_2d


@pytest.fixture
def transformer_small():
    return HealpixTransformer(nside=NSIDE_TEST_SMALL, l_max=LMAX_TEST_SMALL)

@pytest.fixture
def transformer_medium():
    return HealpixTransformer(nside=NSIDE_TEST_MEDIUM, l_max=LMAX_TEST_MEDIUM)

# --- Test Map Reshaping ---
def test_map_reshape_roundtrip(transformer_small):
    nside = transformer_small.nside
    npix = transformer_small.grid.npix

    map_1d_orig = make_simple_1d_map(nside, peak_pix_idx=npix//2)

    map_2d = transformer_small.convert_map_1d_to_2d(map_1d_orig)
    assert map_2d.shape == (4 * nside - 1, 4 * nside)

    map_1d_reconv = transformer_small.convert_map_2d_to_1d(map_2d)
    assert map_1d_reconv.shape == (npix,)
    assert jnp.allclose(map_1d_orig, map_1d_reconv, atol=1e-9)

    map_1d_batch = jnp.stack([map_1d_orig, map_1d_orig * 0.5])
    map_2d_batch = transformer_small.convert_map_1d_to_2d(map_1d_batch)
    assert map_2d_batch.shape == (2, 4 * nside - 1, 4 * nside)
    map_1d_reconv_batch = transformer_small.convert_map_2d_to_1d(map_2d_batch)
    assert map_1d_reconv_batch.shape == (2, npix,)
    assert jnp.allclose(map_1d_batch, map_1d_reconv_batch, atol=1e-9)


# --- Test Alm Reshaping ---
def test_alm_reshape_roundtrip(transformer_small):
    lmax = transformer_small.l_max

    alm_1d_hp_orig = make_simple_1d_alm(lmax, l_peak=lmax//2, m_peak=lmax//2)
    alm_2d = HealpixTransformer.reshape_alm_1d_to_2d(lmax, alm_1d_hp_orig)
    assert alm_2d.shape == (lmax + 1, lmax + 1)

    alm_1d_hp_reconv = HealpixTransformer.stack_alm_2d_to_1d(lmax, alm_2d, order='m')
    alm_size_healpy = hp.Alm.getsize(lmax)
    assert alm_1d_hp_reconv.shape == (alm_size_healpy,)

    # Check consistency for m-order (healpy order)
    idx_test = hp.Alm.getidx(lmax, lmax//2, lmax//2)
    if idx_test < alm_size_healpy: # Ensure index is valid
         assert jnp.allclose(alm_1d_hp_reconv[idx_test], alm_1d_hp_orig[idx_test])


    alm_2d_batch = jnp.stack([alm_2d, alm_2d * (0.5+0.5j)])
    alm_1d_batch_reconv = HealpixTransformer.stack_alm_2d_to_1d(lmax, alm_2d_batch, order='m')
    assert alm_1d_batch_reconv.shape == (2, alm_size_healpy)
    if idx_test < alm_size_healpy:
        assert jnp.allclose(alm_1d_batch_reconv[0, idx_test], alm_1d_hp_orig[idx_test])
        assert jnp.allclose(alm_1d_batch_reconv[1, idx_test], alm_1d_hp_orig[idx_test]*(0.5+0.5j))

    num_coeffs_l_order = (lmax + 1) * (lmax + 2) // 2
    alm_1d_l_reconv = HealpixTransformer.stack_alm_2d_to_1d(lmax, alm_2d, order='l')
    assert alm_1d_l_reconv.shape == (num_coeffs_l_order,)


# --- Test SHT Round Trip (T-only) ---
@pytest.mark.parametrize("nside, lmax_factor_mult", [(NSIDE_TEST_SMALL, 2), (NSIDE_TEST_MEDIUM, 2)])
def test_sht_roundtrip_temperature(nside, lmax_factor_mult):
    lmax = nside * lmax_factor_mult
    transformer = HealpixTransformer(nside=nside, l_max=lmax)
    npix = transformer.grid.npix

    map_1d_T_orig = make_simple_1d_map(nside, peak_pix_idx=npix // 3, peak_val=1.0)
    # Add some more structure to avoid pure delta function map
    angles_theta, angles_phi = hp.pix2ang(nside, np.arange(npix))
    map_1d_T_orig += 0.1 * jnp.cos(angles_theta) * jnp.sin(2*angles_phi)

    map_2d_T_orig = transformer.convert_map_1d_to_2d(map_1d_T_orig)

    alm_dict_T = transformer.map2alm(maps_dict={0: map_2d_T_orig}, spins_in_map=(0,))
    assert 0 in alm_dict_T
    alm_T_2d = alm_dict_T[0]
    assert alm_T_2d.shape == (lmax + 1, lmax + 1)

    map_dict_T_reconv = transformer.alm2map(alm_dict={0: alm_T_2d}, spins_to_map=(0,)) # Use spins_to_map
    assert 0 in map_dict_T_reconv
    map_2d_T_reconv = map_dict_T_reconv[0]

    map_1d_T_reconv = transformer.convert_map_2d_to_1d(map_2d_T_reconv)

    # High tolerance due to lmax truncation and numerical precision
    mean_abs_diff = jnp.mean(jnp.abs(map_1d_T_orig - map_1d_T_reconv))
    assert mean_abs_diff < 1e-3 # Check average absolute difference


# --- Test SHT Round Trip (TQU/TEB) ---
@pytest.mark.parametrize("nside, lmax_factor_mult", [(NSIDE_TEST_SMALL, 2)])
def test_sht_roundtrip_polarization(nside, lmax_factor_mult):
    lmax = nside * lmax_factor_mult
    transformer = HealpixTransformer(nside=nside, l_max=lmax)
    npix = transformer.grid.npix

    map_1d_Q_orig = make_simple_1d_map(nside, peak_pix_idx=npix // 2, peak_val=0.5)
    map_1d_U_orig = make_simple_1d_map(nside, peak_pix_idx=npix // 3, peak_val=-0.2)

    map_2d_Q_orig = transformer.convert_map_1d_to_2d(map_1d_Q_orig)
    map_2d_U_orig = transformer.convert_map_1d_to_2d(map_1d_U_orig)

    maps_in = {2: map_2d_Q_orig, -2: map_2d_U_orig}

    alm_dict_pol = transformer.map2alm(maps_dict=maps_in, spins_in_map=(2, -2))
    assert 2 in alm_dict_pol
    assert -2 in alm_dict_pol
    alm_E_2d = alm_dict_pol[2]
    alm_B_2d = alm_dict_pol[-2]
    assert alm_E_2d.shape == (lmax+1, lmax+1)
    assert alm_B_2d.shape == (lmax+1, lmax+1)

    map_dict_pol_reconv = transformer.alm2map(alm_dict=alm_dict_pol, spins_to_map=(2,-2)) # Use spins_to_map
    assert 2 in map_dict_pol_reconv
    assert -2 in map_dict_pol_reconv
    map_2d_Q_reconv = map_dict_pol_reconv[2]
    map_2d_U_reconv = map_dict_pol_reconv[-2]

    map_1d_Q_reconv = transformer.convert_map_2d_to_1d(map_2d_Q_reconv)
    map_1d_U_reconv = transformer.convert_map_2d_to_1d(map_2d_U_reconv)

    assert jnp.mean(jnp.abs(map_1d_Q_orig - map_1d_Q_reconv)) < 1e-3
    assert jnp.mean(jnp.abs(map_1d_U_orig - map_1d_U_reconv)) < 1e-3


# --- Test vs Healpy (T-only map2alm) ---
@pytest.mark.parametrize("nside, lmax_factor_mult", [(NSIDE_TEST_SMALL, 2)])
def test_map2alm_vs_healpy_temperature(nside, lmax_factor_mult):
    lmax = nside * lmax_factor_mult
    transformer = HealpixTransformer(nside=nside, l_max=lmax)
    npix = transformer.grid.npix

    map_1d_T_jax = make_simple_1d_map(nside, peak_pix_idx=npix//4, peak_val=1.0)
    angles_theta, angles_phi = hp.pix2ang(nside, np.arange(npix))
    map_1d_T_jax += 0.2 * jnp.sin(angles_theta*3) * jnp.cos(angles_phi*2)
    map_2d_T_jax = transformer.convert_map_1d_to_2d(map_1d_T_jax)

    alm_dict_jax = transformer.map2alm({0: map_2d_T_jax}, spins_in_map=(0,))
    alm_T_2d_jax = alm_dict_jax[0]

    map_1d_T_np = np.array(map_1d_T_jax)
    alm_1d_hp = hp.map2alm(map_1d_T_np, lmax=lmax, iter=0)

    alm_T_1d_jax_hp_order = HealpixTransformer.stack_alm_2d_to_1d(lmax, alm_T_2d_jax, order='m')

    # Healpy map2alm does not multiply by dOmega (pixel area).
    # Our map2alm includes pix_area = 4pi/Npix.
    # So, alm_jax * Npix/(4pi) should be comparable to alm_hp.
    scaling_factor = npix / (4.0 * np.pi)

    # Compare a few low-l, low-m coefficients
    for l_test in range(min(lmax + 1, 3)): # Test up to l=2 or lmax
        for m_test in range(l_test + 1):
            idx_hp = hp.Alm.getidx(lmax, l_test, m_test)
            if idx_hp < len(alm_1d_hp): # Ensure index is valid for healpy alm array
                val_jax = alm_T_1d_jax_hp_order[idx_hp] * scaling_factor
                val_hp = alm_1d_hp[idx_hp]
                assert jnp.allclose(val_jax, val_hp, atol=1e-5)


# --- Test compute_cl ---
def test_compute_cl(transformer_small):
    lmax = transformer_small.l_max

    alm_T_2d = make_simple_2d_alm(lmax, l_peak=2, m_peak=1, val=1.0+0.0j)

    cl_TT_array = transformer_small.compute_cl(alm_T_2d)
    assert cl_TT_array.shape == (lmax+1,)

    expected_cl_val = 0.0
    if lmax >= 2 and 2 <= lmax : # Check if l_peak=2 is valid
        # For a_2,1 = 1.0:
        # C_2 = (1/(2*2+1)) * ( |a_2,0|^2 (0) + |sqrt(2)*a_2,1|^2 + |sqrt(2)*a_2,2|^2 (0) )
        # C_2 = (1/5) * (sqrt(2)*1.0)^2 = 2/5 = 0.4
        expected_cl_val = (1.0 / (2*2.0+1.0)) * (jnp.sqrt(2.0)*1.0)**2
        assert jnp.isclose(cl_TT_array[2], expected_cl_val, atol=1e-7)
        for l_idx_cl in range(lmax+1):
            if l_idx_cl != 2:
                assert jnp.isclose(cl_TT_array[l_idx_cl], 0.0, atol=1e-7)

    alm_dict = {0: alm_T_2d}
    cl_dict_auto = transformer_small.compute_cl(alm_dict)
    assert (0,0) in cl_dict_auto
    assert jnp.allclose(cl_dict_auto[(0,0)], cl_TT_array, atol=1e-7)

    alm_E_2d = make_simple_2d_alm(lmax, l_peak=2, m_peak=1, val=0.5j)
    alm_dict2 = {2: alm_E_2d} # Spin 2 for E-modes

    cl_dict_cross = transformer_small.compute_cl(alm_dict, alm_dict2)
    assert (0,2) in cl_dict_cross
    cl_TE = cl_dict_cross[(0,2)]

    if lmax >=2 and 2 <= lmax:
        # C_l^TE for l=2: (1/5) * Re( sum_m (sqrt(2)*a_T,2,m) * conj(sqrt(2)*a_E,2,m) )
        # For m=1: (1/5) * Re( (sqrt(2)*1.0) * conj(sqrt(2)*0.5j) ) = (1/5) * Re( 2.0 * (-0.5j) ) = 0
        assert jnp.isclose(cl_TE[2], 0.0, atol=1e-7)

```

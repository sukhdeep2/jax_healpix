import pytest
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True) # Enable float64 for precision
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
    return jnp.array(m, dtype=jnp.float64) # Ensure float64

# Helper to generate simple 1D alms (healpy order, m-primary)
def make_simple_1d_alm(lmax, l_peak=1, m_peak=0, val=1.0+0.5j):
    alm_size = hp.Alm.getsize(lmax)
    alms_1d = np.zeros(alm_size, dtype=np.complex128) # complex128 for healpy
    if m_peak < 0 or m_peak > l_peak or l_peak > lmax : # Basic check
        pass
    else:
        idx = hp.Alm.getidx(lmax, l_peak, m_peak)
        alms_1d[idx] = val
    return jnp.array(alms_1d, dtype=jnp.complex128) # Ensure complex128 for JAX

# Helper to generate simple 2D alms (L,M order)
def make_simple_2d_alm(lmax, l_peak=1, m_peak=0, val=1.0+0.5j):
    alms_2d = jnp.zeros((lmax+1, lmax+1), dtype=jnp.complex128) # complex128 for JAX
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

    idx_test = hp.Alm.getidx(lmax, lmax//2, lmax//2)
    if idx_test < alm_size_healpy:
         assert jnp.allclose(alm_1d_hp_reconv[idx_test], alm_1d_hp_orig[idx_test])


    alm_2d_batch = jnp.stack([alm_2d, alm_2d * jnp.array(0.5+0.5j, dtype=jnp.complex128)])
    alm_1d_batch_reconv = HealpixTransformer.stack_alm_2d_to_1d(lmax, alm_2d_batch, order='m')
    assert alm_1d_batch_reconv.shape == (2, alm_size_healpy)
    if idx_test < alm_size_healpy:
        assert jnp.allclose(alm_1d_batch_reconv[0, idx_test], alm_1d_hp_orig[idx_test])
        assert jnp.allclose(alm_1d_batch_reconv[1, idx_test], alm_1d_hp_orig[idx_test]*jnp.array(0.5+0.5j, dtype=jnp.complex128))

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
    angles_theta, angles_phi = hp.pix2ang(nside, np.arange(npix))
    map_1d_T_orig += 0.1 * jnp.cos(jnp.array(angles_theta, dtype=jnp.float64)) * jnp.sin(2*jnp.array(angles_phi, dtype=jnp.float64))

    map_2d_T_orig = transformer.convert_map_1d_to_2d(map_1d_T_orig)

    alm_dict_T = transformer.map2alm(maps_dict={0: map_2d_T_orig}, spins_in_map=(0,))
    assert 0 in alm_dict_T
    alm_T_2d = alm_dict_T[0]
    assert alm_T_2d.shape == (lmax + 1, lmax + 1)

    map_dict_T_reconv = transformer.alm2map(alm_dict={0: alm_T_2d}, spins_to_map=(0,))
    assert 0 in map_dict_T_reconv
    map_2d_T_reconv = map_dict_T_reconv[0]

    map_1d_T_reconv = transformer.convert_map_2d_to_1d(map_2d_T_reconv)

    mean_abs_diff = jnp.mean(jnp.abs(map_1d_T_orig - map_1d_T_reconv))
    assert mean_abs_diff < 1e-3


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

    map_dict_pol_reconv = transformer.alm2map(alm_dict=alm_dict_pol, spins_to_map=(2,-2))
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
    map_1d_T_jax += 0.2 * jnp.sin(jnp.array(angles_theta*3, dtype=jnp.float64)) * jnp.cos(jnp.array(angles_phi*2, dtype=jnp.float64))
    map_2d_T_jax = transformer.convert_map_1d_to_2d(map_1d_T_jax)

    alm_dict_jax = transformer.map2alm({0: map_2d_T_jax}, spins_in_map=(0,))
    alm_T_2d_jax = alm_dict_jax[0]

    map_1d_T_np = np.array(map_1d_T_jax, dtype=np.float64)
    alm_1d_hp = hp.map2alm(map_1d_T_np, lmax=lmax, iter=0)

    alm_T_1d_jax_hp_order = HealpixTransformer.stack_alm_2d_to_1d(lmax, alm_T_2d_jax, order='m')

    scaling_factor = npix / (4.0 * np.pi)

    for l_test in range(min(lmax + 1, 3)):
        for m_test in range(l_test + 1):
            idx_hp = hp.Alm.getidx(lmax, l_test, m_test)
            if idx_hp < len(alm_1d_hp):
                val_jax = alm_T_1d_jax_hp_order[idx_hp] * scaling_factor
                val_hp = jnp.array(alm_1d_hp[idx_hp], dtype=jnp.complex128) # Ensure JAX array for comparison
                assert jnp.allclose(val_jax, val_hp, atol=1e-5)


# --- Test compute_cl ---
def test_compute_cl(transformer_small):
    lmax = transformer_small.l_max

    alm_T_2d = make_simple_2d_alm(lmax, l_peak=2, m_peak=1, val=jnp.array(1.0+0.0j, dtype=jnp.complex128))

    cl_TT_array = transformer_small.compute_cl(alm_T_2d)
    assert cl_TT_array.shape == (lmax+1,)

    if lmax >= 2:
        expected_cl_val = (1.0 / (2*2.0+1.0)) * (jnp.sqrt(2.0)*1.0)**2
        assert jnp.isclose(cl_TT_array[2], jnp.array(expected_cl_val, dtype=jnp.float64))
        for l_idx_cl in range(lmax+1):
            if l_idx_cl != 2:
                assert jnp.isclose(cl_TT_array[l_idx_cl], jnp.array(0.0, dtype=jnp.float64), atol=1e-7)

    alm_dict = {0: alm_T_2d}
    cl_dict_auto = transformer_small.compute_cl(alm_dict)
    assert (0,0) in cl_dict_auto
    assert jnp.allclose(cl_dict_auto[(0,0)], cl_TT_array, atol=1e-7)

    alm_E_2d = make_simple_2d_alm(lmax, l_peak=2, m_peak=1, val=jnp.array(0.5j, dtype=jnp.complex128))
    alm_dict2 = {2: alm_E_2d}

    cl_dict_cross = transformer_small.compute_cl(alm_dict, alm_dict2)
    assert (0,2) in cl_dict_cross
    cl_TE = cl_dict_cross[(0,2)]

    if lmax >=2:
        assert jnp.isclose(cl_TE[2], jnp.array(0.0, dtype=jnp.float64), atol=1e-7)

# --- Test SHT with Logarithmic YLM Recurrence ---

@pytest.mark.parametrize("nside, lmax_factor", [(NSIDE_TEST_SMALL, 2), (NSIDE_TEST_MEDIUM, 2)])
def test_sht_roundtrip_log_ylm_temperature(nside, lmax_factor):
    lmax = nside * lmax_factor - 1
    # Instantiate transformer to use log YLM
    transformer = HealpixTransformer(nside=nside, l_max=lmax, ylm_use_log_recurrence=True)
    npix = transformer.grid.npix

    map_1d_T_orig = make_simple_1d_map(nside, peak_pix_idx=npix // 3, peak_val=1.0)
    map_2d_T_orig = transformer.convert_map_1d_to_2d(map_1d_T_orig)

    # map2alm for T-only using log YLMs
    alm_dict_T = transformer.map2alm(maps_dict={0: map_2d_T_orig}, spins_in_map=(0,))
    assert 0 in alm_dict_T
    alm_T_2d = alm_dict_T[0]
    assert alm_T_2d.shape == (lmax + 1, lmax + 1)

    # alm2map for T-only using log YLMs
    map_dict_T_reconv = transformer.alm2map(alm_dict={0: alm_T_2d}, spins_in_alm=(0,))
    assert 0 in map_dict_T_reconv
    map_2d_T_reconv = map_dict_T_reconv[0]

    map_1d_T_reconv = transformer.convert_map_2d_to_1d(map_2d_T_reconv)

    # Compare original 1D map with reconverted 1D map
    assert jnp.allclose(map_1d_T_orig, map_1d_T_reconv, atol=1e-5)


@pytest.mark.parametrize("nside, lmax_factor", [(NSIDE_TEST_SMALL, 2)]) # Keep this test smaller for now
def test_map2alm_log_ylm_vs_healpy_temperature(nside, lmax_factor):
    lmax = nside * lmax_factor - 1
    # Instantiate transformer to use log YLM
    transformer = HealpixTransformer(nside=nside, l_max=lmax, ylm_use_log_recurrence=True)

    npix = hp.nside2npix(nside) # For map generation
    map_1d_T_jax = make_simple_1d_map(nside, peak_pix_idx=1, peak_val=1.0)
    # Add a bit more variation to the map
    hp_indices_temp = np.arange(npix)
    theta_temp, phi_temp = hp.pix2ang(nside, hp_indices_temp, nest=False) # Ensure RING order for pix2ang
    map_1d_T_jax += 0.1 * jnp.cos(jnp.array(theta_temp, dtype=jnp.float64)) * jnp.sin(2*jnp.array(phi_temp, dtype=jnp.float64))

    map_2d_T_jax = transformer.convert_map_1d_to_2d(map_1d_T_jax)

    # JAX version with log YLMs
    alm_dict_jax = transformer.map2alm({0: map_2d_T_jax}, spins_in_map=(0,))
    alm_T_2d_jax = alm_dict_jax[0]

    # Healpy version (always 64-bit, non-log YLM implicitly)
    map_1d_T_np = np.array(map_1d_T_jax, dtype=np.float64)
    alm_1d_hp = hp.map2alm(map_1d_T_np, lmax=lmax, mmax=lmax, iter=0)

    # Convert JAX 2D alm to 1D healpy order for comparison
    alm_T_1d_jax_hp_order = HealpixTransformer.stack_alm_2d_to_1d(lmax, alm_T_2d_jax, order='m')

    # JAX alm * pix_area should be comparable to healpy alm. Our map2alm already multiplies by pix_area.
    # Healpy's map2alm effectively divides by Npix.
    # So alm_jax should be compared to alm_hp * (4*pi/Npix) for unit consistency if Ylms are same.
    # Or, alm_jax * (Npix / (4*pi)) vs alm_hp.
    # The previous scaling was `alm_jax * scaling_factor` where `scaling_factor = npix / (4.0 * np.pi)`.
    # This implies alm_hp is like sum(map * Ylm_conj), and alm_jax is sum(map*Ylm_conj*pix_area).
    # So alm_jax / pix_area = sum(map*Ylm_conj)
    # Comparison should be: alm_jax / pix_area vs alm_hp
    # OR alm_jax vs alm_hp * pix_area

    pix_area = 4.0 * np.pi / npix
    scaled_alm_hp = jnp.array(alm_1d_hp, dtype=jnp.complex128) * pix_area


    # Check a few alms
    idx_00 = hp.Alm.getidx(lmax, 0, 0)
    assert jnp.isclose(alm_T_1d_jax_hp_order[idx_00].real, scaled_alm_hp[idx_00].real, atol=1e-6)
    assert jnp.isclose(alm_T_1d_jax_hp_order[idx_00].imag, 0.0, atol=1e-7) # a00 is real
    assert jnp.isclose(scaled_alm_hp[idx_00].imag, 0.0, atol=1e-9)

    if lmax >= 1:
        idx_10 = hp.Alm.getidx(lmax, 1, 0)
        assert jnp.isclose(alm_T_1d_jax_hp_order[idx_10].real, scaled_alm_hp[idx_10].real, atol=1e-6)
        assert jnp.isclose(alm_T_1d_jax_hp_order[idx_10].imag, 0.0, atol=1e-7) # a_l0 is real

        idx_11 = hp.Alm.getidx(lmax, 1, 1)
        assert jnp.isclose(alm_T_1d_jax_hp_order[idx_11].real, scaled_alm_hp[idx_11].real, atol=1e-6)
        assert jnp.isclose(alm_T_1d_jax_hp_order[idx_11].imag, scaled_alm_hp[idx_11].imag, atol=1e-6)

    if lmax >= 2:
        idx_21 = hp.Alm.getidx(lmax, 2, 1)
        assert jnp.isclose(alm_T_1d_jax_hp_order[idx_21].real, scaled_alm_hp[idx_21].real, atol=1e-6)
        assert jnp.isclose(alm_T_1d_jax_hp_order[idx_21].imag, scaled_alm_hp[idx_21].imag, atol=1e-6)

        idx_22 = hp.Alm.getidx(lmax, 2, 2)
        assert jnp.isclose(alm_T_1d_jax_hp_order[idx_22].real, scaled_alm_hp[idx_22].real, atol=1e-6)
        assert jnp.isclose(alm_T_1d_jax_hp_order[idx_22].imag, scaled_alm_hp[idx_22].imag, atol=1e-6)

    max_abs_diff = jnp.max(jnp.abs(alm_T_1d_jax_hp_order - scaled_alm_hp))
    # print(f"\nMax abs diff (log YLM vs Healpy, Nside={nside}, Lmax={lmax}): {max_abs_diff}")
    assert max_abs_diff < 1e-5

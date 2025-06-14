import pytest
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True) # Enable float64 for precision
from jax_healpix.ylm import SphericalHarmonics

def test_spherical_harmonics_creation():
    ylm_calc = SphericalHarmonics(l_max=4, spins_to_compute=(0, 2), use_log_recurrence=False)
    assert ylm_calc.l_max == 4
    assert ylm_calc.spins_to_compute == (0, 2)
    assert not ylm_calc.use_log_recurrence

    ylm_calc_log = SphericalHarmonics(l_max=8, spins_to_compute=(0,), use_log_recurrence=True)
    assert ylm_calc_log.l_max == 8
    assert ylm_calc_log.spins_to_compute == (0,)
    assert ylm_calc_log.use_log_recurrence

    with pytest.raises(ValueError):
        SphericalHarmonics(l_max=-1)

def test_get_ylm_spin0_y00_basic():
    l_max = 0
    beta_values = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float64)

    ylm_calc_std = SphericalHarmonics(l_max=l_max, spins_to_compute=(0,), use_log_recurrence=False)
    s0_ylm_std = ylm_calc_std.get_ylm(spin=0, beta_values=beta_values)

    assert s0_ylm_std.shape == (1, 1, len(beta_values))
    expected_y00 = 1.0 / jnp.sqrt(4 * jnp.pi)
    assert jnp.allclose(s0_ylm_std[0, 0, :], jnp.array(expected_y00, dtype=jnp.float64))

    ylm_calc_log = SphericalHarmonics(l_max=l_max, spins_to_compute=(0,), use_log_recurrence=True)
    s0_ylm_log_abs, s0_ylm_log_signs = ylm_calc_log.get_ylm(spin=0, beta_values=beta_values)

    assert s0_ylm_log_abs.shape == (1, 1, len(beta_values))
    assert s0_ylm_log_signs.shape == (1, 1, len(beta_values))

    assert jnp.allclose(s0_ylm_log_abs[0, 0, :], jnp.log(jnp.array(expected_y00, dtype=jnp.float64)))
    assert jnp.all(s0_ylm_log_signs[0, 0, :] == 1)

def test_get_ylm_caching():
    l_max = 2
    beta_values = jnp.array([0.5], dtype=jnp.float64)
    ylm_calc = SphericalHarmonics(l_max=l_max, spins_to_compute=(0,))

    ylm1 = ylm_calc.get_ylm(spin=0, beta_values=beta_values)
    ylm2 = ylm_calc.get_ylm(spin=0, beta_values=beta_values)

    assert jnp.array_equal(ylm1, ylm2)

    ylm_calc_log = SphericalHarmonics(l_max=l_max, spins_to_compute=(0,), use_log_recurrence=True)
    ylm_abs1, ylm_sign1 = ylm_calc_log.get_ylm(spin=0, beta_values=beta_values)
    ylm_abs2, ylm_sign2 = ylm_calc_log.get_ylm(spin=0, beta_values=beta_values)
    assert jnp.array_equal(ylm_abs1, ylm_abs2)
    assert jnp.array_equal(ylm_sign1, ylm_sign2)


@pytest.mark.parametrize("order", ['l', 'm'])
def test_stack_ylm_2d_to_1d_single_array(order):
    l_max = 2
    ylm_2d_data = jnp.arange((l_max+1)*(l_max+1)*2, dtype=jnp.float64).reshape((l_max+1, l_max+1, 2))

    stacked_ylm = SphericalHarmonics.process_ylm_stacking(l_max, ylm_2d_data, order=order)

    num_coeffs = (l_max + 1) * (l_max + 2) // 2
    assert stacked_ylm.shape == (num_coeffs, 2)

    assert jnp.array_equal(stacked_ylm[0,:], ylm_2d_data[0,0,:])

    if order == 'l':
        assert jnp.array_equal(stacked_ylm[1,:], ylm_2d_data[1,0,:])
        assert jnp.array_equal(stacked_ylm[2,:], ylm_2d_data[1,1,:])
        assert jnp.array_equal(stacked_ylm[3,:], ylm_2d_data[2,0,:])
        assert jnp.array_equal(stacked_ylm[4,:], ylm_2d_data[2,1,:])
        assert jnp.array_equal(stacked_ylm[5,:], ylm_2d_data[2,2,:])
    elif order == 'm':
        assert jnp.array_equal(stacked_ylm[1,:], ylm_2d_data[1,0,:])
        assert jnp.array_equal(stacked_ylm[2,:], ylm_2d_data[2,0,:])
        assert jnp.array_equal(stacked_ylm[3,:], ylm_2d_data[1,1,:])
        assert jnp.array_equal(stacked_ylm[4,:], ylm_2d_data[2,1,:])
        assert jnp.array_equal(stacked_ylm[5,:], ylm_2d_data[2,2,:])

    with pytest.raises(ValueError):
        SphericalHarmonics.process_ylm_stacking(l_max, ylm_2d_data, order='x')


@pytest.mark.parametrize("order", ['l', 'm'])
def test_stack_ylm_2d_to_1d_dict_input(order):
    l_max = 1
    ylm_dict_2d = {
        0: jnp.arange((l_max+1)*(l_max+1)*1, dtype=jnp.float64).reshape((l_max+1, l_max+1, 1)),
        2: jnp.arange(100, 100 + (l_max+1)*(l_max+1)*1, dtype=jnp.float64).reshape((l_max+1, l_max+1, 1))
    }

    stacked_ylm_dict = SphericalHarmonics.process_ylm_stacking(l_max, ylm_dict_2d, order=order)

    assert isinstance(stacked_ylm_dict, dict)
    assert 0 in stacked_ylm_dict
    assert 2 in stacked_ylm_dict

    num_coeffs = (l_max + 1) * (l_max + 2) // 2
    assert stacked_ylm_dict[0].shape == (num_coeffs, 1)
    assert stacked_ylm_dict[2].shape == (num_coeffs, 1)

    assert jnp.array_equal(stacked_ylm_dict[0][0,:], ylm_dict_2d[0][0,0,:])
    assert jnp.array_equal(stacked_ylm_dict[2][0,:], ylm_dict_2d[2][0,0,:])

    ylm_dict_bad_shape = {0: jnp.ones((l_max, l_max, 1), dtype=jnp.float64)}
    with pytest.raises(ValueError):
        SphericalHarmonics.process_ylm_stacking(l_max, ylm_dict_bad_shape, order=order)

    ylm_dict_bad_type = {0: "not an array"}
    with pytest.raises(TypeError):
        SphericalHarmonics.process_ylm_stacking(l_max, ylm_dict_bad_type, order=order)

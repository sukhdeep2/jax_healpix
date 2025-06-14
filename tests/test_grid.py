import pytest
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True) # Enable float64 for precision
from jax_healpix.grid import HealpixGrid, nside_to_npix as standalone_nside_to_npix
import healpy as hp
import numpy as np # For healpy comparisons

# Test standalone nside_to_npix function
def test_standalone_nside_to_npix():
    assert standalone_nside_to_npix(1) == 12
    assert standalone_nside_to_npix(2) == 48
    assert standalone_nside_to_npix(4) == 192
    with pytest.raises(ValueError):
        standalone_nside_to_npix(jnp.array([1,2]))
    with pytest.raises(ValueError): # nside=0 should raise ValueError now
        standalone_nside_to_npix(0)
    with pytest.raises(ValueError): # nside=3 (non-power of 2) should raise ValueError
        standalone_nside_to_npix(3)


def test_healpix_grid_creation_and_npix():
    grid_n1 = HealpixGrid(nside=1)
    assert grid_n1.nside == 1
    assert grid_n1.npix == 12

    grid_n2 = HealpixGrid(nside=2)
    assert grid_n2.nside == 2
    assert grid_n2.npix == 48

    with pytest.raises(ValueError):
        HealpixGrid(nside=0)
    with pytest.raises(ValueError):
        HealpixGrid(nside=3)
    with pytest.raises(ValueError):
        HealpixGrid(nside=-1)

def test_ring_beta():
    nside = 2
    grid = HealpixGrid(nside=nside)
    beta_values = grid.ring_beta()

    assert beta_values.shape == (4 * nside - 1,)

    expected_beta_pole_ring1 = 1.0 - 1.0**2 / (3.0 * nside**2)
    assert jnp.isclose(beta_values[0], jnp.array(expected_beta_pole_ring1, dtype=jnp.float64))

    assert jnp.isclose(beta_values[2*nside-1], jnp.array(0.0, dtype=jnp.float64))

    assert jnp.isclose(beta_values[4*nside-2], jnp.array(-expected_beta_pole_ring1, dtype=jnp.float64))

    assert jnp.allclose(beta_values[:(2*nside-1)], -beta_values[(2*nside):][::-1], atol=1e-9)


def test_get_ring_properties():
    nside = 2
    grid = HealpixGrid(nside=nside)

    phi0_p, npix_p, beta_p = grid.get_ring_properties(ring_i=1)
    assert npix_p == 4 * 1
    assert jnp.isclose(beta_p, jnp.array(1 - 1**2 / (3 * nside**2), dtype=jnp.float64))
    assert jnp.isclose(phi0_p, jnp.array(jnp.pi / (2 * 1) * 0.5, dtype=jnp.float64))

    phi0_e, npix_e, beta_e = grid.get_ring_properties(ring_i=2*nside)
    assert npix_e == 4 * nside
    assert jnp.isclose(beta_e, jnp.array(0.0, dtype=jnp.float64))

    if nside == 2:
        assert jnp.isclose(phi0_e, jnp.array(0.0, dtype=jnp.float64))

    if nside == 1:
        grid_n1 = HealpixGrid(nside=1)
        phi0_e_n1, _, _ = grid_n1.get_ring_properties(ring_i=2*1)
        assert jnp.isclose(phi0_e_n1, jnp.array(jnp.pi / 4.0, dtype=jnp.float64))
    elif nside == 2:
        pass

    phi0_s, npix_s, beta_s = grid.get_ring_properties(ring_i=4*nside-1)
    assert npix_s == 4 * 1
    assert jnp.isclose(beta_s, jnp.array(-(1 - 1**2 / (3 * nside**2)), dtype=jnp.float64))
    assert jnp.isclose(phi0_s, jnp.array(jnp.pi / (2 * 1) * 0.5, dtype=jnp.float64))

    with pytest.raises(ValueError):
        grid.get_ring_properties(0)
    with pytest.raises(ValueError):
        grid.get_ring_properties(4*nside)

@pytest.mark.parametrize("nside_val", [1, 2, 4])
def test_pixel_to_angle_vs_healpy(nside_val):
    grid = HealpixGrid(nside=nside_val)
    npix = grid.npix

    jax_theta, jax_phi = grid.pixel_to_angle_ring_colatitude_longitude()

    assert jax_theta.shape == (npix,)
    assert jax_phi.shape == (npix,)

    hp_indices = np.arange(npix)
    hp_theta, hp_phi = hp.pix2ang(nside_val, hp_indices, nest=False)

    assert jnp.allclose(jax_theta, jnp.array(hp_theta, dtype=jnp.float64), atol=1e-9)
    assert jnp.allclose(jax_phi, jnp.array(hp_phi, dtype=jnp.float64), atol=1e-9)

    jax_theta_cached, jax_phi_cached = grid.pixel_to_angle_ring_colatitude_longitude()
    assert jnp.array_equal(jax_theta, jax_theta_cached)
    assert jnp.array_equal(jax_phi, jax_phi_cached)

def test_pixel_to_angle_nside1_specific():
    nside = 1
    grid = HealpixGrid(nside=nside)
    jax_theta, jax_phi = grid.pixel_to_angle_ring_colatitude_longitude()

    hp_indices = np.arange(12)
    hp_theta_n1, hp_phi_n1 = hp.pix2ang(nside, hp_indices, nest=False)

    assert jnp.allclose(jax_theta, jnp.array(hp_theta_n1, dtype=jnp.float64), atol=1e-9)
    assert jnp.allclose(jax_phi, jnp.array(hp_phi_n1, dtype=jnp.float64), atol=1e-9)

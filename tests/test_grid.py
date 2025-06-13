import pytest
import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True) # Enable float64 for precision
from jax_healpix.grid import HealpixGrid, nside_to_npix as standalone_nside_to_npix
import healpy as hp
import numpy as np # For healpy comparisons

# Test standalone nside_to_npix function
# The standalone nside_to_npix in grid.py has a check for nside being a power of 2.
# Healpy's nside2npix does not have this restriction, it's usually applied at HealpixGrid level.
# For this test, we check against the behavior of the standalone function.
def test_standalone_nside_to_npix():
    assert standalone_nside_to_npix(1) == 12
    assert standalone_nside_to_npix(2) == 48
    assert standalone_nside_to_npix(4) == 192
    # Standalone nside_to_npix does not raise error for non-power-of-2 or 0 itself,
    # that's usually a higher-level check (e.g. in HealpixGrid constructor).
    # It only checks for integer or JAX array type and scalar if JAX array.
    # Let's test its type checks.
    with pytest.raises(ValueError):
        standalone_nside_to_npix(jnp.array([1,2])) # Non-scalar JAX array
    # It does not inherently restrict nside value beyond type.
    # The power-of-2 check is in HealpixGrid constructor.
    assert standalone_nside_to_npix(0) == 0 # nside2npix(0) = 0
    assert standalone_nside_to_npix(3) == 12 * 3 * 3 # nside2npix(3) = 108


def test_healpix_grid_creation_and_npix():
    grid_n1 = HealpixGrid(nside=1)
    assert grid_n1.nside == 1
    assert grid_n1.npix == 12

    grid_n2 = HealpixGrid(nside=2)
    assert grid_n2.nside == 2
    assert grid_n2.npix == 48

    with pytest.raises(ValueError):
        HealpixGrid(nside=0) # Invalid: nside must be positive power of 2
    with pytest.raises(ValueError):
        HealpixGrid(nside=3) # Invalid: nside must be power of 2
    with pytest.raises(ValueError):
        HealpixGrid(nside=-1)

def test_ring_beta():
    nside = 2
    grid = HealpixGrid(nside=nside)
    beta_values = grid.ring_beta()

    assert beta_values.shape == (4 * nside - 1,) # For nside=2, this is 7 rings

    # Ring indices for beta_values array are 0 to 4*nside-2
    # Physical ring indices are 1 to 4*nside-1

    # North polar cap ring (physical ring 1, array index 0)
    # beta = 1 - r^2 / (3*N^2) where r=1
    expected_beta_pole_ring1 = 1.0 - 1.0**2 / (3.0 * nside**2)
    assert jnp.isclose(beta_values[0], expected_beta_pole_ring1)

    # Equator ring (physical ring 2*nside, array index 2*nside-1)
    # beta should be 0 for the equator
    assert jnp.isclose(beta_values[2*nside-1], 0.0)

    # South polar cap ring (physical ring 4*nside-1, array index 4*nside-2)
    # beta should be -expected_beta_pole_ring1 (for r=1 equivalent south ring)
    assert jnp.isclose(beta_values[4*nside-2], -expected_beta_pole_ring1)

    # Symmetry check: beta[i] == -beta[num_rings - 1 - i]
    # For nside=2, num_rings = 7. beta[0]==-beta[6], beta[1]==-beta[5], beta[2]==-beta[4]
    # Middle element beta[3] (equator) is 0.
    # This means beta_values[:2*nside-1] should be approx -beta_values[2*nside:][::-1]
    # num_polar_rings_half = nside - 1 (polar cap rings)
    # num_equatorial_rings_half = nside (part of equatorial belt on one side of equator)
    # Total before equator: (nside-1) polar + nside equatorial_half = 2*nside-1 rings
    assert jnp.allclose(beta_values[:(2*nside-1)], -beta_values[(2*nside):][::-1], atol=1e-9)


def test_get_ring_properties():
    nside = 2
    grid = HealpixGrid(nside=nside)

    # Test a polar ring (physical ring 1, North cap)
    phi0_p, npix_p, beta_p = grid.get_ring_properties(ring_i=1)
    assert npix_p == 4 * 1
    assert jnp.isclose(beta_p, 1 - 1**2 / (3 * nside**2))
    assert jnp.isclose(phi0_p, jnp.pi / (2 * 1) * 0.5)

    # Test an equatorial ring (physical ring 2*nside = 4 for nside=2, which is the equator)
    phi0_e, npix_e, beta_e = grid.get_ring_properties(ring_i=2*nside)
    assert npix_e == 4 * nside
    assert jnp.isclose(beta_e, 0.0)

    # phi_0 for equator (ring_i = 2*nside):
    # s_orig_healpix = (ring_i - nside + 1) % 2. If 0, s_healpix_factor=1. If 1, s_healpix_factor=0.5
    # phi_0 = (pi / (2*nside)) * (1.0 - s_factor_from_code)
    # In grid.py: s = jnp.where((ring_i - self.nside + 1) % 2 == 0, 1, 2)
    # phi_0 = jnp.pi / (2 * self.nside) * (1.0 - s / 2.0)
    # For ring_i = 2*nside: (2*nside - nside + 1) = nside + 1.
    # If nside=2, nside+1=3 (odd). s=2. phi_0 = pi/(2*2) * (1.0 - 2/2.0) = pi/4 * 0 = 0.
    if nside == 2:
        assert jnp.isclose(phi0_e, 0.0)

    # Test for nside=1 equator (ring_i=2*1=2)
    # nside=1, nside+1=2 (even). s=1. phi_0 = pi/(2*1) * (1.0 - 1/2.0) = pi/2 * 0.5 = pi/4
    if nside == 1: # This test case for nside=1 needs its own grid
        grid_n1 = HealpixGrid(nside=1)
        phi0_e_n1, _, _ = grid_n1.get_ring_properties(ring_i=2*1)
        assert jnp.isclose(phi0_e_n1, jnp.pi / 4.0)
    elif nside == 2: # Check the value for nside=2, which is phi0_e=0.0 as asserted above.
        pass


    # Test a south polar ring (physical ring 4*nside-1, which is ring 7 for nside=2)
    # For this ring, effective ring_i for calculation (ring_i_calc) is 1.
    phi0_s, npix_s, beta_s = grid.get_ring_properties(ring_i=4*nside-1)
    assert npix_s == 4 * 1
    assert jnp.isclose(beta_s, -(1 - 1**2 / (3 * nside**2)))
    assert jnp.isclose(phi0_s, jnp.pi / (2 * 1) * 0.5)

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

    assert jnp.allclose(jax_theta, hp_theta, atol=1e-9)
    assert jnp.allclose(jax_phi, hp_phi, atol=1e-9)

    jax_theta_cached, jax_phi_cached = grid.pixel_to_angle_ring_colatitude_longitude()
    assert jnp.array_equal(jax_theta, jax_theta_cached)
    assert jnp.array_equal(jax_phi, jax_phi_cached)

def test_pixel_to_angle_nside1_specific():
    nside = 1
    grid = HealpixGrid(nside=nside)
    jax_theta, jax_phi = grid.pixel_to_angle_ring_colatitude_longitude()

    hp_indices = np.arange(12)
    hp_theta_n1, hp_phi_n1 = hp.pix2ang(nside, hp_indices, nest=False)

    assert jnp.allclose(jax_theta, hp_theta_n1, atol=1e-9)
    assert jnp.allclose(jax_phi, hp_phi_n1, atol=1e-9)

```

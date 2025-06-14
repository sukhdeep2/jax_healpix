"""Provides the HealpixGrid class for HEALPix grid computations in JAX."""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

def nside_to_npix(nside):
    """Calculates the number of pixels in a HEALPix map.

    Args:
        nside (int): The HEALPix NSIDE parameter. Must be a power of 2.

    Returns:
        int: The total number of pixels (npix = 12 * nside^2).
    """
    if not isinstance(nside, (int, jnp.ndarray)):
        raise ValueError("nside must be an integer or JAX array.")
    if isinstance(nside, jnp.ndarray) and nside.ndim != 0:
        raise ValueError("nside must be a scalar if it's a JAX array.")
    if isinstance(nside, int):
        if nside <= 0 or (nside & (nside - 1) != 0 and nside != 0) :
            raise ValueError("nside (if int) must be a positive integer that is a power of 2.")
    return 12 * nside * nside

class HealpixGrid:
    """Manages HEALPix grid information and provides geometry calculations.

    Attributes:
        nside (int): The HEALPix nside parameter.
        npix (int): The total number of pixels in the map (12 * nside^2).
    """
    def __init__(self, nside: int):
        if not isinstance(nside, int) or nside <= 0 or (nside & (nside - 1) != 0 and nside !=0) :
            raise ValueError("nside must be a positive integer that is a power of 2.")
        self.nside = nside
        self.npix = int(nside_to_npix(self.nside))

        self._ring_beta = None
        self._ring_phi0_npix_beta_pol = {}
        self._ring_phi0_npix_beta_eq = {}
        self._pixel_angles_theta_phi = None

    @Partial(jax.jit, static_argnums=(0,))
    def ring_beta(self) -> jnp.ndarray:
        """Returns beta = cos(theta) for all rings."""
        beta_calc = jnp.zeros(4 * self.nside - 1, dtype=jnp.float64)
        ring_i_calc_1based = jnp.arange(1, 4 * self.nside, dtype=jnp.float64)

        idx_north_cap = (ring_i_calc_1based <= self.nside)
        beta_north_values = 1.0 - ring_i_calc_1based**2 / (3.0 * self.nside**2)
        beta_calc = jnp.where(idx_north_cap, beta_north_values, beta_calc)

        idx_south_cap = (ring_i_calc_1based > 3 * self.nside)
        eff_ring_idx_south = 4.0 * self.nside - ring_i_calc_1based
        beta_south_values = -(1.0 - eff_ring_idx_south**2 / (3.0 * self.nside**2))
        beta_calc = jnp.where(idx_south_cap, beta_south_values, beta_calc)

        idx_equatorial = jnp.logical_and(ring_i_calc_1based > self.nside, ring_i_calc_1based <= 3 * self.nside)
        beta_equatorial_values = (4.0/3.0) - (2.0/3.0) * ring_i_calc_1based / self.nside
        beta_calc = jnp.where(idx_equatorial, beta_equatorial_values, beta_calc)

        return beta_calc

    @Partial(jax.jit, static_argnums=(0,))
    def _calculate_ring_pol(self, ring_i_key_jax: jnp.ndarray) -> jnp.ndarray:
        ring_i_calc = jnp.where(ring_i_key_jax < self.nside, ring_i_key_jax, 4.0 * self.nside - ring_i_key_jax)
        phi_0 = (jnp.pi / (2.0 * ring_i_calc)) * 0.5
        npix_in_ring = 4.0 * ring_i_calc
        beta_val = 1.0 - (ring_i_calc**2) / (3.0 * self.nside**2)
        beta_val = jnp.where(ring_i_key_jax < self.nside, beta_val, beta_val * -1.0)
        return jnp.array([phi_0, npix_in_ring, beta_val], dtype=jnp.float64)

    @Partial(jax.jit, static_argnums=(0,))
    def _calculate_ring_eq(self, ring_i_jax: jnp.ndarray) -> jnp.ndarray:
        s = jnp.where((ring_i_jax - self.nside + 1.0) % 2.0 == 0.0, 1.0, 2.0)
        phi_0 = (jnp.pi / (2.0 * self.nside)) * (1.0 - s / 2.0)
        npix_in_ring = 4.0 * self.nside
        beta_val = (4.0 / 3.0) - (2.0 / 3.0) * ring_i_jax / self.nside
        return jnp.array([phi_0, npix_in_ring, beta_val], dtype=jnp.float64)

    def get_ring_properties(self, ring_i) -> jnp.ndarray:
        # Convert Python scalars to JAX arrays for consistency if needed by JITted functions
        if isinstance(ring_i, (int, float)):
            ring_i_jax = jnp.array(float(ring_i), dtype=jnp.float64)
        elif isinstance(ring_i, jnp.ndarray) and ring_i.ndim == 0:
            ring_i_jax = ring_i.astype(jnp.float64)
        else:
            # This path will be taken if ring_i is a tracer that's not 0-dim,
            # or an unsupported type.
            # For JITted callers, ring_i might be a tracer.
            # The check below would cause TracerBoolConversionError if ring_i_jax is a tracer.
            # We remove the explicit Python-style validation for JIT compatibility when called by JITted functions.
            # Validation should happen before JIT or be done with JAX ops.
            pass # Assuming ring_i_jax is now a 0-dim JAX array or becomes one.

        # The problematic 'if' for JIT context is removed.
        # Callers like _get_phi_m_for_ring (if JITted) must ensure ring_i is valid or handle errors.
        # For direct Python calls, the check was useful but breaks JIT if ring_i is a tracer.
        # ring_i_val = ring_i_jax.item() # This would fail if ring_i_jax is a tracer.
        # if not (1.0 <= ring_i_val <= 4.0 * self.nside - 1.0):
        #     raise ValueError(f"ring_i must be between 1 and {4 * self.nside - 1}, got {ring_i_val}")

        is_polar = jnp.logical_or(ring_i_jax < self.nside, ring_i_jax > 3.0 * self.nside)

        return jax.lax.cond(
            is_polar,
            self._calculate_ring_pol,
            self._calculate_ring_eq,
            ring_i_jax
        )

    def pixel_to_angle_ring_colatitude_longitude(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self._pixel_angles_theta_phi is not None and isinstance(self._pixel_angles_theta_phi, tuple):
            return self._pixel_angles_theta_phi

        npix_total_int = self.npix
        nside_int = self.nside

        z_parts = []
        phi_parts = []

        # North Cap Rings
        for r_idx_1based_py_int in range(1, nside_int):
            props = self.get_ring_properties(float(r_idx_1based_py_int))
            phi_0_val, npix_ring_f, beta_val = props[0].item(), props[1].item(), props[2].item() # Use .item()
            npix_ring_c = int(npix_ring_f)

            if npix_ring_c > 0:
                z_ring_vals = jnp.full(npix_ring_c, beta_val, dtype=jnp.float64)
                j_indices_1b = jnp.arange(1, npix_ring_c + 1, dtype=jnp.float64)
                phi_ring_vals = (jnp.pi / (2.0 * float(r_idx_1based_py_int))) * (j_indices_1b - 0.5)
                z_parts.append(z_ring_vals)
                phi_parts.append(phi_ring_vals)

        # Equatorial Rings
        for r_idx_1based_py_int in range(nside_int, 3 * nside_int + 1):
            props = self.get_ring_properties(float(r_idx_1based_py_int))
            phi_0_val, npix_ring_f, beta_val = props[0].item(), props[1].item(), props[2].item() # Use .item()
            npix_ring_c = int(npix_ring_f)

            if npix_ring_c > 0:
                z_ring_vals = jnp.full(npix_ring_c, beta_val, dtype=jnp.float64)
                delta_phi_eq = (2.0 * jnp.pi) / npix_ring_f
                j_indices_0b = jnp.arange(npix_ring_c, dtype=jnp.float64)
                phi_ring_vals = phi_0_val + j_indices_0b * delta_phi_eq
                z_parts.append(z_ring_vals)
                phi_parts.append(phi_ring_vals)

        # South Cap Rings
        south_z_parts_temp = []
        south_phi_parts_temp = []
        for r_idx_north_equiv_py_int in range(1, nside_int):
            r_idx_south_actual_py_float = float(4 * nside_int - r_idx_north_equiv_py_int)
            props = self.get_ring_properties(r_idx_south_actual_py_float)
            phi_0_val, npix_ring_f, beta_val = props[0].item(), props[1].item(), props[2].item() # Use .item()
            npix_ring_c = int(npix_ring_f)

            if npix_ring_c > 0:
                z_ring_vals = jnp.full(npix_ring_c, beta_val, dtype=jnp.float64)
                j_indices_1b = jnp.arange(1, npix_ring_c + 1, dtype=jnp.float64)
                phi_ring_vals = (jnp.pi / (2.0 * float(r_idx_north_equiv_py_int))) * (j_indices_1b - 0.5)
                south_z_parts_temp.append(z_ring_vals)
                south_phi_parts_temp.append(phi_ring_vals)

        if south_z_parts_temp:
            z_parts.extend(s_z for s_z in reversed(south_z_parts_temp))
            phi_parts.extend(s_phi for s_phi in reversed(south_phi_parts_temp))

        if not z_parts:
             final_z = jnp.array([], dtype=jnp.float64)
             final_phi = jnp.array([], dtype=jnp.float64)
        else:
            final_z = jnp.concatenate(z_parts)
            final_phi = jnp.concatenate(phi_parts)

        if final_z.shape[0] != npix_total_int:
             print(f"Warning: pixel_to_angle_ring_colatitude_longitude produced z array of size {final_z.shape[0]} "
                   f"but expected {npix_total_int} for nside={nside_int}. Check logic.")

        final_theta = jnp.arccos(jnp.clip(final_z, -1.0, 1.0))

        self._pixel_angles_theta_phi = (final_theta, final_phi)
        return final_theta, final_phi

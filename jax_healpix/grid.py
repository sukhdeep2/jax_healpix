import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

# Based on nside2npix from healpy
def nside_to_npix(nside):
    if not isinstance(nside, (int, jnp.ndarray)):
        raise ValueError("nside must be an integer or JAX array.")
    if isinstance(nside, jnp.ndarray) and nside.ndim != 0:
        raise ValueError("nside must be a scalar if it's a JAX array.")
    return 12 * nside * nside

class HealpixGrid:
    def __init__(self, nside):
        if not isinstance(nside, int) or nside <= 0 or (nside & (nside - 1) != 0 and nside !=0) :
            raise ValueError("nside must be a positive integer that is a power of 2.")
        self.nside = nside
        self.npix = nside_to_npix(nside)

        # Cache for ring properties
        self._ring_beta = None
        self._ring_phi0_npix_beta_pol = {} # Cache for polar rings: ring_i -> (phi_0, npix, beta)
        self._ring_phi0_npix_beta_eq = {}  # Cache for equatorial rings: ring_i -> (phi_0, npix, beta)
        self._pixel_angles_theta_phi = None

    @partial(jax.jit, static_argnums=(0,))
    def ring_beta(self):
        """
        Return beta=cos(theta) for all rings, given nside.
        Corresponds to ring_beta from SPHT_jax.py
        """
        if self._ring_beta is not None:
            return self._ring_beta

        beta = jnp.zeros(4 * self.nside - 1)
        ring_i = jnp.arange(4 * self.nside - 1) + 1
        l1 = self.nside

        # North pole cap
        beta_north_cap = 1 - ring_i[:l1] ** 2 / (3 * self.nside**2)
        beta = beta.at[:l1].set(beta_north_cap)

        # South pole cap (reversed and negated North cap)
        beta = beta.at[-l1:].set(beta_north_cap[::-1] * -1)

        # Equatorial belt
        beta_equatorial = 4.0 / 3.0 - (2.0 / 3.0) * ring_i[l1:-l1] / self.nside
        beta = beta.at[l1:-l1].set(beta_equatorial)

        self._ring_beta = beta
        return beta

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_ring_pol(self, ring_i_key):
        """
        Ring quantities for a specific polar belt ring_i.
        Corresponds to ring_pol from SPHT_jax.py
        ring_i_key is the original ring_i (1 to nside-1 or 3*nside+1 to 4*nside-1)
        """
        # Normalize ring_i for calculation (1 to nside-1)
        ring_i_calc = jnp.where(ring_i_key < self.nside, ring_i_key, 4 * self.nside - ring_i_key)

        phi_0 = jnp.pi / (2 * ring_i_calc) * 0.5  # for j=1 pixel in the ring
        npix_in_ring = 4 * ring_i_calc

        beta_val = 1 - (ring_i_calc**2) / (3 * self.nside**2)
        beta_val = jnp.where(ring_i_key < self.nside, beta_val, beta_val * -1) # negate for south pole

        return jnp.array([phi_0, npix_in_ring, beta_val])

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_ring_eq(self, ring_i):
        """
        Ring quantities for a specific equatorial belt ring_i.
        Corresponds to ring_eq from SPHT_jax.py
        ring_i is (nside to 3*nside)
        """
        # s determines the phi offset based on whether the ring index relative to the start of eq belt is even or odd
        s = jnp.where((ring_i - self.nside + 1) % 2 == 0, 1, 2) # Matches HEALPix convention for phi offset
        phi_0 = jnp.pi / (2 * self.nside) * (1.0 - s / 2.0) # for j=1 pixel in the ring
        npix_in_ring = 4 * self.nside # All equatorial rings have 4*nside pixels

        beta_val = (4.0 / 3.0) - (2.0 / 3.0) * ring_i / self.nside
        return jnp.array([phi_0, npix_in_ring, beta_val])

    def get_ring_properties(self, ring_i):
        """
        Get phi_0, npix_in_ring, and beta for a given ring_i.
        ring_i is 1-indexed.
        """
        if not (1 <= ring_i <= 4 * self.nside - 1):
            raise ValueError(f"ring_i must be between 1 and {4 * self.nside - 1}")

        is_polar = jnp.logical_or(ring_i < self.nside, ring_i > 3 * self.nside)

        # For JIT compilation, we can't cache directly based on ring_i inside a JITted context
        # if it's not a static argument.
        # However, the _calculate methods are JITted.
        # Caching here is more for Python-level calls if the same ring is requested multiple times.

        if is_polar:
            # JAX device arrays cannot be used as dict keys directly if they are not static.
            # For caching, we'd typically convert to Python int if ring_i is a JAX scalar.
            # However, to keep it JAX-friendly for potential vmap, we might need a different caching strategy
            # or accept that caching happens at a higher level or not at all for individual vmapped calls.
            # For now, let's assume ring_i is a Python int for caching keys.
            ring_i_int = int(ring_i)
            if ring_i_int not in self._ring_phi0_npix_beta_pol:
                self._ring_phi0_npix_beta_pol[ring_i_int] = self._calculate_ring_pol(ring_i)
            return self._ring_phi0_npix_beta_pol[ring_i_int]
        else:
            ring_i_int = int(ring_i)
            if ring_i_int not in self._ring_phi0_npix_beta_eq:
                 self._ring_phi0_npix_beta_eq[ring_i_int] = self._calculate_ring_eq(ring_i)
            return self._ring_phi0_npix_beta_eq[ring_i_int]

    @partial(jax.jit, static_argnums=(0,))
    def pixel_to_angle_ring_colatitude_longitude(self):
        """
        Calculates the colatitude (theta) and longitude (phi) for all pixels.
        This is a JAX adaptation of parts of hp_pix2ang from Healpy_test_fn.py
        Returns:
            theta (jnp.array): Colatitude for each pixel.
            phi (jnp.array): Longitude for each pixel.
        """
        if self._pixel_angles_theta_phi is not None:
            return self._pixel_angles_theta_phi

        npix = self.npix
        nside = self.nside

        # Ring indices (1-based) and pixel indices within rings (j, 1-based)
        # This part is complex to vectorize directly in JAX without explicit loops or scatters,
        # mimicking healpy's pix2ang logic.
        # Healpy's internal logic is quite intricate.
        # For a pure JAX version, one might need to re-derive or simplify.
        # Let's attempt a direct translation of the logic found in hp_pix2ang.

        z = jnp.zeros(npix)
        phi = jnp.zeros(npix)

        # North Polar Cap
        # Pixels 0 to 2*nside*(nside-1) - 1
        # ring_i goes from 1 to nside-1
        # For each ring_i, there are 4*i pixels
        # lastpix_north = 2 * nside * (nside - 1) # Exclusive end

        # This section requires careful handling of array indexing and assignments
        # which can be tricky to do efficiently and correctly in a JITted JAX function
        # without dynamic sized arrays or explicit loops that JAX can unroll.

        # A simplified approach for getting (z, phi) for all pixels:
        # 1. Determine ring index for each pixel
        # 2. Determine pixel index within ring (j) for each pixel
        # 3. Apply formulas based on ring type (polar/equatorial)

        # This is a placeholder for the full pixel_to_angle logic.
        # Implementing a full, efficient JAX version of pix2ang from scratch
        # that matches healpy bit-for-bit is non-trivial and might be beyond
        # a single step if aiming for high performance without loops.

        # Fallback: use Healpy for now if a direct JAX port is too complex for this step
        # and mark for future improvement. Or, use the existing z, phi from hp_pix2ang
        # if that's acceptable (though it uses numpy).

        # For the purpose of this refactoring step, we'll define the structure.
        # A full JAX port of pix2ang might be a separate, detailed task.
        # Let's assume we have a way to get z and phi for now.
        # As an example, let's use the ring betas for z, and a simplified phi.
        # THIS IS A SIMPLIFIED AND INCOMPLETE VERSION OF PIX2ANG.

        indices = jnp.arange(npix)

        # The following is a highly simplified placeholder and NOT a correct pix2ang implementation.
        # It's here to allow the class structure to be built.
        # A proper implementation would involve mapping each ipix to its (ring, offset_in_ring)
        # and then applying the correct z and phi formulae.

        # Example: Get ring index for each pixel (this is also non-trivial in JAX without healpy.pix2ring)
        # For now, we cannot fully implement this without more complex logic or depending on healpy.
        # We will leave this method to be fully implemented in a later stage or assume
        # that if precise pixel_to_angle is needed, it might call out to healpy or use
        # precomputed values if possible.

        # For now, let's return arrays of zeros and mark for completion.
        # raise NotImplementedError("Full JAX pixel_to_angle_ring_colatitude_longitude is not yet implemented.")
        # Returning zeros to allow class to be used, but this method is incomplete.
        # A proper implementation would calculate z and phi for each pixel.
        # z = cos(theta)

        # To make progress, we will adapt the logic from hp_pix2ang directly here.
        # This will be verbose but aims to replicate the numpy logic in JAX.

        ipix = jnp.arange(npix)

        # Determine ring index (i_ring) and intra-ring pixel index (j_pix) for each global pixel index ipix
        # This is the most complex part. Healpy uses a lookup for this.
        # We are trying to reproduce the "forward" calculation (pix -> ang)

        # North polar cap
        # ring_i from 1 to nside-1
        # npix_in_ring = 4 * ring_i
        # cumulative_pix_north = 2 * nside * (nside-1)

        # Equatorial belt
        # ring_i from nside to 3*nside
        # npix_in_ring = 4 * nside
        # cumulative_pix_eq = cumulative_pix_north + (2*nside+1)*4*nside

        # South polar cap
        # ring_i from 3*nside+1 to 4*nside-1
        # (mirrors north cap)

        # This direct translation of healpy's pix2ang pixel iteration logic is hard in JAX
        # without loops. A more common JAX approach would be to define the geometry
        # per ring and then map pixels to these.

        # Let's use the structure from hp_pix2ang in Healpy_test_fn.py, adapted for JAX.
        # This is a direct, somewhat verbose, translation.

        # Temporary arrays for ring indices and intra-ring pixel counts
        # This is a challenging part to do purely in JAX without loops or precomputed indices.
        # For now, this method will be marked as needing a full JAX-native implementation.
        # The original hp_pix2ang in the provided files is numpy-based.

        # To allow progress, we will implement a simplified version that gets unique z values
        # (from ring_beta) and a placeholder for phi.
        # A full pix2ang is a substantial piece of work.

        # Placeholder implementation:
        # This is NOT a full pix2ang. It demonstrates structure.
        # A full implementation would need to correctly map each of the 12*nside^2 pixels
        # to their unique theta and phi.

        # The z values (cos_theta) are constant per ring.
        # We can get the unique z values from self.ring_beta()
        # The phi values change within each ring.

        # For now, this method will remain incomplete as a full JAX pix2ang is complex.
        # We will focus on the ring properties first.
        # If this method is critical for subsequent steps, we might need to use
        # a numpy version temporarily or simplify its requirements.

        # Let's assume for now that this function will be properly implemented later
        # or an alternative approach will be used if exact pixel angles are needed by other classes.
        # For the purpose of class structure, we define it.

        # A more JAX-idiomatic way to get all pixel coordinates might involve
        # generating coordinates for each ring type and then concatenating.

        # North Cap Rings: ring_i from 1 to nside-1
        z_north_cap_rings = []
        phi_north_cap_rings = []
        for r_idx in range(1, nside): # Python loop for JAX array construction
            ring_i = r_idx
            _, npix_ring, beta_ring = self._calculate_ring_pol(ring_i)
            npix_ring = int(npix_ring) # for arange
            z_ring = jnp.full(npix_ring, beta_ring)

            # phi for this ring: pi/(2*ring_i) * (j - 0.5) for j=1 to 4*ring_i
            j_indices = jnp.arange(1, npix_ring + 1)
            phi_ring = (jnp.pi / (2 * ring_i)) * (j_indices - 0.5)

            z_north_cap_rings.append(z_ring)
            phi_north_cap_rings.append(phi_ring)

        # Equatorial Rings: ring_i from nside to 3*nside
        z_eq_rings = []
        phi_eq_rings = []
        for r_idx in range(nside, 3 * nside + 1): # Python loop
            ring_i = r_idx
            _, npix_ring, beta_ring = self._calculate_ring_eq(ring_i)
            npix_ring = int(npix_ring)

            z_ring = jnp.full(npix_ring, beta_ring)

            s = 1 if (ring_i - nside + 1) % 2 == 0 else 2 # Matches healpy's s=2 for odd, s=1 for even row number in belt
            if (ring_i - nside) % 2 == 0: # (ring_i - nside + 1) is odd, s=2 in healpy's (j_pix - s/2)
                s_factor = 1.0 # (j - 1.0)
            else: # (ring_i - nside + 1) is even, s=1 in healpy's (j_pix - s/2)
                s_factor = 0.5 # (j - 0.5)

            j_indices = jnp.arange(1, npix_ring + 1)
            # phi for this ring: pi/(2*nside) * (j - s/2)
            # s = (ring_i - nside + 1) % 2, if 0 then s=2 (healpy)
            s_healpy = 2 if (ring_i - nside + 1) % 2 == 0 else 1
            phi_ring = (jnp.pi / (2 * nside)) * (j_indices - s_healpy / 2.0)

            z_eq_rings.append(z_ring)
            phi_eq_rings.append(phi_ring)

        # South Cap Rings: ring_i from 3*nside+1 to 4*nside-1
        # These mirror the North Cap rings
        z_south_cap_rings = []
        phi_south_cap_rings = []
        for r_idx in range(1, nside): # Iterate like North Cap but use corresponding South ring_i
            # Corresponding south ring_i for north ring_i=r_idx is (4*nside - r_idx)
            ring_i_south_equiv = 4 * self.nside - r_idx
            # Calculation uses effective r_idx for geometry
            ring_i_calc_equiv = r_idx # The geometry (npix_in_ring, phi structure) is like r_idx

            _, npix_ring, beta_ring = self._calculate_ring_pol(ring_i_south_equiv) # beta is correctly negative
            npix_ring = int(npix_ring)

            z_ring = jnp.full(npix_ring, beta_ring)

            j_indices = jnp.arange(1, npix_ring + 1)
            phi_ring = (jnp.pi / (2 * ring_i_calc_equiv)) * (j_indices - 0.5) # Same phi structure as north

            z_south_cap_rings.append(z_ring)
            phi_south_cap_rings.append(phi_ring)

        # Concatenate all pixel coordinates
        # The order must match HEALPix RING ordering.
        # North polar cap rings are ordered from r_idx=1 to nside-1
        # Equatorial rings are ordered from nside to 3*nside
        # South polar cap rings are ordered from 3*nside+1 to 4*nside-1 (which means r_idx from nside-1 down to 1)

        final_z = jnp.concatenate(
            [arr for arr in z_north_cap_rings] + \
            [arr for arr in z_eq_rings] + \
            [arr for arr in reversed(z_south_cap_rings)] # South cap pixels are ordered by ring index descending in geometry
        )
        final_phi = jnp.concatenate(
            [arr for arr in phi_north_cap_rings] + \
            [arr for arr in phi_eq_rings] + \
            [arr for arr in reversed(phi_south_cap_rings)]
        )

        # z is cos(theta), so theta = arccos(z)
        final_theta = jnp.arccos(final_z)

        self._pixel_angles_theta_phi = (final_theta, final_phi)
        return final_theta, final_phi

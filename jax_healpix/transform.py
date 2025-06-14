import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial
from jax import jit

# Assuming grid.py and ylm.py are in the same directory (jax_healpix)
from .grid import HealpixGrid
from .ylm import SphericalHarmonics

# RING_ITER_SIZE was used in SPHT_jax.py for loop chunking.
# This can be a class attribute or a parameter if needed.
DEFAULT_RING_ITER_SIZE = 16

class HealpixTransformer:
    def __init__(self, nside, l_max, ring_iter_size=DEFAULT_RING_ITER_SIZE, ylm_use_log_recurrence=False):
        if not isinstance(nside, int) or nside <= 0: # Basic check, HealpixGrid does more
            raise ValueError("nside must be a positive integer.")
        if not isinstance(l_max, int) or l_max < 0:
            raise ValueError("l_max must be a non-negative integer.")

        self.nside = nside
        self.l_max = l_max
        self.ring_iter_size = ring_iter_size

        self.grid = HealpixGrid(nside=self.nside)

        self.spins_for_ylm_init = (0, 2, -2)
        self.ylm_calculator = SphericalHarmonics(
            l_max=self.l_max,
            spins_to_compute=self.spins_for_ylm_init,
            use_log_recurrence=ylm_use_log_recurrence
        )

        self.ring_beta_values = self.grid.ring_beta()

    @partial(jit, static_argnums=(0,))
    def _get_phi_m_for_ring(self, ring_i, phase_sign):
        phi_0, npix_in_ring, beta_val = self.grid.get_ring_properties(ring_i)
        npix_in_ring = npix_in_ring.astype(jnp.int32)

        m_vals = jnp.arange(self.l_max + 1)
        j_pix_indices = jnp.arange(npix_in_ring)

        phis_in_ring = phi_0 + (2 * jnp.pi * j_pix_indices) / npix_in_ring
        m_phi = m_vals[:, None] * phis_in_ring[None, :]

        phi_factors = jnp.exp(1j * phase_sign * m_phi)
        return phi_factors, beta_val, npix_in_ring

    @partial(jit, static_argnums=(0,))
    def _map2alm_ring_contribution(self, ring_i, maps_TQU_0, maps_TQU_2, maps_TQU_neg2, current_alm_0, current_alm_2, current_alm_neg2):
        alm_0_contrib = jnp.zeros_like(current_alm_0)
        alm_2_contrib = jnp.zeros_like(current_alm_2)
        alm_neg2_contrib = jnp.zeros_like(current_alm_neg2)

        phi_factors_north, _, npix_north = self._get_phi_m_for_ring(ring_i, phase_sign=-1.0)
        beta_north_val_scalar = self.ring_beta_values[ring_i-1]

        ylm_north = {}
        for s_ylm in self.spins_for_ylm_init:
            if self.ylm_calculator.use_log_recurrence:
                ylm_val, ylm_s = self.ylm_calculator.get_ylm(s_ylm, jnp.array([beta_north_val_scalar]))
                ylm_north[s_ylm] = (ylm_val * ylm_s).squeeze(axis=-1)
            else:
                ylm_val_raw = self.ylm_calculator.get_ylm(s_ylm, jnp.array([beta_north_val_scalar]))
                ylm_north[s_ylm] = ylm_val_raw.squeeze(axis=-1)

        if maps_TQU_0 is not None and current_alm_0 is not None:
            map_T_ring_north = maps_TQU_0[..., ring_i-1, 0:npix_north]
            G_m_T_north = jnp.dot(map_T_ring_north, jnp.conjugate(phi_factors_north.T))
            alm_0_contrib += G_m_T_north[None, :] * ylm_north[0]

        if maps_TQU_2 is not None and maps_TQU_neg2 is not None and current_alm_2 is not None and current_alm_neg2 is not None:
            map_Q_ring_north = maps_TQU_2[..., ring_i-1, 0:npix_north]
            map_U_ring_north = maps_TQU_neg2[..., ring_i-1, 0:npix_north]
            G_m_Q_north = jnp.dot(map_Q_ring_north, jnp.conjugate(phi_factors_north.T))
            G_m_U_north = jnp.dot(map_U_ring_north, jnp.conjugate(phi_factors_north.T))

            alm_2_contrib = (G_m_Q_north[None,:] - 1j * G_m_U_north[None,:]) * ylm_north[2]
            alm_neg2_contrib = (G_m_Q_north[None,:] + 1j * G_m_U_north[None,:]) * ylm_north[-2]

        if ring_i < 2 * self.nside :
            south_ring_actual_idx = (4 * self.nside) - ring_i
            phi_factors_south, _, npix_south = self._get_phi_m_for_ring(south_ring_actual_idx, phase_sign=-1.0)

            ylm_south = {}
            l_indices = jnp.arange(self.l_max + 1)[:, None]
            m_indices = jnp.arange(self.l_max + 1)[None, :]
            parity_factor = (-1)**(l_indices + m_indices)

            for s_ylm_val in self.spins_for_ylm_init:
                ylm_south[s_ylm_val] = ylm_north[s_ylm_val] * parity_factor
                if s_ylm_val == -2:
                    ylm_south[s_ylm_val] *= -1.0

            if maps_TQU_0 is not None and current_alm_0 is not None:
                map_T_ring_south = maps_TQU_0[..., south_ring_actual_idx-1, 0:npix_south]
                G_m_T_south = jnp.dot(map_T_ring_south, jnp.conjugate(phi_factors_south.T))
                alm_0_contrib += G_m_T_south[None, :] * ylm_south[0]

            if maps_TQU_2 is not None and maps_TQU_neg2 is not None and current_alm_2 is not None and current_alm_neg2 is not None:
                map_Q_ring_south = maps_TQU_2[..., south_ring_actual_idx-1, 0:npix_south]
                map_U_ring_south = maps_TQU_neg2[..., south_ring_actual_idx-1, 0:npix_south]
                G_m_Q_south = jnp.dot(map_Q_ring_south, jnp.conjugate(phi_factors_south.T))
                G_m_U_south = jnp.dot(map_U_ring_south, jnp.conjugate(phi_factors_south.T))

                alm_2_contrib += (G_m_Q_south[None,:] - 1j*G_m_U_south[None,:]) * ylm_south[2]
                alm_neg2_contrib += (G_m_Q_south[None,:] + 1j*G_m_U_south[None,:]) * ylm_south[-2]

        return alm_0_contrib, alm_2_contrib, alm_neg2_contrib

    def map2alm(self, maps_dict, spins_in_map):
        alm_output = {}
        ref_map_spin = spins_in_map[0]
        batch_shape = maps_dict[ref_map_spin].shape[:-2]

        maps_TQU_0_data = maps_dict.get(0)
        maps_TQU_2_data = maps_dict.get(2)
        maps_TQU_neg2_data = maps_dict.get(-2)

        current_alm_0_data = jnp.zeros(batch_shape + (self.l_max+1, self.l_max+1), dtype=jnp.complex64) if 0 in spins_in_map else None
        current_alm_2_data = jnp.zeros(batch_shape + (self.l_max+1, self.l_max+1), dtype=jnp.complex64) if 2 in spins_in_map else None
        current_alm_neg2_data = jnp.zeros(batch_shape + (self.l_max+1, self.l_max+1), dtype=jnp.complex64) if -2 in spins_in_map else None

        def loop_body_map2alm_scan(carry, r_idx_1based):
            prev_alm_0, prev_alm_2, prev_alm_neg2 = carry
            contrib_0, contrib_2, contrib_neg2 = self._map2alm_ring_contribution(
                r_idx_1based,
                maps_TQU_0_data, maps_TQU_2_data, maps_TQU_neg2_data,
                prev_alm_0, prev_alm_2, prev_alm_neg2
            )
            new_alm_0 = prev_alm_0 + contrib_0 if prev_alm_0 is not None else None
            new_alm_2 = prev_alm_2 + contrib_2 if prev_alm_2 is not None else None
            new_alm_neg2 = prev_alm_neg2 + contrib_neg2 if prev_alm_neg2 is not None else None
            return (new_alm_0, new_alm_2, new_alm_neg2), None

        initial_carry = (current_alm_0_data, current_alm_2_data, current_alm_neg2_data)
        final_alms, _ = jax.lax.scan(
            loop_body_map2alm_scan,
            initial_carry,
            jnp.arange(1, 2 * self.nside + 1)
        )
        final_alm_0, final_alm_2, final_alm_neg2 = final_alms

        if final_alm_0 is not None: alm_output[0] = final_alm_0
        if final_alm_2 is not None: alm_output[2] = final_alm_2
        if final_alm_neg2 is not None: alm_output[-2] = final_alm_neg2

        pix_area = 4 * jnp.pi / (12 * self.nside**2)
        for s_key in alm_output:
            if alm_output[s_key] is not None:
                 alm_output[s_key] *= pix_area

        if 2 in alm_output and alm_output[2] is not None:
            alm_output[2] *= -0.5
        if -2 in alm_output and alm_output[-2] is not None:
            alm_output[-2] *= (0.5j)

        return alm_output

    @partial(jit, static_argnums=(0,))
    def _alm2map_ring_contribution(self, ring_i, alm_T, alm_E, alm_B, current_map_T, current_map_Q, current_map_U):
        phi_factors_north, _, npix_north = self._get_phi_m_for_ring(ring_i, phase_sign=+1.0)
        beta_north_val_scalar = self.ring_beta_values[ring_i - 1]

        ylm_north = {}
        for s_ylm in self.spins_for_ylm_init:
            if self.ylm_calculator.use_log_recurrence:
                ylm_val, ylm_s = self.ylm_calculator.get_ylm(s_ylm, jnp.array([beta_north_val_scalar]))
                ylm_north[s_ylm] = (ylm_val * ylm_s).squeeze(axis=-1)
            else:
                ylm_val_raw = self.ylm_calculator.get_ylm(s_ylm, jnp.array([beta_north_val_scalar]))
                ylm_north[s_ylm] = ylm_val_raw.squeeze(axis=-1)

        map_T_ring_contrib = jnp.zeros_like(current_map_T)
        map_Q_ring_contrib = jnp.zeros_like(current_map_Q)
        map_U_ring_contrib = jnp.zeros_like(current_map_U)

        if alm_T is not None and current_map_T is not None:
            F_m_T_north = jnp.sum(alm_T * ylm_north[0], axis=-2)
            map_contrib_T_north_pix = jnp.dot(F_m_T_north, phi_factors_north)
            map_T_ring_contrib = map_T_ring_contrib.at[..., ring_i-1, 0:npix_north].set(jnp.real(map_contrib_T_north_pix))

        if alm_E is not None and alm_B is not None and current_map_Q is not None and current_map_U is not None:
            term_E_ylm2 = jnp.sum(alm_E * ylm_north[2], axis=-2)
            term_B_ylm2 = jnp.sum(alm_B * ylm_north[2], axis=-2)
            term_E_ylm_neg2 = jnp.sum(alm_E * ylm_north[-2], axis=-2)
            term_B_ylm_neg2 = jnp.sum(alm_B * ylm_north[-2], axis=-2)

            map_complex_Q_prime_north = jnp.dot(term_E_ylm2 + 1j * term_B_ylm2, phi_factors_north)
            map_complex_U_prime_north = jnp.dot(term_E_ylm_neg2 - 1j * term_B_ylm_neg2, phi_factors_north)

            map_Q_ring_contrib = map_Q_ring_contrib.at[...,ring_i-1,0:npix_north].set(
                jnp.real(map_complex_Q_prime_north - map_complex_U_prime_north)
            )
            map_U_ring_contrib = map_U_ring_contrib.at[...,ring_i-1,0:npix_north].set(
                jnp.imag(map_complex_Q_prime_north + map_complex_U_prime_north)
            )

        if ring_i < 2 * self.nside :
            south_ring_actual_idx = (4 * self.nside) - ring_i
            phi_factors_south, _, npix_south = self._get_phi_m_for_ring(south_ring_actual_idx, phase_sign=+1.0)

            ylm_south = {}
            l_indices = jnp.arange(self.l_max + 1)[:, None]
            m_indices = jnp.arange(self.l_max + 1)[None, :]
            parity_factor = (-1)**(l_indices + m_indices)
            for s_ylm_val in self.spins_for_ylm_init:
                ylm_south[s_ylm_val] = ylm_north[s_ylm_val] * parity_factor
                if s_ylm_val == -2:
                    ylm_south[s_ylm_val] *= -1.0

            if alm_T is not None and current_map_T is not None:
                F_m_T_south = jnp.sum(alm_T * ylm_south[0], axis=-2)
                map_contrib_T_south_pix = jnp.dot(F_m_T_south, phi_factors_south)
                map_T_ring_contrib = map_T_ring_contrib.at[..., south_ring_actual_idx-1, 0:npix_south].set(jnp.real(map_contrib_T_south_pix))

            if alm_E is not None and alm_B is not None and current_map_Q is not None and current_map_U is not None:
                term_E_ylm2_s = jnp.sum(alm_E * ylm_south[2], axis=-2)
                term_B_ylm2_s = jnp.sum(alm_B * ylm_south[2], axis=-2)
                term_E_ylm_neg2_s = jnp.sum(alm_E * ylm_south[-2], axis=-2)
                term_B_ylm_neg2_s = jnp.sum(alm_B * ylm_south[-2], axis=-2)

                map_complex_Q_prime_south = jnp.dot(term_E_ylm2_s + 1j*term_B_ylm2_s, phi_factors_south)
                map_complex_U_prime_south = jnp.dot(term_E_ylm_neg2_s - 1j*term_B_ylm_neg2_s, phi_factors_south)

                map_Q_ring_contrib = map_Q_ring_contrib.at[...,south_ring_actual_idx-1,0:npix_south].set(
                    jnp.real(map_complex_Q_prime_south - map_complex_U_prime_south)
                )
                map_U_ring_contrib = map_U_ring_contrib.at[...,south_ring_actual_idx-1,0:npix_south].set(
                    jnp.imag(map_complex_Q_prime_south + map_complex_U_prime_south)
                )

        return map_T_ring_contrib, map_Q_ring_contrib, map_U_ring_contrib

    def alm2map(self, alm_dict, spins_to_map):
        map_output = {}
        ref_alm_spin = list(alm_dict.keys())[0]
        batch_shape = alm_dict[ref_alm_spin].shape[:-2]
        map_shape = batch_shape + (4 * self.nside - 1, 4 * self.nside)

        internal_alm_dict = {}
        for s_alm_key_proc in alm_dict:
            alm = alm_dict[s_alm_key_proc]
            m_indices_map = jnp.arange(alm.shape[-1])
            scaled_alm = jnp.where(m_indices_map[None,:] > 0, alm * 2.0, alm)
            if s_alm_key_proc == -2:
                scaled_alm = scaled_alm / 1j
            internal_alm_dict[s_alm_key_proc] = scaled_alm

        alm_T_data = internal_alm_dict.get(0)
        alm_E_data = internal_alm_dict.get(2)
        alm_B_data_proc = internal_alm_dict.get(-2)

        generate_T = 0 in spins_to_map
        generate_QU = 2 in spins_to_map or -2 in spins_to_map

        current_map_T_data = jnp.zeros(map_shape, dtype=jnp.float32) if generate_T else None
        current_map_Q_data = jnp.zeros(map_shape, dtype=jnp.float32) if generate_QU else None
        current_map_U_data = jnp.zeros(map_shape, dtype=jnp.float32) if generate_QU else None

        def loop_body_alm2map_scan(carry, r_idx_1based):
            prev_map_T, prev_map_Q, prev_map_U = carry
            contrib_T, contrib_Q, contrib_U = self._alm2map_ring_contribution(
                r_idx_1based,
                alm_T_data, alm_E_data, alm_B_data_proc,
                prev_map_T, prev_map_Q, prev_map_U
            )
            new_map_T = prev_map_T + contrib_T if prev_map_T is not None else None
            new_map_Q = prev_map_Q + contrib_Q if prev_map_Q is not None else None
            new_map_U = prev_map_U + contrib_U if prev_map_U is not None else None
            return (new_map_T, new_map_Q, new_map_U), None

        initial_carry_map = (current_map_T_data, current_map_Q_data, current_map_U_data)
        final_maps, _ = jax.lax.scan(
            loop_body_alm2map_scan,
            initial_carry_map,
            jnp.arange(1, 2 * self.nside + 1)
        )
        final_map_T, final_map_Q, final_map_U = final_maps

        if generate_T and final_map_T is not None:
            map_output[0] = final_map_T
        if generate_QU and final_map_Q is not None:
            map_output[2] = final_map_Q
        if generate_QU and final_map_U is not None:
            map_output[-2] = final_map_U

        return map_output

    # --- Static helper methods for alm reshaping ---
    @staticmethod
    @partial(jit, static_argnames=("l_max",))
    def _alm_indxs(l_max):
        n_coeffs = ((l_max + 1) * (l_max + 2)) // 2
        l_primary_to_healpy_1d = jnp.zeros(n_coeffs, dtype=jnp.int32)
        count = 0
        idx_map_lm_to_healpy1d = HealpixTransformer._set_m_indxs(l_max)
        for l_val in range(l_max + 1):
            for m_val in range(l_val + 1):
                l_primary_to_healpy_1d = l_primary_to_healpy_1d.at[count].set(idx_map_lm_to_healpy1d[l_val, m_val])
                count += 1
        return l_primary_to_healpy_1d

    @staticmethod
    @partial(jit, static_argnames=("l_max",))
    def _set_m_indxs(l_max):
        idx_map = jnp.zeros((l_max + 1, l_max + 1), dtype=jnp.int32)
        current_idx = 0
        for m_val in range(l_max + 1):
            for l_val in range(m_val, l_max + 1):
                idx_map = idx_map.at[l_val, m_val].set(current_idx)
                current_idx += 1
        return idx_map

    @classmethod
    def reshape_alm_1d_to_2d(cls, l_max, alm_1d):
        batch_shape = alm_1d.shape[:-1]
        output_shape = batch_shape + (l_max + 1, l_max + 1)
        alm_2d = jnp.zeros(output_shape, dtype=alm_1d.dtype)
        indxs_map_2d_to_1d = cls._set_m_indxs(l_max)
        for l_val in range(l_max + 1):
            for m_val in range(l_val + 1):
                one_d_idx = indxs_map_2d_to_1d[l_val, m_val]
                alm_2d = alm_2d.at[..., l_val, m_val].set(alm_1d[..., one_d_idx])
        return alm_2d

    @staticmethod
    @partial(jit, static_argnames=("l_max",))
    def _l_stack_mask(l_max):
        return jnp.array([l_idx for l_idx in range(l_max + 1) for _ in range(l_idx + 1)])

    @staticmethod
    @partial(jit, static_argnames=("l_max",))
    def _m_stack_mask(l_max):
        return jnp.array([m_idx for l_idx in range(l_max + 1) for m_idx in range(l_idx + 1)])

    @classmethod
    def stack_alm_2d_to_1d(cls, l_max, alm_2d, order='m'):
        if order not in ['m', 'l']:
            raise ValueError("Order must be 'm' or 'l'.")
        is_dict = isinstance(alm_2d, dict)
        input_map = alm_2d if is_dict else {0: alm_2d}
        output_map = {}
        n_coeffs = ((l_max + 1) * (l_max + 2)) // 2
        for spin, current_alm_2d in input_map.items():
            batch_shape = current_alm_2d.shape[:-2]
            alm_1d_shape = batch_shape + (n_coeffs,)
            current_alm_1d = jnp.zeros(alm_1d_shape, dtype=current_alm_2d.dtype)
            if order == 'm':
                indxs_map_2d_to_1d = cls._set_m_indxs(l_max)
                for l_val in range(l_max + 1):
                    for m_val in range(l_val + 1):
                        one_d_idx = indxs_map_2d_to_1d[l_val, m_val]
                        current_alm_1d = current_alm_1d.at[..., one_d_idx].set(current_alm_2d[..., l_val, m_val])
            elif order == 'l':
                l_indices = cls._l_stack_mask(l_max)
                m_indices = cls._m_stack_mask(l_max)
                current_alm_1d = current_alm_2d[..., l_indices, m_indices]
            output_map[spin] = current_alm_1d
        return output_map if is_dict else output_map[0]

    # --- Instance methods for map reshaping ---
    @staticmethod
    @partial(jit, static_argnames=("nside",))
    def _pol_ring_indxs(nside):
        if nside == 1:
             return jnp.array([], dtype=jnp.int32)
        north_rings = jnp.arange(nside - 1)
        south_rings_start_idx = 3 * nside -1
        south_rings = jnp.arange(south_rings_start_idx, 4 * nside - 1)
        return jnp.concatenate([north_rings, south_rings])

    @staticmethod
    @partial(jit, static_argnames=("nside",))
    def _eq_ring_indxs(nside):
        start_idx = nside - 1
        end_idx = 3 * nside -1
        return jnp.arange(start_idx, end_idx + 1)

    def convert_map_1d_to_2d(self, map_1d):
        npix = 12 * self.nside * self.nside
        if map_1d.shape[-1] != npix:
            raise ValueError(f"Input map_1d last dimension should be {npix}, got {map_1d.shape[-1]}")
        batch_shape = map_1d.shape[:-1]
        map_2d_shape = batch_shape + (4 * self.nside - 1, 4 * self.nside)
        map_2d = jnp.zeros(map_2d_shape, dtype=map_1d.dtype)
        pix_offset = 0
        for r in range(1, 4 * self.nside):
            ring_idx_0_based = r - 1
            is_polar = r < self.nside or r > 3 * self.nside
            if is_polar:
                ring_i_calc = r if r < self.nside else 4 * self.nside - r
                pixels_in_ring = 4 * ring_i_calc
            else:
                pixels_in_ring = 4 * self.nside
            map_slice = map_1d[..., pix_offset : pix_offset + pixels_in_ring]
            map_2d = map_2d.at[..., ring_idx_0_based, 0:pixels_in_ring].set(map_slice)
            pix_offset += pixels_in_ring
        return map_2d

    @staticmethod
    @partial(jit, static_argnames=("nside",))
    def _ring_pol_mask(nside):
        n_rings_total = 4 * nside - 1
        max_pix_per_ring = 4 * nside
        mask = jnp.zeros((n_rings_total, max_pix_per_ring), dtype=bool)
        for r_idx_0based in range(n_rings_total):
            r_phys = r_idx_0based + 1
            is_polar = r_phys < nside or r_phys > 3 * nside
            if is_polar:
                ring_i_calc = r_phys if r_phys < nside else 4 * nside - r_phys
                pixels_in_ring = 4 * ring_i_calc
                mask = mask.at[r_idx_0based, 0:pixels_in_ring].set(True)
            else:
                pixels_in_ring = 4 * nside
                mask = mask.at[r_idx_0based, 0:pixels_in_ring].set(True)
        return mask

    def convert_map_2d_to_1d(self, map_2d):
        expected_shape_suffix = (4 * self.nside - 1, 4 * self.nside)
        if map_2d.shape[-2:] != expected_shape_suffix:
            raise ValueError(f"Input map_2d last two dimensions should be {expected_shape_suffix}, got {map_2d.shape[-2:]}")
        batch_shape = map_2d.shape[:-2]
        num_rings = map_2d.shape[-2]
        npix = 12 * self.nside * self.nside
        output_list = []
        for r_idx_0based in range(num_rings):
            r_phys = r_idx_0based + 1
            is_polar = r_phys < self.nside or r_phys > 3 * self.nside
            if is_polar:
                ring_i_calc = r_phys if r_phys < self.nside else 4 * self.nside - r_phys
                pixels_in_ring = 4 * ring_i_calc
            else:
                pixels_in_ring = 4 * self.nside
            ring_slice = map_2d[..., r_idx_0based, 0:pixels_in_ring]
            output_list.append(ring_slice)
        map_1d = jnp.concatenate(output_list, axis=-1)
        return map_1d.reshape(batch_shape + (npix,))

    # --- Method for Cl computation ---
    @staticmethod
    @partial(jit, static_argnums=(0,)) # l_max is static
    def _compute_cl_from_single_alm_set(l_max, alm1_coeffs, alm2_coeffs_optional=None):
        # alm_coeffs are expected to be [..., l_max+1(L), l_max+1(M)]

        # Validate shapes roughly, full validation might be too complex for JIT
        # if alm1_coeffs.ndim < 2: # This check might not JIT well.
            # raise ValueError("alm1_coeffs must have at least 2 dimensions (L,M).")
            # Consider removing runtime checks for JIT or using jax.debug.print for checks.

        # Factor for m>0 modes. Assumes m is the last dimension.
        # alm_X_processed = alm_X_coeffs for m=0
        # alm_X_processed = alm_X_coeffs * sqrt(2) for m>0

        # Process alm1
        alm1_processed = alm1_coeffs
        if l_max > 0: # only apply if there are m>0 modes
            # This applies sqrt(2) to m=1, ..., l_max columns for all l rows
            alm1_processed = alm1_processed.at[..., 0].set(alm1_coeffs[..., 0]) # m=0 unchanged
            alm1_processed = alm1_processed.at[..., 1:].set(alm1_coeffs[..., 1:] * jnp.sqrt(2.0))

        if alm2_coeffs_optional is None:
            alm2_processed = alm1_processed # Auto-correlation
        else:
            # if alm2_coeffs_optional.ndim < 2: # Similar JIT concern
                # raise ValueError("alm2_coeffs_optional must have at least 2 dimensions (L,M).")
            alm2_processed = alm2_coeffs_optional
            if l_max > 0:
                alm2_processed = alm2_processed.at[..., 0].set(alm2_coeffs_optional[..., 0])
                alm2_processed = alm2_processed.at[..., 1:].set(alm2_coeffs_optional[..., 1:] * jnp.sqrt(2.0))

        # Sum over m: real(alm1 * conj(alm2))
        # Resulting shape should be [..., l_max+1 (L)]
        cl_val = jnp.real( (alm1_processed * jnp.conjugate(alm2_processed)).sum(axis=-1) )

        # Denominator (2l+1)
        l_values = jnp.arange(l_max + 1, dtype=jnp.float32) # Ensures float division
        denominator = 2.0 * l_values + 1.0

        # cl_val has shape [..., L], denominator has shape [L]
        cl_val = cl_val / denominator # Broadcasting should handle batch dims
        return cl_val

    def compute_cl(self, alm_dict_or_array, alm2_dict_or_array=None):
        """
        Computes power spectra (Cl) from alm coefficients.
        Args:
            alm_dict_or_array: A single alm array [..., l_max+1, l_max+1] or
                               a dictionary {spin: alm_array}.
            alm2_dict_or_array: Optional. Same format as alm_dict_or_array.
                                If None, auto-correlation is computed.
        Returns:
            A single Cl array [..., l_max+1] or a dictionary {(s1,s2): Cl_array}.
        """
        l_max = self.l_max # Use l_max from instance

        is_dict1 = isinstance(alm_dict_or_array, dict)
        is_dict2 = isinstance(alm2_dict_or_array, dict)

        if not is_dict1: # Input1 is a single array
            if is_dict2:
                raise TypeError("If alm1 is an array, alm2 must be an array or None.")
            # Both are arrays or alm2 is None
            return HealpixTransformer._compute_cl_from_single_alm_set(
                l_max, alm_dict_or_array, alm2_dict_or_array
            )
        else: # Input1 is a dictionary
            output_cls_dict = {}
            if alm2_dict_or_array is None: # Auto-correlations for dict1
                for s1, alm1 in alm_dict_or_array.items():
                    output_cls_dict[(s1, s1)] = HealpixTransformer._compute_cl_from_single_alm_set(
                        l_max, alm1, None
                    )
            elif not is_dict2: # alm1 is dict, alm2 is array -> invalid combination for clarity
                 raise TypeError("If alm1 is a dict, alm2 must be a dict or None.")
            else: # Both are dictionaries, compute all cross-correlations
                for s1, alm1 in alm_dict_or_array.items():
                    for s2, alm2 in alm2_dict_or_array.items():
                        # Optional: could sort (s1,s2) to avoid duplicate (s2,s1) if Cl_s1s2 = Cl_s2s1
                        output_cls_dict[(s1, s2)] = HealpixTransformer._compute_cl_from_single_alm_set(
                            l_max, alm1, alm2
                        )
            return output_cls_dict

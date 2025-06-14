import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial
from jax.scipy.special import gammaln as loggamma # Assuming utils.py's loggamma was this

# Utility functions (logdiffexp, logsumexp, log_A_lm, log_alpha_lm) are needed by YLM calculations.
# These were originally in YLM_jax.py, YLM_jax_log.py or utils.py.
# They should be co-located or imported appropriately. For now, define them here.

# From utils.py (or jax_healpix.utils if it exists and is preferred)
@jax.jit
def logdiffexp(log_a, log_b):  # assume a > b or handle appropriately
    """Compute log(a-b), using log_a and log_b."""
    # Ensure log_a is greater, or handle cases where a < b if necessary (e.g., return NaN or error)
    # This version assumes a >= b, which is typical for specific recurrence relations
    return log_a + jnp.log1p(-jnp.exp(log_b - log_a))

@jax.jit
def logsumexp_custom_signs(log_A, log_B, sign_A, sign_B):
    """
    Compute log(sign_A * exp(log_A) + sign_B * exp(log_B)), and the resulting sign.
    This is a more general logsumexp that handles signs.
    """
    # Ensure log_A, log_B, sign_A, sign_B are broadcastable
    log_A, log_B = jnp.broadcast_arrays(log_A, log_B)
    sign_A, sign_B = jnp.broadcast_arrays(sign_A, sign_B)

    # Determine which term is larger in magnitude
    max_abs_log = jnp.maximum(log_A, log_B)
    min_abs_log = jnp.minimum(log_A, log_B)

    # Signs of terms
    s_max = jnp.where(log_A >= log_B, sign_A, sign_B)
    s_min = jnp.where(log_A >= log_B, sign_B, sign_A)

    relative_sign = s_min * s_max

    log_sum = max_abs_log + jnp.log1p(relative_sign * jnp.exp(min_abs_log - max_abs_log))

    final_sign = jnp.where(jnp.isneginf(log_sum), 1, s_max)

    return log_sum, final_sign


@jax.jit
def _log_A_lm_coeff(l, m):
    """Eq. 14 of ref 1 (https://arxiv.org/pdf/1010.2084.pdf), in log space."""
    log_num = jnp.log(2*l-1) + jnp.log(2*l+1)
    valid_lm = jnp.logical_and(l > 0, l > m)

    log_den = jnp.log(l-m) + jnp.log(l+m)
    log_Alm_val = 0.5 * (log_num - log_den)

    return jnp.where(valid_lm, log_Alm_val, jnp.nan)


@jax.jit
def _alpha_lm_coeff(l, m):
    """Eq. A8 in ref 2 (https://arxiv.org/pdf/astro-ph/0502469.pdf). For spin-2."""
    valid_lm = jnp.logical_and(l > 0, l >= abs(m))
    is_zero_case = (l == m)

    term1_num = 2*l+1.0 # Ensure float for division
    term1_den = 2*l-1.0
    term2_num = l**2 - m**2

    val = jnp.zeros_like(l, dtype=jnp.float64)

    calc_cond = jnp.logical_and(valid_lm, term1_den > 0)
    safe_term1_den = jnp.where(term1_den <= 0, 1.0, term1_den)

    calculated_val = jnp.sqrt(term1_num * term2_num / safe_term1_den)

    return jnp.where(l==m, 0.0, jnp.where(calc_cond, calculated_val, jnp.nan))


@jax.jit
def _log_alpha_lm_coeff(l,m):
    """Log of alpha_lm. Handles l=m where alpha_lm=0 (log is -inf)."""
    is_zero_case = (l == m)
    log_val_cond = jnp.logical_and(l>0.0, l>m)

    log_val = 0.5 * (jnp.log(2*l+1.0) + jnp.log(l-m) + jnp.log(l+m) - jnp.log(2*l-1.0))

    return jnp.where(jnp.logical_and(is_zero_case, l>0), -jnp.inf, jnp.where(log_val_cond, log_val, jnp.nan))


class SphericalHarmonics:
    def __init__(self, l_max, spins_to_compute=(0,), use_log_recurrence=False):
        if not isinstance(l_max, int) or l_max < 0:
            raise ValueError("l_max must be a non-negative integer.")
        self.l_max = l_max
        self.spins_to_compute = tuple(sorted(list(set(spins_to_compute))))
        self.use_log_recurrence = use_log_recurrence

        self._ylm_cache = {}
        self._ylm_signs_cache = {}

    def get_ylm(self, spin, beta_values):
        if spin not in self.spins_to_compute:
            raise ValueError(f"Spin {spin} was not in spins_to_compute: {self.spins_to_compute}")
        beta_key = (spin, tuple(beta_values.tolist()))

        if beta_key in self._ylm_cache:
            if self.use_log_recurrence:
                return self._ylm_cache[beta_key], self._ylm_signs_cache[beta_key]
            else:
                return self._ylm_cache[beta_key]

        if self.use_log_recurrence:
            beta_abs = jnp.abs(beta_values)
            beta_sign = jnp.sign(beta_values)
            log_beta_input = jnp.log(jnp.where(beta_abs == 0, 1e-30, beta_abs))
            ylm_abs, ylm_signs = self._sYLM_recur_log_internal(self.l_max, self.spins_to_compute, log_beta_input, beta_sign)
            self._ylm_cache[beta_key] = ylm_abs[spin]
            self._ylm_signs_cache[beta_key] = ylm_signs[spin]
            return ylm_abs[spin], ylm_signs[spin]
        else:
            ylm_values_all_spins = self._sYLM_recur_internal(self.l_max, self.spins_to_compute, beta_values)
            self._ylm_cache[beta_key] = ylm_values_all_spins[spin]
            return ylm_values_all_spins[spin]

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def _sYLM_recur_internal(self, l_max, spins, beta):
        n_theta = beta.shape[-1]
        beta_clipped = jnp.clip(beta, -1.0, 1.0)
        beta_s = jnp.sqrt(1.0 - beta_clipped**2)
        # Initialize ylm dictionary with correct dtypes based on spin
        ylm = {}
        if 0 in spins:
            ylm[0] = jnp.zeros((l_max + 1, l_max + 1, n_theta), dtype=jnp.float64)
        if 2 in spins:
            ylm[2] = jnp.zeros((l_max + 1, l_max + 1, n_theta), dtype=jnp.complex128)
        if -2 in spins:
            ylm[-2] = jnp.zeros((l_max + 1, l_max + 1, n_theta), dtype=jnp.complex128)

        if 0 in spins:
            ylm[0] = ylm[0].at[0, 0, :].set(1.0)
            if l_max >= 1:
                ylm[0] = ylm[0].at[1,1,:].set(-jnp.sqrt(1.5) * beta_s)
                ylm[0] = ylm[0].at[1,0,:].set(jnp.sqrt(3.0) * beta)
            for l_val in range(2, l_max + 1):
                ylm[0] = ylm[0].at[l_val,l_val,:].set(
                    -jnp.sqrt((2*l_val+1.0)/(2*l_val)) * beta_s * ylm[0][l_val-1,l_val-1,:]
                )
                ylm[0] = ylm[0].at[l_val,l_val-1,:].set(
                     jnp.sqrt(2.0*l_val+1.0) * beta * ylm[0][l_val-1,l_val-1,:]
                )
                for m_val in range(l_val - 1):
                    if (l_val - 1) > m_val :
                        Alm_val = jnp.sqrt((4*l_val**2-1.0)/(l_val**2-m_val**2))
                        Alm_prev_val = jnp.sqrt((4*(l_val-1.0)**2-1.0)/((l_val-1.0)**2-m_val**2))
                        Blm_val = Alm_val / Alm_prev_val
                        term1 = beta * Alm_val * ylm[0][l_val-1, m_val, :]
                        term2 = -Blm_val * ylm[0][l_val-2, m_val, :]
                        ylm[0] = ylm[0].at[l_val, m_val, :].set(term1 + term2)
                    elif (l_val-1) == m_val:
                        Alm_val = jnp.sqrt((4*l_val**2-1.0)/(l_val**2-m_val**2))
                        ylm[0] = ylm[0].at[l_val, m_val, :].set(
                            beta * Alm_val * ylm[0][l_val-1, m_val, :]
                        )
            ylm[0] = ylm[0] / jnp.sqrt(4 * jnp.pi)

        if 2 in spins or -2 in spins:
            if 0 not in ylm:
                raise ValueError("Spin 0 Ylms are required to compute spin +/-2 Ylms.")

            for l_val in range(2, l_max + 1):
                for m_val in range(l_val + 1):
                    log_fact_l_minus_2 = loggamma(l_val - 2.0 + 1.0)
                    log_fact_l_plus_2 = loggamma(l_val + 2.0 + 1.0)
                    log_factorial_norm = 0.5 * (log_fact_l_minus_2 - log_fact_l_plus_2)
                    factorial_norm = jnp.exp(log_factorial_norm)

                    alm_coeff_val = _alpha_lm_coeff(l_val.astype(jnp.float64), m_val.astype(jnp.float64))

                    beta_s_sq = 1.0 - beta_clipped**2
                    safe_beta_s_sq = jnp.where(beta_s_sq == 0, 1e-30, beta_s_sq)
                    term1_coeff_numerator = 2.0 * (m_val**2 - l_val)

                    term1_s2 = (term1_coeff_numerator / safe_beta_s_sq - l_val * (l_val - 1.0)) * ylm[0][l_val, m_val, :]

                    ylm_prev_l_m_val = 0.0
                    if l_val-1 >= m_val:
                        ylm_prev_l_m_val = ylm[0][l_val-1, m_val, :]

                    term2_s2 = (2.0 * beta / safe_beta_s_sq) * alm_coeff_val * ylm_prev_l_m_val
                    term2_s2 = jnp.where(jnp.isnan(alm_coeff_val), 0.0, term2_s2)

                    if 2 in ylm:
                        ylm[2] = ylm[2].at[l_val,m_val,:].set(factorial_norm * (term1_s2 + term2_s2))

                    term1_s_neg2 = -(l_val - 1.0) * beta * ylm[0][l_val, m_val, :]
                    term2_s_neg2 = alm_coeff_val * ylm_prev_l_m_val
                    term2_s_neg2 = jnp.where(jnp.isnan(alm_coeff_val), 0.0, term2_s_neg2)

                    if -2 in ylm:
                        if m_val == 0:
                             ylm[-2] = ylm[-2].at[l_val,m_val,:].set(0.0)
                        else:
                            ylm[-2] = ylm[-2].at[l_val,m_val,:].set(
                                factorial_norm * (2.0 * m_val / safe_beta_s_sq) * (term1_s_neg2 + term2_s_neg2)
                            )
        return ylm

    @partial(jax.jit, static_argnums=(0,1,2))
    def _sYLM_recur_log_internal(self, l_max, spins, log_beta_abs, beta_sign):
        n_theta = log_beta_abs.shape[-1]
        cos_sq_theta = jnp.exp(2 * log_beta_abs)
        sin_sq_theta = 1.0 - cos_sq_theta
        log_sin_sq_theta = jnp.log(jnp.where(sin_sq_theta <=0, 1e-60, sin_sq_theta))
        log_sin_abs_theta = 0.5 * log_sin_sq_theta
        ylm_abs = {}
        ylm_signs = {}

        if 0 in spins:
            ylm_abs[0] = jnp.full((l_max + 1, l_max + 1, n_theta), -jnp.inf)
            ylm_signs[0] = jnp.ones((l_max + 1, l_max + 1, n_theta), dtype=jnp.int8)
            ylm_abs[0] = ylm_abs[0].at[0,0,:].set(0.0)
            ylm_signs[0] = ylm_signs[0].at[0,0,:].set(1)
            l_arr = jnp.arange(1, l_max + 1)
            log_prefact_yll = 0.5 * jnp.cumsum(jnp.log(2*l_arr+1.0) - jnp.log(2*l_arr))
            log_yll_abs = log_prefact_yll[:,None] + l_arr[:,None] * log_sin_abs_theta[None,:]
            sign_yll = ((-1)**l_arr)[:,None] * jnp.ones_like(log_yll_abs)
            ylm_abs[0] = ylm_abs[0].at[l_arr,l_arr,:].set(log_yll_abs)
            ylm_signs[0] = ylm_signs[0].at[l_arr,l_arr,:].set(sign_yll)
            if l_max >=1:
                log_y10_abs = ylm_abs[0][0,0,:] + log_beta_abs + 0.5*jnp.log(3.0)
                sign_y10 = ylm_signs[0][0,0,:] * beta_sign
                ylm_abs[0] = ylm_abs[0].at[1,0,:].set(log_y10_abs)
                ylm_signs[0] = ylm_signs[0].at[1,0,:].set(sign_y10)
                if l_max > 1:
                    for l_val_for_lm1_recur in range(2, l_max + 1):
                        log_ylm1_lm1_abs = ylm_abs[0][l_val_for_lm1_recur-1,l_val_for_lm1_recur-1,:]
                        sign_ylm1_lm1 = ylm_signs[0][l_val_for_lm1_recur-1,l_val_for_lm1_recur-1,:]

                        current_log_abs = log_ylm1_lm1_abs + log_beta_abs + 0.5*jnp.log(2.0*l_val_for_lm1_recur+1.0)
                        current_sign = sign_ylm1_lm1 * beta_sign
                        ylm_abs[0] = ylm_abs[0].at[l_val_for_lm1_recur,l_val_for_lm1_recur-1,:].set(current_log_abs)
                        ylm_signs[0] = ylm_signs[0].at[l_val_for_lm1_recur,l_val_for_lm1_recur-1,:].set(current_sign)
            for l_val in range(2, l_max + 1):
                for m_val in range(l_val - 1):
                    log_Alm_val = _log_A_lm_coeff(float(l_val), float(m_val))
                    log_T1_abs = log_Alm_val + log_beta_abs + ylm_abs[0][l_val-1,m_val,:]
                    sign_T1 = beta_sign * ylm_signs[0][l_val-1,m_val,:]
                    log_Alm_prev_val = _log_A_lm_coeff(float(l_val-1), float(m_val))
                    log_Blm_val = log_Alm_val - log_Alm_prev_val
                    log_T2_abs = log_Blm_val + ylm_abs[0][l_val-2,m_val,:]
                    sign_T2 = -1 * ylm_signs[0][l_val-2,m_val,:]
                    current_log_abs, current_sign = logsumexp_custom_signs(
                        log_T1_abs, log_T2_abs, sign_T1, sign_T2
                    )
                    ylm_abs[0] = ylm_abs[0].at[l_val,m_val,:].set(current_log_abs)
                    ylm_signs[0] = ylm_signs[0].at[l_val,m_val,:].set(current_sign)
            ylm_abs[0] = ylm_abs[0] - (0.5 * jnp.log(4 * jnp.pi))

        if 2 in spins or -2 in spins:
            if 0 not in ylm_abs:
                 raise ValueError("Spin 0 Ylms (log-space) are required for spin +/-2.")
            ylm_abs[2] = jnp.full((l_max + 1, l_max + 1, n_theta), -jnp.inf)
            ylm_signs[2] = jnp.ones((l_max + 1, l_max + 1, n_theta), dtype=jnp.int8)
            ylm_abs[-2] = jnp.full((l_max + 1, l_max + 1, n_theta), -jnp.inf)
            ylm_signs[-2] = jnp.ones((l_max + 1, l_max + 1, n_theta), dtype=jnp.int8)
            for l_val in range(2, l_max + 1):
                for m_val in range(l_val + 1):
                    pass
        return ylm_abs, ylm_signs

    @staticmethod
    @partial(jax.jit, static_argnames=("l_max",))
    def _l_stack_mask(l_max):
        m_coords = jnp.arange(l_max + 1)[None, :]
        l_coords = jnp.arange(l_max + 1)[:, None]
        return m_coords <= l_coords

    @staticmethod
    @partial(jax.jit, static_argnames=("l_max",))
    def _m_stack_mask(l_max):
        m_coords = jnp.arange(l_max + 1)[None, :]
        l_coords = jnp.arange(l_max + 1)[:, None]
        return m_coords <= l_coords


    @staticmethod
    @partial(jax.jit, static_argnames=("l_max", "order"))
    def stack_ylm_2d_to_1d(l_max, ylm_2d_single_array, order='l'):
        if order == 'l':
            l_indices_flat = jnp.array([l for l in range(l_max + 1) for _ in range(l + 1)])
            m_indices_flat = jnp.array([m for l in range(l_max + 1) for m in range(l + 1)])
            return ylm_2d_single_array[l_indices_flat, m_indices_flat]
        elif order == 'm':
            l_indices_flat = jnp.array([l for m in range(l_max + 1) for l in range(m, l_max + 1)])
            m_indices_flat = jnp.array([m for m in range(l_max + 1) for _ in range(m, l_max + 1)])
            return ylm_2d_single_array[l_indices_flat, m_indices_flat]
        else:
             return ylm_2d_single_array


    @staticmethod
    def process_ylm_stacking(l_max, ylm_2d, order='l'):
        if order not in ['l', 'm']:
            raise ValueError("Order must be 'l' or 'm'")

        if isinstance(ylm_2d, dict):
            ylm_s = {}
            for k, v_array in ylm_2d.items():
                if not isinstance(v_array, jnp.ndarray): # Add this check first
                    raise TypeError(f"Value for spin {k} in ylm_2d dict must be a JAX array, got {type(v_array)}")
                if v_array.shape[0] != l_max + 1 or v_array.shape[1] != l_max + 1:
                    raise ValueError(f"Input YLM array for spin {k} has incorrect first two dimensions: {v_array.shape}")
                ylm_s[k] = SphericalHarmonics.stack_ylm_2d_to_1d(l_max, v_array, order=order)
            return ylm_s
        elif isinstance(ylm_2d, jnp.ndarray):
            if ylm_2d.shape[0] != l_max + 1 or ylm_2d.shape[1] != l_max + 1:
                raise ValueError(f"Input YLM array has incorrect first two dimensions: {ylm_2d.shape}")
            return SphericalHarmonics.stack_ylm_2d_to_1d(l_max, ylm_2d, order=order)
        else:
            raise TypeError("ylm_2d must be a dictionary or a JAX array.")


if __name__ == '__main__':
    l_max_test = 2
    beta_test = jnp.array([0.0, 0.5, 1.0])

    ylm_calc_std = SphericalHarmonics(l_max_test, spins_to_compute=(0,), use_log_recurrence=False)
    s0_ylm_std_2d = ylm_calc_std.get_ylm(spin=0, beta_values=beta_test)
    print("Spin 0 Ylm (2D standard) shape:", s0_ylm_std_2d.shape)

    s0_ylm_std_1d_l_order = SphericalHarmonics.process_ylm_stacking(l_max_test, s0_ylm_std_2d, order='l')
    print("Spin 0 Ylm (1D L-order) shape:", s0_ylm_std_1d_l_order.shape)

    s0_ylm_std_1d_m_order = SphericalHarmonics.process_ylm_stacking(l_max_test, s0_ylm_std_2d, order='m')
    print("Spin 0 Ylm (1D M-order) shape:", s0_ylm_std_1d_m_order.shape)

    ylm_dict_2d = {0: s0_ylm_std_2d}
    ylm_dict_1d_l_order = SphericalHarmonics.process_ylm_stacking(l_max_test, ylm_dict_2d, order='l')
    print("Spin 0 Ylm from dict (1D L-order) shape:", ylm_dict_1d_l_order[0].shape)

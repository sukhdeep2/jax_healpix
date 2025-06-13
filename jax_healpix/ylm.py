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

    # Effective sign of the smaller term relative to the larger term
    # sign_factor = 1 if signs are same, -1 if different, when adding s_max*exp(max_abs_log) + s_min*exp(min_abs_log)
    # This is equivalent to s_max * (1 + (s_min/s_max) * exp(min_abs_log - max_abs_log))
    # (s_min/s_max) is s_min * s_max since signs are +/-1.
    relative_sign = s_min * s_max

    log_sum = max_abs_log + jnp.log1p(relative_sign * jnp.exp(min_abs_log - max_abs_log))

    # Determine the sign of the result
    # If log_A == log_B and sign_A == -sign_B, result is zero (log_sum = -inf). Sign can be convention (e.g., 1)
    # Otherwise, the sign of the result is the sign of the term with larger absolute value.
    # If they are equal in magnitude and signs are different, result is 0.
    # If they are equal in magnitude and signs are same, result is that sign.

    # A robust way to determine sign:
    # Calculate exp(log_A)*sign_A + exp(log_B)*sign_B and check its sign.
    # This is tricky without high precision.
    # The sign of the sum is s_max if (1 + relative_sign * exp(min_abs_log - max_abs_log)) is positive.
    # This term inside log1p is (1 + relative_sign * exp_diff).
    # If relative_sign is 1, it's (1 + exp_diff) > 0. Sign is s_max.
    # If relative_sign is -1, it's (1 - exp_diff).
    #   If exp_diff < 1, (1-exp_diff) > 0. Sign is s_max.
    #   If exp_diff == 1 (i.e. log_A==log_B and signs differ), sum is 0. log_sum is -inf. Sign is arbitrary (e.g. 1)
    #   If exp_diff > 1, (1-exp_diff) < 0. This case should not happen if max_abs_log is chosen correctly
    #      unless we are dealing with complex numbers or a mistake in logic.
    #      If log_A and log_B are real, exp_diff = exp(min_abs_log - max_abs_log) <= 1.
    #      So (1 - exp_diff) is >= 0.

    # The sign of the result is s_max, unless the sum is exactly zero.
    # If log_sum becomes -inf, the sum was zero.
    final_sign = jnp.where(jnp.isneginf(log_sum), 1, s_max) # Sign is 1 by convention if sum is 0

    return log_sum, final_sign


# From YLM_jax.py / YLM_jax_log.py
@jax.jit
def _log_A_lm_coeff(l, m): # Renamed from log_A_lm to avoid conflict if class method is named same
    """Eq. 14 of ref 1 (https://arxiv.org/pdf/1010.2084.pdf), in log space."""
    # log(sqrt((4*l^2 - 1) / (l^2 - m^2)))
    # = 0.5 * (log(4*l^2 - 1) - log(l^2 - m^2))
    # = 0.5 * (log((2l-1)(2l+1)) - log((l-m)(l+m)))
    # = 0.5 * (log(2l-1) + log(2l+1) - log(l-m) - log(l+m))
    # Need to handle m=l case (denominator is zero, A_lm is infinite)
    # The recurrence relations using A_lm typically apply for m < l.

    # Original uses logdiffexp:
    # log_Alm = -1 * logdiffexp(2 * jnp.log(l), 2 * jnp.log(m)) # -log(l^2-m^2)
    # log_Alm += logdiffexp(jnp.log(4) + 2 * jnp.log(l), 0)     # +log(4l^2-1)
    # log_Alm *= 0.5

    # Safer version:
    # Numerator: log(4*l^2 - 1) = log((2*l-1)*(2*l+1)) = log(2*l-1) + log(2*l+1)
    log_num = jnp.log(2*l-1) + jnp.log(2*l+1)
    # Denominator: log(l^2 - m^2) = log((l-m)*(l+m)) = log(l-m) + log(l+m)
    # Handle l=m separately where this term is undefined or recurrence differs.
    # The recurrence relation Y_lm propto A_lm Y_{l-1,m} is for m < l.
    # If m=l, log(l-m) is -inf.
    # We can use a mask for m < l.

    # Mask for invalid cases like l=0, or l=m for denominator
    valid_lm = jnp.logical_and(l > 0, l > m)

    log_den = jnp.log(l-m) + jnp.log(l+m)
    log_Alm_val = 0.5 * (log_num - log_den)

    return jnp.where(valid_lm, log_Alm_val, jnp.nan) # Or some other appropriate value for m>=l


@jax.jit
def _alpha_lm_coeff(l, m): # Renamed from alpha_lm
    """Eq. A8 in ref 2 (https://arxiv.org/pdf/astro-ph/0502469.pdf). For spin-2."""
    # sqrt((2*l+1)*(l^2-m^2)/(2*l-1))
    # = sqrt((2*l+1)/(2*l-1)) * sqrt(l^2-m^2)
    # Valid for l > 0. For l=0, terms like 2l-1 are problematic.
    # Also, l^2-m^2 implies |m| <= l. If |m|=l, l^2-m^2=0, so alpha_lm=0.
    # If l=0, this is not used. Typically, sYlm for s=2 starts at l=2.

    valid_lm = jnp.logical_and(l > 0, l >= abs(m)) # abs(m) because m can be negative in general Ylm context
                                              # but here m is an index 0 to l_max

    # term1 = (2*l+1)/(2*l-1) -> log(2*l+1) - log(2*l-1)
    # term2 = l^2-m^2 -> log(l-m) + log(l+m)

    # Condition: 2*l-1 > 0  => l >= 1
    # Condition: l-m > 0 and l+m > 0 (if using logs)
    # Or, l^2-m^2 >=0 => l >= m (since m is positive index here)

    # If l=m, (l^2-m^2) is 0, so alpha_lm is 0. Log would be -inf.
    is_zero_case = (l == m)

    # log_val = 0.5 * (jnp.log(2*l+1) + jnp.log(l-m) + jnp.log(l+m) - jnp.log(2*l-1)) # This is for log(alpha_lm)

    # If l=m, result is 0. If l < m, invalid. If 2l-1 is zero or neg, invalid.
    # alpha_lm is used for recurrence, e.g. s_Y_l,m from s_Y_{l-1},m
    # Valid range for l in recurrence: l >= 1 or l >= |s|

    # Value is 0 if l=m.
    # Value is NaN if l < m or l=0 (due to division by zero or sqrt of negative)
    # This function returns the sqrt value directly, not log.
    # Ensure terms are positive before sqrt
    term1_num = 2*l+1
    term1_den = 2*l-1
    term2_num = l**2 - m**2 # This is l*l - m*m

    # Handle cases to avoid NaN from sqrt of negative or division by zero.
    # Valid_lm ensures l > 0 and l >= m.
    # If l=m, term2_num is 0, so val is 0.
    # If term1_den is 0 (l=0.5), this is not for integer l.
    # If l=0, valid_lm is false.

    # Calculate val only for valid cases where term1_den > 0
    # and term2_num >=0 (guaranteed by l >= m)
    # and term1_num >=0 (guaranteed by l >= 0)
    val = jnp.zeros_like(l, dtype=jnp.float32) # Ensure float output

    # Condition for non-NaN result for val calculation part (excluding l==m)
    calc_cond = jnp.logical_and(valid_lm, term1_den > 0)

    # Avoid division by zero if term1_den is zero, though valid_lm (l>0) prevents l=0.5
    safe_term1_den = jnp.where(term1_den <= 0, 1, term1_den) # Should not happen for l>=1

    calculated_val = jnp.sqrt(term1_num * term2_num / safe_term1_den)

    return jnp.where(l==m, 0.0, jnp.where(calc_cond, calculated_val, jnp.nan))


@jax.jit
def _log_alpha_lm_coeff(l,m):
    """Log of alpha_lm. Handles l=m where alpha_lm=0 (log is -inf)."""
    valid_lm = jnp.logical_and(l > 0, l > m) # l > m for non-zero alpha_lm
    is_zero_case = (l == m) # and l > 0

    # Ensure terms are positive before log
    # Condition: 2*l-1 > 0 => l >= 1
    # Condition: l-m > 0 and l+m > 0 => l > m
    log_val_cond = jnp.logical_and(l>0, l>m) # Stricter than valid_lm for log

    log_val = 0.5 * (jnp.log(2*l+1) + jnp.log(l-m) + jnp.log(l+m) - jnp.log(2*l-1))

    # If l=m and l>0, alpha_lm is 0, so log(alpha_lm) is -inf.
    # If l < m or l=0 or (2l-1 <=0), it's NaN.
    return jnp.where(jnp.logical_and(is_zero_case, l>0), -jnp.inf, jnp.where(log_val_cond, log_val, jnp.nan))


class SphericalHarmonics:
    def __init__(self, l_max, spins_to_compute=(0,), use_log_recurrence=False):
        if not isinstance(l_max, int) or l_max < 0:
            raise ValueError("l_max must be a non-negative integer.")
        self.l_max = l_max
        self.spins_to_compute = tuple(sorted(list(set(spins_to_compute)))) # Ensure unique, sorted
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
        ylm = {s: jnp.zeros((l_max + 1, l_max + 1, n_theta)) for s in spins if s == 0}
        if 0 not in ylm and 0 in spins:
             ylm[0] = jnp.zeros((l_max + 1, l_max + 1, n_theta))

        if 0 in spins:
            ylm[0] = ylm[0].at[0, 0, :].set(1.0)
            if l_max >= 1:
                ylm[0] = ylm[0].at[1,1,:].set(-jnp.sqrt(1.5) * beta_s)
                ylm[0] = ylm[0].at[1,0,:].set(jnp.sqrt(3.0) * beta)
            for l_val in range(2, l_max + 1):
                ylm[0] = ylm[0].at[l_val,l_val,:].set(
                    -jnp.sqrt((2*l_val+1)/(2*l_val)) * beta_s * ylm[0][l_val-1,l_val-1,:]
                )
                ylm[0] = ylm[0].at[l_val,l_val-1,:].set(
                     jnp.sqrt(2.0*l_val+1.0) * beta * ylm[0][l_val-1,l_val-1,:]
                )
                for m_val in range(l_val - 1):
                    if (l_val - 1) > m_val :
                        Alm_val = jnp.sqrt((4*l_val**2-1)/(l_val**2-m_val**2))
                        Alm_prev_val = jnp.sqrt((4*(l_val-1)**2-1)/((l_val-1)**2-m_val**2))
                        Blm_val = Alm_val / Alm_prev_val
                        term1 = beta * Alm_val * ylm[0][l_val-1, m_val, :]
                        term2 = -Blm_val * ylm[0][l_val-2, m_val, :]
                        ylm[0] = ylm[0].at[l_val, m_val, :].set(term1 + term2)
                    elif (l_val-1) == m_val:
                        Alm_val = jnp.sqrt((4*l_val**2-1)/(l_val**2-m_val**2))
                        ylm[0] = ylm[0].at[l_val, m_val, :].set(
                            beta * Alm_val * ylm[0][l_val-1, m_val, :]
                        )
            ylm[0] = ylm[0] / jnp.sqrt(4 * jnp.pi)

        if 2 in spins or -2 in spins:
            if 0 not in ylm: # Need spin 0 Ylms to compute spin +/- 2
                # This case implies spins=(2,) or (-2,) but not (0,2,-2). Compute s=0 first.
                # This recursive call might be problematic for JIT if spins tuple changes.
                # For simplicity, assume if 2 or -2 is requested, 0 is implicitly computed or available.
                # A robust way: ensure s=0 is always computed if s=2 or s=-2 are needed.
                # This is handled by spins_to_compute in __init__ if used by internal calls.
                # Here, ylm[0] must exist. If it wasn't created (0 not in spins), this is an issue.
                # Let's assume ylm[0] is available if this part is reached.
                # A practical solution: if 2 or -2 in spins, ensure 0 is added to computation list.
                # The current structure of _sYLM_recur_internal computes requested spins.
                # If s=0 is not in `spins` argument, ylm[0] won't be populated.
                raise ValueError("Spin 0 Ylms are required to compute spin +/-2 Ylms.")


            ylm[2] = jnp.zeros((l_max + 1, l_max + 1, n_theta))
            ylm[-2] = jnp.zeros((l_max + 1, l_max + 1, n_theta))
            for l_val in range(2, l_max + 1):
                for m_val in range(l_val + 1):
                    if l_val < 2: continue
                    log_fact_l_minus_2 = loggamma(l_val - 2 + 1.0)
                    log_fact_l_plus_2 = loggamma(l_val + 2 + 1.0)
                    log_factorial_norm = 0.5 * (log_fact_l_minus_2 - log_fact_l_plus_2)
                    factorial_norm = jnp.exp(log_factorial_norm)
                    alm_coeff = _alpha_lm_coeff(l_val, m_val)
                    beta_s_sq = 1.0 - beta_clipped**2
                    safe_beta_s_sq = jnp.where(beta_s_sq == 0, 1e-30, beta_s_sq)
                    term1_coeff_numerator = 2 * (m_val**2 - l_val) # Note: m_val needs to be float for potential non-int results

                    term1_s2 = (term1_coeff_numerator / safe_beta_s_sq - l_val * (l_val - 1)) * ylm[0][l_val, m_val, :]
                    # alm_coeff can be NaN if l_val-1 < m_val. ylm[0] access also needs care.
                    # For term2_s2, ylm[0][l_val-1, m_val,:] is used. This is valid if l_val-1 >= m_val.
                    # If l_val-1 < m_val, alm_coeff is NaN, term2_s2 becomes NaN.
                    # We should ensure that alm_coeff is zero or terms handled if indices are out of bounds.
                    # _alpha_lm_coeff returns NaN if l < m.
                    # If l_val-1 < m_val, alm_coeff is NaN.
                    # If l_val-1 = m_val, alm_coeff is 0 for that (l-1,m) pair.
                    # If l_val-1 > m_val, alm_coeff is valid.

                    # Mask for ylm[0][l_val-1, m_val, :] access and alm_coeff validity
                    # alm_coeff is for (l_val, m_val), not (l_val-1, m_val)
                    # The formula uses alpha_lm(l,m) and Y_{l-1,m}.
                    # Y_{l-1,m} is zero if l-1 < m.
                    ylm_prev_l_m_val = ylm[0][l_val-1, m_val, :] if l_val-1 >= m_val else 0.0

                    term2_s2 = (2 * beta / safe_beta_s_sq) * alm_coeff * ylm_prev_l_m_val
                    # Handle NaNs from alm_coeff if they occur (e.g. if l_val < m_val, though loop prevents this)
                    term2_s2 = jnp.where(jnp.isnan(alm_coeff), 0.0, term2_s2)


                    ylm[2] = ylm[2].at[l_val,m_val,:].set(factorial_norm * (term1_s2 + term2_s2))

                    term1_s_neg2 = -(l_val - 1) * beta * ylm[0][l_val, m_val, :]
                    term2_s_neg2 = alm_coeff * ylm_prev_l_m_val
                    term2_s_neg2 = jnp.where(jnp.isnan(alm_coeff), 0.0, term2_s_neg2)

                    # Factor (2*m/beta_s^2) can be problematic if m=0 or beta_s=0.
                    # If m=0, product is 0.
                    # If beta_s=0 (poles):
                    #   If m!=0, this is inf. Y_s,lm is zero at poles if m!=s.
                    #   If m=0, this is 0/0 (NaN).
                    # The YLM_jax.py version had a specific m=0 handling for spin -2.
                    if m_val == 0:
                         ylm[-2] = ylm[-2].at[l_val,m_val,:].set(0.0)
                    else:
                        ylm[-2] = ylm[-2].at[l_val,m_val,:].set(
                            factorial_norm * (2 * m_val / safe_beta_s_sq) * (term1_s_neg2 + term2_s_neg2)
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
            log_prefact_yll = 0.5 * jnp.cumsum(jnp.log(2*l_arr+1) - jnp.log(2*l_arr))
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
                    for l_val_iter in range(len(l_arr)):
                        l_val = l_arr[l_val_iter]
                        if l_val == 0: continue
                        if l_val == 1: continue
                        log_ylm1_lm1_abs = ylm_abs[0][l_val-1,l_val-1,:]
                        sign_ylm1_lm1 = ylm_signs[0][l_val-1,l_val-1,:]
                        current_log_abs = log_ylm1_lm1_abs + log_beta_abs + 0.5*jnp.log(2.0*l_val+1.0)
                        current_sign = sign_ylm1_lm1 * beta_sign
                        ylm_abs[0] = ylm_abs[0].at[l_val,l_val-1,:].set(current_log_abs)
                        ylm_signs[0] = ylm_signs[0].at[l_val,l_val-1,:].set(current_sign)
            for l_val in range(2, l_max + 1):
                for m_val in range(l_val - 1):
                    log_Alm_val = _log_A_lm_coeff(l_val, m_val)
                    log_T1_abs = log_Alm_val + log_beta_abs + ylm_abs[0][l_val-1,m_val,:]
                    sign_T1 = beta_sign * ylm_signs[0][l_val-1,m_val,:]
                    log_Alm_prev_val = _log_A_lm_coeff(l_val-1, m_val)
                    if jnp.isnan(log_Alm_prev_val).any():
                        pass
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
            if 0 not in ylm_abs: # Ensure spin 0 is computed
                 raise ValueError("Spin 0 Ylms (log-space) are required for spin +/-2.")
            ylm_abs[2] = jnp.full((l_max + 1, l_max + 1, n_theta), -jnp.inf)
            ylm_signs[2] = jnp.ones((l_max + 1, l_max + 1, n_theta), dtype=jnp.int8)
            ylm_abs[-2] = jnp.full((l_max + 1, l_max + 1, n_theta), -jnp.inf)
            ylm_signs[-2] = jnp.ones((l_max + 1, l_max + 1, n_theta), dtype=jnp.int8)
            for l_val in range(2, l_max + 1):
                for m_val in range(l_val + 1):
                    pass
        return ylm_abs, ylm_signs

    # --- Static helper methods for YLM reshaping ---
    @staticmethod
    @partial(jax.jit, static_argnames=("l_max",))
    def _l_stack_mask(l_max):
        """ Returns a boolean mask for (l,m) valid region (m<=l). """
        m_mat = jnp.tile(jnp.arange(l_max + 1), l_max + 1).reshape(l_max + 1, l_max + 1)
        l_mat = m_mat.T
        return m_mat <= l_mat

    @staticmethod
    @partial(jax.jit, static_argnames=("l_max",))
    def _m_stack_mask(l_max):
        """ Returns a boolean mask for (l,m) valid region (m<=l). """
        # Same as _l_stack_mask, as it's just a lower/upper triangle mask.
        # The naming might imply different ordering if used for selection,
        # but as a plain mask m<=l, it's identical.
        l_mat = jnp.tile(jnp.arange(l_max + 1), l_max + 1).reshape(l_max + 1, l_max + 1)
        m_mat = l_mat.T # Corrected: if l_mat is m repeated, m_mat is l repeated.
                        # To get l_mat as rows of L: jnp.arange(l_max+1)[:, None]
                        # To get m_mat as cols of M: jnp.arange(l_max+1)[None, :]
        # Correct implementation for m <= l mask:
        m_coords = jnp.arange(l_max + 1)[None, :]
        l_coords = jnp.arange(l_max + 1)[:, None]
        return m_coords <= l_coords


    @staticmethod
    @partial(jax.jit, static_argnames=("l_max", "order")) # JIT stack_ylm_2d_to_1d
    def stack_ylm_2d_to_1d(l_max, ylm_2d_single_array, order='l'):
        """
        Stacks a single 2D YLM array into 1D.
        Helper for the main stack_ylm_2d_to_1d that handles dicts.
        ylm_2d_single_array shape: (l_max+1, l_max+1, ...other_dims...)
        """
        if ylm_2d_single_array.shape[0] != l_max + 1 or ylm_2d_single_array.shape[1] != l_max + 1:
            # This check cannot be JITted if shapes are dynamic.
            # However, l_max is static, so shapes should be static.
            # For JIT, error raising is tricky. Usually, JAX replaces with NaN or fixed values.
            # Caller should ensure valid shapes.
            pass # Or handle error appropriately if not JITting this part of check

        if order == 'l':
            # L-primary order: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), ...
            # This means we iterate l, then m.
            # A direct way to get this order is to select using flat indices for l and m.
            l_indices_flat = jnp.array([l for l in range(l_max + 1) for _ in range(l + 1)])
            m_indices_flat = jnp.array([m for l in range(l_max + 1) for m in range(l + 1)])
            return ylm_2d_single_array[l_indices_flat, m_indices_flat]
        elif order == 'm':
            # M-primary order (Healpy): (0,0), (1,0), ..., (L,0), (1,1), ..., (L,1), ...
            # Iterate m, then l.
            l_indices_flat = jnp.array([l for m in range(l_max + 1) for l in range(m, l_max + 1)])
            m_indices_flat = jnp.array([m for m in range(l_max + 1) for _ in range(m, l_max + 1)])
            return ylm_2d_single_array[l_indices_flat, m_indices_flat]
        else:
            # This will fail JIT if error is raised.
            # For static method, can't raise ValueError in JITted func easily.
            # Assume valid order or handle outside.
            return ylm_2d_single_array # Placeholder for error or ensure valid input

    @staticmethod
    def process_ylm_stacking(l_max, ylm_2d, order='l'): # Renamed from stack_ylm_2d_to_1d to avoid JIT issues with dicts
        """
        Stacks 2D YLMs (l,m) into a 1D array.
        ylm_2d: Can be a single JAX array (l_max+1, l_max+1, ...) or a dict {spin: ylm_array}.
        order: 'l' for l-primary, 'm' for m-primary (healpy).
        Output: Single array (n_coeffs, ...) or dict {spin: array_1d}.
        """
        if order not in ['l', 'm']:
            raise ValueError("Order must be 'l' or 'm'")

        # The JITted helper SphericalHarmonics.stack_ylm_2d_to_1d works on single arrays.
        # This outer function handles the dictionary logic.

        if isinstance(ylm_2d, dict):
            ylm_s = {}
            for k, v_array in ylm_2d.items():
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


# Example usage (for testing this file independently):
if __name__ == '__main__':
    l_max_test = 2 # Keep small for printing
    beta_test = jnp.array([0.0, 0.5, 1.0])

    ylm_calc_std = SphericalHarmonics(l_max_test, spins_to_compute=(0,), use_log_recurrence=False)
    s0_ylm_std_2d = ylm_calc_std.get_ylm(spin=0, beta_values=beta_test)
    print("Spin 0 Ylm (2D standard) shape:", s0_ylm_std_2d.shape)
    # Expected: (l_max+1, l_max+1, n_beta) = (3,3,3)

    # Test stacking
    s0_ylm_std_1d_l_order = SphericalHarmonics.process_ylm_stacking(l_max_test, s0_ylm_std_2d, order='l')
    # Num coeffs for l_max=2 is (2+1)*(2+2)/2 = 3*4/2 = 6
    print("Spin 0 Ylm (1D L-order) shape:", s0_ylm_std_1d_l_order.shape)
    # Expected: (num_coeffs, n_beta) = (6,3)

    s0_ylm_std_1d_m_order = SphericalHarmonics.process_ylm_stacking(l_max_test, s0_ylm_std_2d, order='m')
    print("Spin 0 Ylm (1D M-order) shape:", s0_ylm_std_1d_m_order.shape)
    # Expected: (num_coeffs, n_beta) = (6,3)

    # Test with dict
    ylm_dict_2d = {0: s0_ylm_std_2d}
    ylm_dict_1d_l_order = SphericalHarmonics.process_ylm_stacking(l_max_test, ylm_dict_2d, order='l')
    print("Spin 0 Ylm from dict (1D L-order) shape:", ylm_dict_1d_l_order[0].shape)


```

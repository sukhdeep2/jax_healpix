"""
Debug: Compare YLM values between CUDA and JAX
"""

import numpy as np
import sys
sys.path.insert(0, 'jax_healpix')

# Patch skylens
class FakeModule:
    pass
sys.modules['skylens'] = FakeModule()
sys.modules['skylens.wigner_transform'] = FakeModule()

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from YLM_jax_log import sYLM_recur_log, logsumexp

def ring_log_beta(nside):
    beta = jnp.zeros(4 * nside - 1)
    ring_i = jnp.arange(4 * nside - 1) + 1
    l1 = nside
    pole_beta = jnp.log(ring_i[:l1]) * 2 - jnp.log(nside) * 2 - jnp.log(3)
    pole_beta, _ = logsumexp(
        pole_beta,
        jnp.zeros_like(pole_beta),
        -1 * jnp.ones_like(pole_beta),
        jnp.ones_like(pole_beta),
    )
    beta = beta.at[:l1].set(pole_beta)
    beta = beta.at[-l1:].set(pole_beta[::-1])
    eq_beta = jnp.absolute(4.0 / 3 - 2.0 / 3 * ring_i[l1:-l1] / nside)
    eq_beta = jnp.log(eq_beta)
    beta = beta.at[l1:-l1].set(eq_beta)
    beta_sign = jnp.ones(4 * nside - 1, dtype=jnp.int8)
    beta_sign = beta_sign.at[2 * nside :].set(-1)
    return beta, beta_sign

nside = 32
l_max = 10  # Small for debugging
n_rings_test = 5  # Test first 5 rings

print(f"=== YLM Comparison Debug ===")
print(f"nside={nside}, l_max={l_max}, testing {n_rings_test} rings")

# Get JAX log_beta for north rings
log_beta_jax, beta_sign_jax = ring_log_beta(nside)
log_beta_north = np.array(log_beta_jax[:2*nside])  # North hemisphere only

print(f"\nlog_beta for first {n_rings_test} rings:")
for i in range(n_rings_test):
    print(f"  Ring {i+1}: log_beta={log_beta_north[i]:.10e}, cos(theta)={np.exp(log_beta_north[i]):.10e}")

# Compute JAX YLM for north rings
ylm_jax = sYLM_recur_log(l_max=l_max, spins=(0,), log_beta=log_beta_north[:n_rings_test])
ylm_jax_0 = np.array(ylm_jax[0])  # [l_max+1, l_max+1, n_rings]

print(f"\nJAX YLM shape: {ylm_jax_0.shape}")

print(f"\nJAX Y_lm values for first ring (ring 1, near north pole):")
print(f"  Y_00 = {ylm_jax_0[0, 0, 0]:.10e}")
print(f"  Y_10 = {ylm_jax_0[1, 0, 0]:.10e}")
print(f"  Y_11 = {ylm_jax_0[1, 1, 0]:.10e}")
print(f"  Y_20 = {ylm_jax_0[2, 0, 0]:.10e}")
print(f"  Y_21 = {ylm_jax_0[2, 1, 0]:.10e}")
print(f"  Y_22 = {ylm_jax_0[2, 2, 0]:.10e}")

print(f"\nJAX Y_l0 values summed over first {n_rings_test} rings:")
for l in range(l_max + 1):
    sum_ylm = np.sum(ylm_jax_0[l, 0, :])
    print(f"  sum(Y_{l}0) = {sum_ylm:.10e}")

# Save reference values for comparison
np.save('/tmp/jax_log_beta.npy', log_beta_north[:n_rings_test])
np.save('/tmp/jax_ylm.npy', ylm_jax_0)

print(f"\nSaved reference to /tmp/jax_log_beta.npy and /tmp/jax_ylm.npy")

# Now let's check if the expected Y_00 value is correct
expected_Y00 = 1.0 / np.sqrt(4 * np.pi)
print(f"\nExpected Y_00 = 1/sqrt(4*pi) = {expected_Y00:.10e}")
print(f"JAX Y_00 = {ylm_jax_0[0, 0, 0]:.10e}")
print(f"Difference = {abs(ylm_jax_0[0, 0, 0] - expected_Y00):.2e}")

# Check Y_10 at cos(theta)
# Y_10 = sqrt(3/4pi) * cos(theta)
cos_theta = np.exp(log_beta_north[0])  # First ring
expected_Y10 = np.sqrt(3.0 / (4 * np.pi)) * cos_theta
print(f"\nFor ring 1 (cos(theta) = {cos_theta:.10e}):")
print(f"Expected Y_10 = sqrt(3/4pi)*cos(theta) = {expected_Y10:.10e}")
print(f"JAX Y_10 = {ylm_jax_0[1, 0, 0]:.10e}")
print(f"Difference = {abs(ylm_jax_0[1, 0, 0] - expected_Y10):.2e}")

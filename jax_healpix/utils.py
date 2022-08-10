import numpy as np
import jax
import jax.numpy as jnp
from jax import jit


def logdiffexp(log_a, log_b):  # assume a>b
    """
    compute log(a-b), using log_a and log_b
    """
    log_diff = jnp.log(1 - jnp.exp(log_b - log_a))
    log_diff += log_a
    return log_diff


def logsumexp(log_A, log_B, sign_A, sign_B):
    """
    compute log(a +/- b), using log_a and log_b and the signs of a and b
    """
    log_A, log_B = np.broadcast_arrays(log_A, log_B)
    sign_A, sign_B = np.broadcast_arrays(sign_A, sign_B)
    sign_s = sign_A * sign_B

    max_Arr = np.maximum(log_A, log_B)
    min_Arr = np.minimum(log_A, log_B)

    log_s = max_Arr + np.log(1 + sign_s * np.exp(min_Arr - max_Arr))

    sign_s[sign_A == sign_B] = sign_A[sign_A == sign_B]

    x = np.logical_and(sign_A != sign_B, log_A >= log_B)
    sign_s[x] = sign_A[x]
    x = np.logical_and(sign_A != sign_B, log_A < log_B)
    sign_s[x] = sign_B[x]
    return log_s, sign_s

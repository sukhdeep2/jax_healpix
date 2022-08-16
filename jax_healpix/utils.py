import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import pickle, sys
from jax.scipy.special import gammaln as loggamma


# @jit
# def loggamma(n):
#     return jnp.sum(jnp.log(jnp.arange(n) + 1))


def logdiffexp(log_a, log_b):  # assume a>b
    """
    compute log(a-b), using log_a and log_b
    """
    log_diff = jnp.log(1 - jnp.exp(log_b - log_a))
    log_diff += log_a
    return log_diff


# https://stackoverflow.com/questions/2350072/custom-data-types-in-numpy-arrays
@jit
def logsumexp(log_A, log_B, sign_A, sign_B):
    """
    compute log(a +/- b), using log_a and log_b and the signs of a and b
    """
    log_A, log_B = jnp.broadcast_arrays(log_A, log_B)
    sign_A, sign_B = jnp.broadcast_arrays(sign_A, sign_B)
    sign_s = sign_A * sign_B

    max_Arr = jnp.maximum(log_A, log_B)
    min_Arr = jnp.minimum(log_A, log_B)

    # log_s = max_Arr + jnp.log(1 + sign_s * jnp.exp(min_Arr - max_Arr))
    log_s = max_Arr + jnp.log1p(sign_s * jnp.exp(min_Arr - max_Arr))
    # FIXME: this can be made better for very small numbers.
    # FIXME: see https://en.wikipedia.org/wiki/Natural_logarithm#lnp1 and np.log1p

    # sign_s[sign_A == sign_B] = sign_A[sign_A == sign_B]
    # sign_s = jnp.where(sign_A == sign_B, sign_A, sign_s) #combined with x below

    x = jnp.logical_and(sign_A != sign_B, log_A >= log_B)
    x = jnp.logical_or(x, sign_A == sign_B)
    # sign_s[x] = sign_A[x]
    sign_s = jnp.where(x, sign_A, sign_s)

    x = jnp.logical_and(sign_A != sign_B, log_A < log_B)
    # sign_s[x] = sign_B[x]
    sign_s = jnp.where(x, sign_B, sign_s)
    return log_s, sign_s


def get_size_pickle(obj):
    """
    Get the size of an object via pickle.
    """
    yy = pickle.dumps(obj)
    return np.around(sys.getsizeof(yy) / 1.0e9, decimals=3)


def get_size(obj):
    return obj.size * obj.itemsize / 1.0e9

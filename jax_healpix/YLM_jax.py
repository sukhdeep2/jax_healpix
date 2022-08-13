"""
Computes spin weighted spherical harmonics, using recurrence 
relations from reference 1 (spin-0) and 2 (spin-2)

1. https://arxiv.org/pdf/1010.2084.pdf

2. https://arxiv.org/pdf/astro-ph/0502469.pdf

Other useful references.
https://arxiv.org/pdf/1303.4945.pdf

https://arxiv.org/pdf/1804.10382.pdf

https://iopscience.iop.org/article/10.1088/0067-0049/190/2/267/pdf

https://arxiv.org/pdf/0904.2517.pdf (alternate pixelization)
"""

# from functools import partial
from jax.tree_util import Partial as partial


import jax
import jax.numpy as jnp
from jax import jit
from utils import *


@partial(jax.jit, static_argnums=(0))
def sYLM_ll0(l_max, beta, beta_s, ylm):
    """
    Computes spin-0 Ylm for l=m. Eq. 15 of ref 1.
    There are two implementations, first one uses the l-1 Ylm which is already computed.
    The second implementation computes Ylm directly by writing recursion as a combination of
    products and then computing the product in logspace.
    """
    # l_arr = jnp.arange(l_max) + 1
    # log_2l = jnp.log(l) + jnp.log(2)
    # log_pre_fact = jnp.logaddexp(log_2l, 0) - log_2l  # jnp.sqrt((2*l+1)/2/l)
    # pre_fact = -1 * jnp.exp(0.5 * log_pre_fact)
    # ylm[0] = ylm[0].at[l, l, :].add(pre_fact * ylm[0][l - 1, l - 1, :] * beta_s)

    l_arr = jnp.arange(l_max) + 1
    # log_prefact = jnp.sum(jnp.where(l_arr <= l, jnp.log(2 * l_arr + 1), 0))
    # log_prefact -= jnp.sum(jnp.where(l_arr <= l, jnp.log(2 * l_arr), 0))
    log_prefact = jnp.cumsum(jnp.log(2 * l_arr + 1), 0)
    log_prefact -= jnp.cumsum(jnp.log(2 * l_arr), 0)

    log_prefact *= 0.5
    log_yll = (
        jnp.log(beta_s[None, :]) * l_arr[:, None] + log_prefact[:, None]
    )  # FIXME: pass log_beta_s
    yll = ((-1) ** l_arr[:, None]) * jnp.exp(log_yll)
    ylm[0] = ylm[0].at[l_arr, l_arr, :].set(yll)

    ylm[0] = (
        ylm[0]
        .at[l_arr, l_arr - 1, :]
        .set(
            ylm[0][l_arr - 1, l_arr - 1, :]
            * beta[None, :]
            * jnp.sqrt(2 * (l_arr[:, None] - 1) + 3)
        )
    )
    return ylm  # ,None


@jit
def A_lm(l, m):
    """
    Eq. 14 of ref 1
    """
    return jnp.sqrt((4 * l**2 - 1) / (l**2 - m**2))


@jit
def log_A_lm(l, m):
    """
    Eq. 14 of ref 1, written using logs. logdiffexp is imported from utils.py
    """
    log_Alm = -1 * logdiffexp(2 * jnp.log(l), 2 * jnp.log(m))
    log_Alm += logdiffexp(jnp.log(4) + 2 * jnp.log(l), 0)
    log_Alm *= 0.5
    return log_Alm


@jit
def alpha_lm(l, m):
    """
    Eq. A8 in ref 2. Recurrence for spin-2 quantities.
    """
    return jnp.sqrt((2 * l + 1) * (l**2 - m**2) / (2 * l - 1))


# @partial(jax.jit, static_argnums=(0))
def sYLM_l0(l_max, beta, beta_s, l, ylm):
    """
    Computes Ylm using the eq 13 of ref 1. Y_(l-1)m and Y_(l-2)m should be computed already.
    """
    m = jnp.arange(l_max + 1)

    log_Alm = log_A_lm(l, m)
    log_Alm_prev = log_A_lm(l - 1, m)
    Blm = jnp.where(l - 1 > m, jnp.exp(log_Alm - log_Alm_prev), 0)
    Alm = jnp.where(l - 1 > m, jnp.exp(log_Alm), 0)

    # Alm = A_lm(l, m)
    # Alm_prev = A_lm(l - 1, m)

    # Blm = jnp.where(l - 1 > m, Alm / Alm_prev, 0)
    # Alm = jnp.where(l - 1 > m, Alm, 0)

    ylm[0] = ylm[0].at[l, :, :].add(beta[None, :] * Alm[:, None] * ylm[0][l - 1, :, :])
    ylm[0] = ylm[0].at[l, :, :].add(-Blm[:, None] * ylm[0][l - 2, :, :])
    # ylm = sYLM_ll0(l_max, beta_s, l, ylm)
    return ylm


def sYLM_l2(beta, beta_s2, l_max, l, ylm):  # m<l, spin 2
    """
    Compute spin-2 (+/-)  Ylm using spin-0 computation.
    Eq. A7 in ref 2.
    """
    m = jnp.arange(l_max + 1)

    alm = alpha_lm(l, m)
    ylm[2] = (
        ylm[2]
        .at[l, :, :]
        .add(
            (2 * (m[:, None] ** 2 - l) / beta_s2[None, :] - l * (l - 1))
            * ylm[0][l, :, :]
        )
    )  # FIXME: beta_s==0 case
    ylm[2] = (
        ylm[2]
        .at[l, :, :]
        .add(2 * (beta / beta_s2)[None, :] * alm[:, None] * ylm[0][l - 1, :, :])
    )

    ylm[-2] = (
        ylm[-2].at[l, :, :].add(-(l - 1) * beta[None, :] * ylm[0][l, :, :])
    )  # FIXME: beta_s==0 case
    ylm[-2] = ylm[-2].at[l, :, :].add(alm[:, None] * ylm[0][l - 1, :, :])

    ylm[-2] = ylm[-2].at[l, :, :].multiply(2 * m[:, None] / beta_s2[None, :])

    return ylm


@partial(jax.jit, static_argnums=(0, 1))
def sYLM_recur(l_max, spin_max, beta):
    """
    Computes all Ylm using recusion. There is only one loop, over l.
    spin_max: 0 or 2, depending on Ylm desired. Spin_max=2 will return both spin-0 and spin-2 Ylm.
    """
    n_theta = len(beta)
    n_lm = (l_max + 1) ** 2
    # beta_s=-jnp.sqrt(1-beta**2)
    beta_s = jnp.sin(jnp.arccos(beta))

    ylm = {}
    #     ylm[0]=jnp.zeros((n_lm,n_theta))
    ylm[0] = jnp.zeros((l_max + 1, l_max + 1, n_theta))
    ylm[0] = ylm[0].at[0, 0, :].set(1)

    sYLM_ll0_i = jax.tree_util.Partial(
        sYLM_ll0,
        l_max,
        beta,
        beta_s,
    )

    # ylm = jax.lax.fori_loop(1, l_max + 1, sYLM_ll0_i, ylm)
    ylm = sYLM_ll0_i(ylm)

    sYLM_l0_i = jax.tree_util.Partial(  # partial(
        sYLM_l0,
        l_max,
        beta,
        beta_s,
    )

    # for l in range(1, l_max + 1):
    #     ylm = sYLM_l0(l_max, beta, beta_s, l, ylm)

    ylm = jax.lax.fori_loop(2, l_max + 1, sYLM_l0_i, ylm)

    ylm[0] /= jnp.sqrt(4 * jnp.pi)
    # ylm[0] = ylm[0].at[:, 0, :].divide(2)  # FIXME: ????????????
    # ylm[0] = ylm[0].at[:, 1:, :].multiply(2)  # FIXME: ????????????

    if spin_max == 2:
        sYLM_l2_i = partial(sYLM_l2, beta, beta_s**2, l_max)
        ylm[2] = jnp.zeros((l_max + 1, l_max + 1, n_theta))
        ylm[-2] = jnp.zeros((l_max + 1, l_max + 1, n_theta))
        #         for l in range(1,l_max+1):
        #             ylm=sYLM_l2_i(l,ylm)
        ylm = jax.lax.fori_loop(2, l_max + 1, sYLM_l2_i, ylm)

    return ylm


##################################

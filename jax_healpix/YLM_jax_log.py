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

from functools import partial
from utils import *


def sYLM_ll0_log(l_max, log_beta, log_beta_s, ylm):
    """
    Computes spin-0 Ylm for l=m. Eq. 15 of ref 1.
    There are two implementations, first one uses the l-1 Ylm which is already computed.
    The second implementation computes Ylm directly by writing recursion as a combination of
    products and then computing the product in logspace.
    """

    l_arr = np.arange(l_max) + 1
    log_prefact = np.cumsum(np.log(2 * l_arr + 1), 0)
    log_prefact -= np.cumsum(np.log(2 * l_arr), 0)

    log_prefact *= 0.5
    log_yll = log_beta_s[None, :] * l_arr[:, None] + log_prefact[:, None]

    sign_yll = ((-1) ** l_arr[:, None]) * np.ones_like(log_yll)

    ylm[0][l_arr, l_arr, :] = log_yll
    ylm[50][l_arr, l_arr, :] = sign_yll

    ylm[0][l_arr, l_arr - 1, :] = (
        ylm[0][l_arr - 1, l_arr - 1, :]
        + log_beta[None, :]
        + 0.5 * np.log(2 * (l_arr[:, None] - 1) + 3)
    )

    ylm[50][l_arr, l_arr - 1, :] = ylm[50][l_arr - 1, l_arr - 1, :]
    # FIXME: assuming beta is always positive.
    return ylm  # ,None


def log_A_lm(l, m):
    """
    Eq. 14 of ref 1, written using logs. logdiffexp is imported from utils.py
    """
    log_Alm = -1 * logdiffexp(2 * np.log(l), 2 * np.log(m))
    log_Alm += logdiffexp(np.log(4) + 2 * np.log(l), 0)
    log_Alm *= 0.5
    return log_Alm


def sYLM_l0_log(l_max, log_beta, l, ylm):
    """
    Computes Ylm using the eq 13 of ref 1. Y_(l-1)m and Y_(l-2)m should be computed already.
    """
    m = np.arange(l_max + 1)

    log_Alm = log_A_lm(l, m)
    log_Alm_prev = log_A_lm(l - 1, m)
    log_Blm = np.where(l - 1 > m, log_Alm - log_Alm_prev, -np.inf)
    log_Alm = np.where(l - 1 > m, log_Alm, -np.inf)

    R1 = log_beta[None, :] + log_Alm[:, None] + ylm[0][l - 1, :, :]
    S1 = ylm[50][l - 1, :, :]
    R2 = log_Blm[:, None] + ylm[0][l - 2, :, :]
    S2 = ylm[50][l - 2, :, :] * -1
    yt, st = logsumexp(R1, R2, S1, S2)

    yt = np.where(l - 1 > m[:, None], yt, 0)
    yt = np.where(l >= m[:, None], yt, -np.inf)
    st = np.where(l - 1 > m[:, None], st, 1)

    ylm[0][l, :, :] += yt
    ylm[50][l, :, :] *= st
    return ylm


def log_alpha_lm(l, m):
    """
    Eq. A8 in ref 2. Recurrence for spin-2 quantities.
    """
    return 0.5 * (np.log(2 * l + 1) + np.log(l**2 - m**2) - np.log(2 * l - 1))
    # FIXME: use log_l and log_m
    # return np.sqrt((2 * l + 1) * (l**2 - m**2) / (2 * l - 1))


def sYLM_l2(log_beta, log_beta_s2, l_max, l, ylm):  # m<l, spin 2
    """
    Compute spin-2 (+/-)  Ylm using spin-0 computation.
    Eq. A7 in ref 2.
    """
    l_indx = np.atleast_1d(l)
    l = np.atleast_1d(l)[:, None, None]
    m = np.arange(l_max + 1)[None, :, None]
    log_beta = log_beta[None, None, :]
    log_beta_s2 = log_beta_s2[None, None, :]

    log_m = np.log(m)

    log_factorial_norm = 0.5 * (
        loggamma(l - 2 + 1) - loggamma(l + 2 + 1)
    )  # eq. A6 of ref 2

    log_alm = log_alpha_lm(l, m)

    m2l, s_m2l = logsumexp(
        log_m * 2, np.log(l), np.ones_like(log_m), -1 * np.ones_like(l)
    )

    ylm_2t = np.log(2) - log_beta_s2 + m2l
    s_ylm_2t = np.ones_like(log_beta_s2) * s_m2l
    del m2l, s_m2l

    ylm_2t, s_ylm_2t = logsumexp(
        ylm_2t,
        np.log(l) + np.log(l - 1),
        s_ylm_2t,
        np.ones_like(l) * -1,
    )

    ylm_2t += ylm[0][l_indx, :, :]
    s_ylm_2t *= ylm[50][l_indx, :, :]

    ylm_2t, s_ylm_2t = logsumexp(
        ylm_2t,
        np.log(2) + log_beta - log_beta_s2 + log_alm + ylm[0][l_indx - 1, :, :],
        s_ylm_2t,
        ylm[50][l_indx - 1, :, :],
    )
    ylm_2t += log_factorial_norm
    ylm_2t = np.where(l >= m, ylm_2t, -np.inf)
    ylm[2][l_indx, :, :] += ylm_2t
    ylm[52][l_indx, :, :] = s_ylm_2t

    ## now doing -2 case
    ylm_2t, s_ylm_2t = logsumexp(
        log_alm + ylm[0][l_indx - 1, :, :],
        np.log(l - 1) + log_beta + ylm[0][l_indx, :, :],
        ylm[50][l_indx - 1, :, :],
        -1 * ylm[50][l_indx, :, :],
    )
    ylm_2t += np.log(2) + log_m - log_beta_s2
    ylm_2t += log_factorial_norm

    ylm_2t = np.where(l >= m, ylm_2t, -np.inf)
    # ylm_2t = np.where(0< m[:, None], ylm_2t, -np.inf)
    ylm[-2][l_indx, :, :] += ylm_2t
    ylm[-52][l_indx, :, :] = s_ylm_2t

    return ylm


def sYLM_recur_log(l_max, spins, log_beta):
    """
    Computes all Ylm using recusion. There is only one loop, over l.
    spin_max: 0 or 2, depending on Ylm desired. Spin_max=2 will return both spin-0 and spin-2 Ylm.
    """
    n_theta = len(log_beta)
    n_lm = (l_max + 1) ** 2
    log_beta_s, _ = logsumexp(
        2 * log_beta,
        np.zeros_like(log_beta),
        -1 * np.ones_like(log_beta),
        np.ones_like(log_beta),
    )  # sin^2=1-cos^2 = 1-beta^2
    log_beta_s *= 0.5  # sqrt
    log_beta = np.where(log_beta < -100, -100, log_beta)  # to prevent nans
    log_beta_s = np.where(log_beta_s < -100, -100, log_beta_s)  # to prevent nans

    ylm = {}

    ylm[0] = np.zeros((l_max + 1, l_max + 1, n_theta))
    ylm[50] = np.ones(
        (l_max + 1, l_max + 1, n_theta), dtype=np.int8
    )  # because 50 is like s0 or sign-0. jax doesnot like strings in keys
    ylm[0][0, 0, :] = 0
    ylm[0][0, 1:, :] = -np.inf
    ylm[0][1, 2:, :] = -np.inf
    ylm[50][0, 0, :] = 1

    sYLM_ll0_i = partial(
        sYLM_ll0_log,
        l_max,
        log_beta,
        log_beta_s,
    )

    # ylm = jax.lax.fori_loop(1, l_max + 1, sYLM_ll0_i, ylm)
    ylm = sYLM_ll0_i(ylm)

    sYLM_l0_i = partial(
        sYLM_l0_log,
        l_max,
        log_beta,
        # np.log(beta_s),
    )

    for l in range(2, l_max + 1):
        ylm = sYLM_l0_i(l, ylm)

    # ylm = jax.lax.fori_loop(2, l_max + 1, sYLM_l0_i, ylm)

    ylm[0] -= 0.5 * np.log(4 * np.pi)

    # if spin_max == 2:
    if 2 in spins or -2 in spins:
        sYLM_l2_i = partial(sYLM_l2, log_beta, log_beta_s * 2, l_max)

        ylm[2] = np.zeros((l_max + 1, l_max + 1, n_theta))
        ylm[-2] = np.zeros((l_max + 1, l_max + 1, n_theta))

        ylm[2][:2, :, :] = -np.inf
        ylm[-2][:2, :, :] = -np.inf

        ylm[52] = np.ones(
            (l_max + 1, l_max + 1, n_theta), dtype=np.int8
        )  # because 52 is like s2 or sign-2. jax doesnot like strings in keys
        ylm[-52] = np.ones((l_max + 1, l_max + 1, n_theta), dtype=np.int8)  #

        ylm = sYLM_l2_i(np.arange(2, l_max + 1), ylm)
        # for l in range(2, l_max + 1):
        # ylm = sYLM_l2_i(l, ylm)
        # ylm = jax.lax.fori_loop(2, l_max + 1, sYLM_l2_i, ylm)
        ylm[2] = np.exp(ylm[2]) * ylm[52]
        ylm[-2] = np.exp(ylm[-2]) * ylm[-52]

    ylm[0] = np.exp(ylm[0]) * ylm[50]  # FIXME: spin2 not implemented yet
    del ylm[50]
    return ylm


##################################

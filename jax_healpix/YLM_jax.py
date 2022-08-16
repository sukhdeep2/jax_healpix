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


def sYLM_ll0(l_max, beta, beta_s, ylm):
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
    log_yll = np.log(beta_s[None, :]) * l_arr[:, None] + log_prefact[:, None]
    yll = ((-1) ** l_arr[:, None]) * np.exp(log_yll)
    ylm[0][l_arr, l_arr, :] = yll

    ylm[0][l_arr, l_arr - 1, :] = (
        ylm[0][l_arr - 1, l_arr - 1, :]
        * beta[None, :]
        * np.sqrt(2 * (l_arr[:, None] - 1) + 3)
    )
    return ylm  # ,None


def A_lm(l, m):
    """
    Eq. 14 of ref 1
    """
    return np.sqrt((4 * l**2 - 1) / (l**2 - m**2))


def log_A_lm(l, m):
    """
    Eq. 14 of ref 1, written using logs. logdiffexp is imported from utils.py
    """
    log_Alm = -1 * logdiffexp(2 * np.log(l), 2 * np.log(m))
    log_Alm += logdiffexp(np.log(4) + 2 * np.log(l), 0)
    log_Alm *= 0.5
    return log_Alm


def alpha_lm(l, m):
    """
    Eq. A8 in ref 2. Recurrence for spin-2 quantities.
    """
    return np.sqrt((2 * l + 1) * (l**2 - m**2) / (2 * l - 1))


def sYLM_l0(l_max, beta, beta_s, l, ylm):
    """
    Computes Ylm using the eq 13 of ref 1. Y_(l-1)m and Y_(l-2)m should be computed already.
    """
    m = np.arange(l_max + 1)

    log_Alm = log_A_lm(l, m)
    log_Alm_prev = log_A_lm(l - 1, m)
    Blm = np.where(l - 1 > m, np.exp(log_Alm - log_Alm_prev), 0)
    Alm = np.where(l - 1 > m, np.exp(log_Alm), 0)

    # Alm = A_lm(l, m)
    # Alm_prev = A_lm(l - 1, m)

    # Blm = np.where(l - 1 > m, Alm / Alm_prev, 0)
    # Alm = np.where(l - 1 > m, Alm, 0)

    ylm[0][l, :, :] += beta[None, :] * Alm[:, None] * ylm[0][l - 1, :, :]
    ylm[0][l, :, :] -= Blm[:, None] * ylm[0][l - 2, :, :]
    # ylm = sYLM_ll0(l_max, beta_s, l, ylm)
    return ylm


def sYLM_l2(beta, beta_s2, l_max, l, ylm):  # m<l, spin 2
    """
    Compute spin-2 (+/-)  Ylm using spin-0 computation.
    Eq. A7 in ref 2.
    """
    m = np.arange(l_max + 1)
    log_factorial_norm = 0.5 * (
        loggamma(l - 2 + 1) - loggamma(l + 2 + 1)
    )  # eq. A6 of ref 2
    alm = alpha_lm(l, m)
    ylm[2][l, :, :] += (
        2 * (m[:, None] ** 2 - l) / beta_s2[None, :] - l * (l - 1)
    ) * ylm[0][l, :, :]

    ylm[2][l, :, :] += (
        2 * (beta / beta_s2)[None, :] * alm[:, None] * ylm[0][l - 1, :, :]
    )

    ylm[-2][l, :, :] += -(l - 1) * beta[None, :] * ylm[0][l, :, :]

    ylm[-2][l, :, :] += alm[:, None] * ylm[0][l - 1, :, :]

    ylm[-2][l, :, :] *= 2 * m[:, None] / beta_s2[None, :]

    ylm[-2][l, :, :] *= np.exp(log_factorial_norm)
    ylm[2][l, :, :] *= np.exp(log_factorial_norm)

    ylm[2] = np.where(l >= m[:, None], ylm[2], 0)
    ylm[-2] = np.where(l >= m[:, None], ylm[-2], 0)
    return ylm


def sYLM_recur(l_max, spins, beta):
    """
    Computes all Ylm using recusion. There is only one loop, over l.
    spin_max: 0 or 2, depending on Ylm desired. Spin_max=2 will return both spin-0 and spin-2 Ylm.
    """
    n_theta = len(beta)
    n_lm = (l_max + 1) ** 2
    # beta_s=-np.sqrt(1-beta**2)
    beta_s = np.sin(np.arccos(beta))

    ylm = {}
    #     ylm[0]=np.zeros((n_lm,n_theta))
    ylm[0] = np.zeros((l_max + 1, l_max + 1, n_theta))
    ylm[0][0, 0, :] = 1

    sYLM_ll0_i = partial(
        sYLM_ll0,
        l_max,
        beta,
        beta_s,
    )

    # ylm = jax.lax.fori_loop(1, l_max + 1, sYLM_ll0_i, ylm)
    ylm = sYLM_ll0_i(ylm)

    sYLM_l0_i = partial(  # partial(
        sYLM_l0,
        l_max,
        beta,
        beta_s,
    )

    # for l in range(1, l_max + 1):
    #     ylm = sYLM_l0(l_max, beta, beta_s, l, ylm)

    # ylm = jax.lax.fori_loop(2, l_max + 1, sYLM_l0_i, ylm)
    for i in range(2, l_max + 1):
        ylm = sYLM_l0_i(i, ylm)

    ylm[0] /= np.sqrt(4 * np.pi)
    # ylm[0] = ylm[0].at[:, 0, :].divide(2)  # FIXME: ????????????
    # ylm[0] = ylm[0].at[:, 1:, :].multiply(2)  # FIXME: ????????????

    # if spin_max == 2:
    if 2 in spins or -2 in spins:
        sYLM_l2_i = partial(sYLM_l2, beta, beta_s**2, l_max)
        ylm[2] = np.zeros((l_max + 1, l_max + 1, n_theta))
        ylm[-2] = np.zeros((l_max + 1, l_max + 1, n_theta))
        for l in range(2, l_max + 1):
            ylm = sYLM_l2_i(l, ylm)
        # ylm = jax.lax.fori_loop(2, l_max + 1, sYLM_l2_i, ylm)

    return ylm


##################################

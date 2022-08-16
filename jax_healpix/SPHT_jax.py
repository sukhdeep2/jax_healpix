"""
Functions to compute the spherical harmonic transforms on the Healpix grid (ring ordered).
See refs 1-3 for relevant equations and definitions.

1. https://arxiv.org/pdf/1010.2084.pdf

2. https://arxiv.org/pdf/astro-ph/0502469.pdf

3. Healpix paper: https://arxiv.org/pdf/astro-ph/0409513.pdf

https://arxiv.org/pdf/astro-ph/0502469.pdf

https://arxiv.org/pdf/1010.2084.pdf

https://arxiv.org/pdf/1303.4945.pdf

https://arxiv.org/pdf/1804.10382.pdf
"""

from skylens.wigner_transform import *

from functools import partial

from YLM_jax_log import *

RING_ITER_SIZE = 256  # FIXME: User should have some control over this.


def ring_beta(nside):
    """
    return beta=cos(theta) for all rings, given nside.
    """
    beta = np.zeros(4 * nside - 1)
    ring_i = np.arange(4 * nside - 1) + 1
    l1 = nside
    beta[:l1] = 1 - ring_i[:l1] ** 2 / 3 / nside**2
    beta[-l1:] = beta[:l1][::-1] * -1
    beta[l1:-l1] = 4.0 / 3 - 2.0 / 3 * ring_i[l1:-l1] / nside
    return beta


def ring_log_beta(nside):
    """
    return beta=cos(theta) for all rings, given nside.
    """
    beta = np.zeros(4 * nside - 1)
    ring_i = np.arange(4 * nside - 1) + 1
    l1 = nside
    pole_beta = np.log(ring_i[:l1]) * 2 - np.log(nside) * 2 - np.log(3)
    pole_beta, _ = logsumexp(
        pole_beta,
        np.zeros_like(pole_beta),
        -1 * np.ones_like(pole_beta),
        np.ones_like(pole_beta),
    )
    beta[:l1] = pole_beta
    beta[-l1:] = pole_beta[::-1]

    # eq_beta = np.log(2.0 / 3) + np.log(ring_i[l1:-l1]) - np.log(nside)
    # eq_beta,_ = logsumexp(
    #     eq_beta,
    #     np.zeros_like(eq_beta),
    #     -1 * np.ones_like(eq_beta),
    #     np.ones_like(eq_beta),
    # )
    eq_beta = np.absolute(4.0 / 3 - 2.0 / 3 * ring_i[l1:-l1] / nside)
    eq_beta = np.log(eq_beta)  # more accurate
    beta[l1:-l1] = eq_beta
    beta_sign = np.ones(4 * nside - 1, dtype=np.int8)
    beta_sign[2 * nside :] = -1
    return beta, beta_sign


def ring_pol(nside, ring_i):
    """
    Ring quantities, for the polar belts
    """
    ring_i2 = np.where(ring_i <= nside, ring_i, 4 * nside - ring_i)

    phi_0 = np.pi / 2 / ring_i2 * 0.5  # j=1
    npix = 4 * ring_i2
    beta = 1 - (ring_i2**2) / 3 / nside**2
    beta = np.where(ring_i <= nside, beta, beta * -1)
    return np.array([phi_0, npix, beta])


def ring_eq(nside, ring_i):
    """
    Ring quantities, for the equitorial belt
    """
    s = np.where(ring_i % 2 == 0, 1, 2)
    phi_0 = np.pi / 2 / nside * (1 - s / 2.0)  # j=1
    npix = 4 * nside * ring_i / ring_i
    beta = 4.0 / 3 - 2.0 / 3 * ring_i / nside
    return np.array([phi_0, npix, beta])


def phi_m(nside, l_max, ring_i, phase):
    """
    return e^(im\phi) for all pixels in a given ring.
    """
    phi_0, npix, beta = np.where(
        np.logical_or(ring_i < nside, ring_i > 3 * nside),
        ring_pol(nside, ring_i),
        ring_eq(nside, ring_i),
    )
    m = np.arange(l_max + 1)
    x = np.arange(4 * nside)
    # phi = np.where(x < npix, phi_0 + 2 * np.pi * x / npix, 0)
    phi = (
        phi_0[:, None] + 2 * np.pi * x[None, :] / npix[:, None]
    )  # x>=npix nulled below
    phi = np.exp(1j * phase * m[None, None, :] * phi[:, :, None])
    j_pix = x
    phi = np.where(
        j_pix[None, :, None] < npix[:, None, None], phi, 0
    )  # no data for # x>=npix in a ring
    return phi, beta


def north_ring_ylm(nside, l_max, spins, log_beta, ring_i0, phi_phase):
    iter_size = min(RING_ITER_SIZE, 2 * nside)
    ring_i = np.arange(iter_size) + ring_i0 * iter_size + 1
    # ring_i = np.where(ring_i > 2 * nside, 0, ring_i)
    log_beta = np.where(ring_i <= 2 * nside, log_beta[ring_i - 1], 0)

    phi, _ = phi_m(nside, l_max, ring_i, phi_phase)
    ylm = sYLM_recur_log(l_max=l_max, spins=spins, log_beta=log_beta)

    for s in ylm.keys():
        ylm[s] = np.where(
            ring_i[None, None, :] <= 2 * nside, ylm[s], 0
        )  # no ring on south side

    return ring_i, ylm, phi


def south_ring_ylm(nside, l_max, ring_i, ylm):
    l = np.arange(l_max + 1)
    for s in ylm.keys():
        ylm[s] *= (-1) ** (l[:, None, None] + l[None, :, None])
        ylm[s] = np.where(
            ring_i[None, None, :] < 2 * nside, ylm[s], 0
        )  # no repeat on ring at equator
    if 2 in ylm.keys():
        ylm[-2] *= -1
    ring_i = 4 * nside - ring_i - 1  # FIXME: wont work with more than 1 spin
    ring_i += 1  # 1 us subtracted when indexing maps
    return ring_i, ylm


def Gmy(maps, phi):  # tj,jm->tm
    return maps @ phi


v_Gmy = np.vectorize(Gmy)  # , in_axes=(1, 0))  # return shape: rtm


def Gmy2alm(Gmy, ylm):  # rtm,lmr ... rt,lr
    return ylm @ Gmy


v_Gmy2alm = np.vectorize(Gmy2alm)  # , in_axes=(2, 1))  # return shape mlt


def ring2alm_ns_dot(ring_i, maps, phi, ylm):
    alm_t = np.einsum(  #
        "trj,rjm,lmr->tlm",
        maps[:, ring_i - 1, :],
        phi,
        ylm,  # [:, :, ring_i - 1],
    )
    # Gmy = v_Gmy(maps[s][:, ring_i - 1, :], phi)

    # alm_t = v_Gmy2alm(v_Gmy(maps[:, ring_i - 1, :], phi), ylm).transpose(2, 1, 0)
    # alm_t = alm_t.transpose(2, 1, 0)
    return alm_t


def ring2alm_ns(spins, ring_i, ylm, maps, phi, alm):  # north or south
    if 0 in spins:
        s = 0
        alm[s] += ring2alm_ns_dot(ring_i, maps[s], phi, ylm[s])

    if 2 in spins:
        y_s, m_s = 2, 2  # E-mode
        alm[2] += ring2alm_ns_dot(
            ring_i, maps[m_s], phi, ylm[y_s]
        ) + 1j * ring2alm_ns_dot(ring_i, maps[m_s * -1], phi, ylm[m_s * -1])
        # alm[2] += alm_t

        y_s, m_s = -2, 2  # B-mode
        alm[-2] += ring2alm_ns_dot(
            ring_i, maps[m_s], phi, ylm[y_s]
        ) + 1j * ring2alm_ns_dot(ring_i, maps[m_s * -1], phi, ylm[y_s * -1])

        # alm[-2] += alm_t
        # B-mode has extra 1j factor, which is multiplied in th map2alm
    return alm


def ring2alm(nside, l_max, spins, maps, log_beta, ring_i0, alm):
    """
    Computes alm for a given ring and add to the alm vector.
    """
    # ring_i = np.atleast_1d(ring_i)

    ring_i, ylm, phi = north_ring_ylm(nside, l_max, spins, log_beta, ring_i0, -1)
    alm = ring2alm_ns(spins, ring_i, ylm, maps, phi, alm)

    ring_i, ylm = south_ring_ylm(nside, l_max, ring_i, ylm)
    alm = ring2alm_ns(spins, ring_i, ylm, maps, phi, alm)

    return alm


def map2alm(nside, l_max, spins, maps):
    alm = {
        s: np.zeros((maps[s].shape[0], l_max + 1, l_max + 1), dtype=np.complex_)
        for s in maps.keys()
    }
    log_beta, _ = ring_log_beta(nside)
    # ylm = sYLM_recur_log(l_max=l_max, spins=spins, beta=beta)
    ring2alm_i = partial(ring2alm, nside, l_max, spins, maps, log_beta)

    pix_area = 4 * np.pi / 12 / nside**2

    niter = max(1, (2 * nside) // RING_ITER_SIZE)

    # FIXME: loop is costly, needed to lower the peak memory usage inside ring2alm

    for i in range(niter):
        alm = ring2alm_i(i, alm)

    # alm = ring2alm_i(np.arange(1, 4 * nside), alm)

    for s in maps.keys():
        alm[s] *= pix_area
    if 2 in maps.keys():
        alm[-2] *= 1j
        alm[2] *= -1
    return alm


def map2alm_iteration(nside, l_max, spins, maps, iteration, alm):
    map_t = alm2map(nside, l_max, spins, alm)
    diff_map = {s: maps[s] - map_t[s] for s in maps.keys()}
    alm_diff = map2alm(nside, l_max, spins, diff_map)
    for s in maps.keys():
        alm[s] += alm_diff[s]
    return alm


def map2alm_iter(nside, l_max, spins, niter, maps):
    """
    https://healpix.sourceforge.io/html/sub_map2alm_iterative.htm
    https://github.com/healpy/healpy/issues/676#issuecomment-990887530.
    """
    alm = map2alm(nside, l_max, spins, maps)
    map2alm_i = partial(map2alm_iteration, nside, l_max, spins, maps)

    for i in range(0, niter):
        alm = map2alm_i(i, alm)
    #     alm = map2alm_iteration(nside, l_max, spins, maps, i, alm)
    return alm


def Fmy(alm, ylm):  # loop over m
    return alm @ ylm


v_Fmy = np.vectorize(Fmy)  # , in_axes=(2, 1))


def Fmy2map(Fmy, phi):  # loop over r
    return phi @ Fmy


v_Fmy2map = np.vectorize(Fmy2map)  # , in_axes=(2, 0))


def alm2ring_ns_dot(ring_i, ylm, phi, alm):
    mt = np.einsum("tlm,lmr,rjm->trj", alm, ylm, phi)
    return mt
    # # Fmy = v_Fmy(alm[s], ylm[s])  # mtr
    # mt = v_Fmy2map(v_Fmy(alm, ylm), phi)  # rjt
    # return mt.transpose(2, 0, 1)  # trj


def alm2ring_ns(ring_i, ylm, alm, phi, maps):
    if 0 in alm.keys():
        s = 0
        maps[s][:, ring_i - 1, :] += alm2ring_ns_dot(ring_i, ylm[s], phi, alm[s])

    if 2 in alm.keys():
        y_s, a_s, m_s = 2, 2, 2  # Q map
        maps[m_s][:, ring_i - 1, :] += alm2ring_ns_dot(
            ring_i, ylm[y_s], phi, alm[a_s]
        ) + 1j * alm2ring_ns_dot(ring_i, ylm[y_s * -1], phi, alm[a_s * -1])

        y_s, a_s, m_s = -2, 2, -2  # U map
        maps[m_s][:, ring_i - 1, :] += alm2ring_ns_dot(
            ring_i, ylm[y_s], phi, alm[a_s]
        ) + 1j * alm2ring_ns_dot(ring_i, ylm[y_s * -1], phi, alm[a_s * -1])
    return maps


def alm2ring(nside, l_max, spins, alm, log_beta, ring_i0, maps):

    ring_i, ylm, phi = north_ring_ylm(nside, l_max, spins, log_beta, ring_i0, 1)
    maps = alm2ring_ns(ring_i, ylm, alm, phi, maps)

    ring_i, ylm = south_ring_ylm(nside, l_max, ring_i, ylm)
    maps = alm2ring_ns(ring_i, ylm, alm, phi, maps)

    return maps


def alm2map(nside, l_max, spins, alm):

    for s in alm.keys():
        alm[s][:, :, 1:] *= 2  # to correct for missing m<0
    maps = {s: np.zeros((len(alm[s]), 4 * nside - 1, 4 * nside)) for s in alm.keys()}
    if 2 in alm.keys():
        s = -2
        maps = {
            s: np.zeros((len(alm[s]), 4 * nside - 1, 4 * nside), dtype=np.complex_)
            for s in alm.keys()
        }
    log_beta, _ = ring_log_beta(nside)

    niter = max(1, (2 * nside) // RING_ITER_SIZE)
    # FIXME: loop is costly, needed to lower the peak memory usage inside alm2ring
    alm2ring_i = partial(alm2ring, nside, l_max, spins, alm, log_beta)

    for i in range(niter):
        maps = alm2ring_i(i, maps)
    # maps = alm2ring_i(np.arange(1, 4 * nside), maps)

    for s in alm.keys():
        alm[s][:, :, 1:] /= 2  # undo multiplication above
    if 2 in maps.keys():
        maps[-2] *= 1j
        maps[2] *= -1
    return maps


def alm2cl(l_max, alm, alm2=None):
    alm[:, :, 1:] *= np.sqrt(2)  # because we donot compute m<0
    if alm2 is None:
        alm2 = alm
    else:
        alm2[:, :, 1:] *= np.sqrt(2)  # because we donot compute m<0
    Cl = np.real((alm * np.conjugate(alm2)).sum(axis=2))
    Cl /= 2 * np.arange(l_max + 1)[None, :] + 1
    return Cl

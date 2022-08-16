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

# from functools import partial
from jax.tree_util import Partial as partial

import jax
import jax.numpy as jnp
from jax import jit

# from YLM_jax import *
from YLM_jax_log import *

RING_ITER_SIZE = 256  # FIXME: User should have some control over this.


def ring_beta(nside):
    """
    return beta=cos(theta) for all rings, given nside.
    """
    beta = jnp.zeros(4 * nside - 1)
    ring_i = jnp.arange(4 * nside - 1) + 1
    l1 = nside
    beta = beta.at[:l1].set(1 - ring_i[:l1] ** 2 / 3 / nside**2)
    beta = beta.at[-l1:].set(beta[:l1][::-1] * -1)
    beta = beta.at[l1:-l1].set(4.0 / 3 - 2.0 / 3 * ring_i[l1:-l1] / nside)
    return beta


def ring_log_beta(nside):
    """
    return beta=cos(theta) for all rings, given nside.
    """
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

    # eq_beta = jnp.log(2.0 / 3) + jnp.log(ring_i[l1:-l1]) - jnp.log(nside)
    # eq_beta,_ = logsumexp(
    #     eq_beta,
    #     jnp.zeros_like(eq_beta),
    #     -1 * jnp.ones_like(eq_beta),
    #     jnp.ones_like(eq_beta),
    # )
    eq_beta = jnp.absolute(4.0 / 3 - 2.0 / 3 * ring_i[l1:-l1] / nside)
    eq_beta = jnp.log(eq_beta)  # more accurate
    beta = beta.at[l1:-l1].set(eq_beta)
    beta_sign = jnp.ones(4 * nside - 1, dtype=jnp.int8)
    beta_sign = beta_sign.at[2 * nside :].set(-1)
    return beta, beta_sign


def ring_pol(nside, ring_i):
    """
    Ring quantities, for the polar belts
    """
    ring_i2 = jnp.where(ring_i <= nside, ring_i, 4 * nside - ring_i)

    phi_0 = jnp.pi / 2 / ring_i2 * 0.5  # j=1
    npix = 4 * ring_i2
    beta = 1 - (ring_i2**2) / 3 / nside**2
    beta = jnp.where(ring_i <= nside, beta, beta * -1)
    return jnp.array([phi_0, npix, beta])


def ring_eq(nside, ring_i):
    """
    Ring quantities, for the equitorial belt
    """
    s = jnp.where(ring_i % 2 == 0, 1, 2)
    phi_0 = jnp.pi / 2 / nside * (1 - s / 2.0)  # j=1
    npix = 4 * nside * ring_i / ring_i
    beta = 4.0 / 3 - 2.0 / 3 * ring_i / nside
    return jnp.array([phi_0, npix, beta])


@partial(jax.jit, static_argnums=(0, 1))
def phi_m(nside, l_max, ring_i, phase):
    """
    return e^(im\phi) for all pixels in a given ring.
    """
    phi_0, npix, beta = jnp.where(
        jnp.logical_or(ring_i < nside, ring_i > 3 * nside),
        ring_pol(nside, ring_i),
        ring_eq(nside, ring_i),
    )
    m = jnp.arange(l_max + 1)
    x = jnp.arange(4 * nside)
    # phi = jnp.where(x < npix, phi_0 + 2 * jnp.pi * x / npix, 0)
    phi = (
        phi_0[:, None] + 2 * jnp.pi * x[None, :] / npix[:, None]
    )  # x>=npix nulled below
    phi = jnp.exp(1j * phase * m[None, None, :] * phi[:, :, None])
    j_pix = x
    phi = jnp.where(
        j_pix[None, :, None] < npix[:, None, None], phi, 0
    )  # no data for # x>=npix in a ring
    return phi, beta


@partial(jax.jit, static_argnums=(0, 1, 2))
def north_ring_ylm(nside, l_max, spins, log_beta, ring_i0, phi_phase):
    ring_i = jnp.arange(RING_ITER_SIZE) + ring_i0 * RING_ITER_SIZE + 1
    # ring_i = jnp.where(ring_i > 2 * nside, 0, ring_i)
    log_beta = jnp.where(ring_i <= 2 * nside, log_beta[ring_i - 1], 0)

    phi, _ = phi_m(nside, l_max, ring_i, phi_phase)
    ylm = sYLM_recur_log(l_max=l_max, spins=spins, log_beta=log_beta)

    for s in ylm.keys():
        ylm[s] = jnp.where(
            ring_i[None, None, :] <= 2 * nside, ylm[s], 0
        )  # no ring on south side

    return ring_i, ylm, phi


@partial(jax.jit, static_argnums=(0, 1))
def south_ring_ylm(nside, l_max, ring_i, ylm):
    l = jnp.arange(l_max + 1)
    for s in ylm.keys():
        ylm[s] = (
            ylm[s]
            .at[:, :, :]
            .set(ylm[s] * ((-1) ** (l[:, None, None] + l[None, :, None])))
        )  # ylm (-beta) = (-1)^{l+m} ylm(beta)
        ylm[s] = jnp.where(
            ring_i[None, None, :] < 2 * nside, ylm[s], 0
        )  # no repeat on ring at equator
    if 2 in ylm.keys():
        ylm[-2] *= -1
    ring_i = 4 * nside - ring_i - 1  # FIXME: wont work with more than 1 spin
    ring_i += 1  # 1 us subtracted when indexing maps
    return ring_i, ylm


def Gmy(maps, phi):  # tj,jm->tm
    return maps @ phi


v_Gmy = jax.vmap(Gmy, in_axes=(1, 0))  # return shape: rtm


def Gmy2alm(Gmy, ylm):  # rtm,lmr ... rt,lr
    return ylm @ Gmy


v_Gmy2alm = jax.vmap(Gmy2alm, in_axes=(2, 1))  # return shape mlt


@jit
def ring2alm_ns_dot(ring_i, maps, phi, ylm):
    # alm_t = jnp.einsum(  #
    #     "trj,rjm,lmr->tlm",
    #     maps[s][:, ring_i - 1, :],
    #     phi,
    #     ylm[s],  # [:, :, ring_i - 1],
    # )
    # Gmy = v_Gmy(maps[s][:, ring_i - 1, :], phi)

    alm_t = v_Gmy2alm(v_Gmy(maps[:, ring_i - 1, :], phi), ylm).transpose(2, 1, 0)
    # alm_t = alm_t.transpose(2, 1, 0)
    return alm_t


@partial(jax.jit, static_argnums=(0,))
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


@partial(jax.jit, static_argnums=(0, 1, 2))
def ring2alm(nside, l_max, spins, maps, log_beta, ring_i0, alm):
    """
    Computes alm for a given ring and add to the alm vector.
    """
    # ring_i = jnp.atleast_1d(ring_i)

    ring_i, ylm, phi = north_ring_ylm(nside, l_max, spins, log_beta, ring_i0, -1)
    alm = ring2alm_ns(spins, ring_i, ylm, maps, phi, alm)

    ring_i, ylm = south_ring_ylm(nside, l_max, ring_i, ylm)
    alm = ring2alm_ns(spins, ring_i, ylm, maps, phi, alm)

    return alm


@partial(jax.jit, static_argnums=(0, 1, 2))
def map2alm(nside, l_max, spins, maps):
    alm = {
        s: jnp.zeros((maps[s].shape[0], l_max + 1, l_max + 1), dtype=jnp.complex64)
        for s in maps.keys()
    }
    log_beta, _ = ring_log_beta(nside)
    # ylm = sYLM_recur_log(l_max=l_max, spins=spins, beta=beta)
    ring2alm_i = jax.tree_util.Partial(ring2alm, nside, l_max, spins, maps, log_beta)

    pix_area = 4 * jnp.pi / 12 / nside**2

    niter = max(1, (2 * nside) // RING_ITER_SIZE)
    alm = jax.lax.fori_loop(0, niter, ring2alm_i, alm)
    # FIXME: loop is costly, needed to lower the peak memory usage inside ring2alm

    # for i in range(niter):
    #     alm = ring2alm(nside, l_max, spins, maps, beta, i, alm)

    # alm = ring2alm_i(jnp.arange(1, 4 * nside), alm)

    for s in maps.keys():
        alm[s] = alm[s].at[:].multiply(pix_area)
    if 2 in maps.keys():
        alm[-2] *= 1j
        alm[2] *= -1
    return alm


@partial(jax.jit, static_argnums=(0, 1, 2))
def map2alm_iteration(nside, l_max, spins, maps, iteration, alm):
    map_t = alm2map(nside, l_max, spins, alm)
    diff_map = {s: maps[s] - map_t[s] for s in maps.keys()}
    alm_diff = map2alm(nside, l_max, spins, diff_map)
    for s in maps.keys():
        alm[s] = alm[s].at[:].add(alm_diff[s])
    return alm


# @partial(jax.jit, static_argnums=(0, 1, 2))
def map2alm_iter(nside, l_max, spins, niter, maps):
    """
    https://healpix.sourceforge.io/html/sub_map2alm_iterative.htm
    https://github.com/healpy/healpy/issues/676#issuecomment-990887530.
    """
    alm = map2alm(nside, l_max, spins, maps)
    map2alm_i = jax.tree_util.Partial(map2alm_iteration, nside, l_max, spins, maps)
    alm = jax.lax.fori_loop(0, niter, map2alm_i, alm)
    # for i in range(0, niter):
    #     # alm = map2alm_i(i, alm)
    #     alm = map2alm_iteration(nside, l_max, spins, maps, i, alm)
    return alm


@jit
def Fmy(alm, ylm):  # loop over m
    return alm @ ylm


v_Fmy = jax.vmap(Fmy, in_axes=(2, 1))


def Fmy2map(Fmy, phi):  # loop over r
    return phi @ Fmy


v_Fmy2map = jax.vmap(Fmy2map, in_axes=(2, 0))


@jit
def alm2ring_ns(ring_i, ylm, alm, phi, maps):
    s = 0
    # mt = jnp.einsum("tlm,lmr,rjm->trj", alm[s], ylm[s], phi)
    # Fmy = v_Fmy(alm[s], ylm[s])  # mtr
    mt = v_Fmy2map(v_Fmy(alm[s], ylm[s]), phi)  # rjt
    mt = mt.transpose(2, 0, 1)  # trj
    maps[s] = maps[s].at[:, ring_i - 1, :].add(mt)
    del mt

    if 2 in alm.keys():
        for y_s in (2, -2):
            for a_s in (2, -2):
                mt = v_Fmy2map(v_Fmy(alm[a_s], ylm[y_s]), phi)  # rjt
                mt = mt.transpose(2, 0, 1)  # trj

                m_s = a_s * y_s // 2
                map_f = jnp.where(
                    a_s == 2, 1, 1j
                )  # B-mode has extra 1j factor, which is multiplied in th map2alm

                maps[m_s] = maps[m_s].at[:, ring_i - 1, :].add(mt * map_f)
                del mt
    return maps


@partial(jax.jit, static_argnums=(0, 1, 2))
def alm2ring(nside, l_max, spins, alm, log_beta, ring_i0, maps):

    ring_i, ylm, phi = north_ring_ylm(nside, l_max, spins, log_beta, ring_i0, 1)
    maps = alm2ring_ns(ring_i, ylm, alm, phi, maps)

    ring_i, ylm = south_ring_ylm(nside, l_max, ring_i, ylm)
    maps = alm2ring_ns(ring_i, ylm, alm, phi, maps)

    return maps


# @partial(jax.jit, static_argnums=(0, 1, 2))
def alm2map(nside, l_max, spins, alm):

    for s in alm.keys():
        alm[s] = alm[s].at[:, :, 1:].multiply(2)  # to correct for missing m<0
    maps = {s: jnp.zeros((len(alm[s]), 4 * nside - 1, 4 * nside)) for s in alm.keys()}
    if 2 in alm.keys():
        s = -2
        maps = {
            s: jnp.zeros((len(alm[s]), 4 * nside - 1, 4 * nside), dtype=jnp.complex_)
            for s in alm.keys()
        }
    log_beta, _ = ring_log_beta(nside)

    niter = max(1, (2 * nside) // RING_ITER_SIZE)
    # FIXME: loop is costly, needed to lower the peak memory usage inside alm2ring
    alm2ring_i = jax.tree_util.Partial(alm2ring, nside, l_max, spins, alm, log_beta)
    maps = jax.lax.fori_loop(0, niter, alm2ring_i, maps)

    # for i in range(niter):
    #     maps = alm2ring(nside, l_max, spins, alm, beta, i, maps)
    # maps = alm2ring_i(jnp.arange(1, 4 * nside), maps)

    for s in alm.keys():
        alm[s] = alm[s].at[:, :, 1:].divide(2)  # undo multiplication above
    if 2 in maps.keys():
        maps[-2] *= 1j
        maps[2] *= -1
    return maps


def alm2cl(l_max, alm, alm2=None):
    alm = alm.at[:, :, 1:].multiply(jnp.sqrt(2))  # because we donot compute m<0
    if alm2 is None:
        alm2 = alm
    else:
        alm2 = alm2.at[:, :, 1:].multiply(jnp.sqrt(2))  # because we donot compute m<0
    Cl = jnp.real((alm * jnp.conjugate(alm2)).sum(axis=2))
    Cl = Cl.at[:, :].divide(2 * jnp.arange(l_max + 1)[None, :] + 1)
    return Cl

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

import jax
import jax.numpy as jnp
from jax import jit

from YLM_jax import *


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
    npix = 4 * nside
    beta = 4.0 / 3 - 2.0 / 3 * ring_i / nside
    return jnp.array([phi_0, npix, beta])


@partial(jax.jit, static_argnums=(0, 1))
def fft_phi_m(nside, l_max, ring_i, phase):
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
    phi = phi_0 + 2 * jnp.pi * x / npix  # x>=npix nulled below
    fft_phi = jnp.exp(1j * phase * m[None, :] * phi[:, None])
    j_pix = x
    fft_phi = jnp.where(
        j_pix[:, None] < npix, fft_phi, 0
    )  # no data for # x>=npix in a ring
    return fft_phi, beta


# @partial(jax.jit, static_argnums=(0,1,2))
def ring2alm(nside, l_max, spin_max, maps, ylm, ring_i, alm):
    """
    Computes alm for a given ring and add to the alm vector.
    """
    fft_phi, beta = fft_phi_m(nside, l_max, ring_i, -1)

    for s in maps.keys():
        Gmy = maps[s][:, ring_i - 1, :] @ fft_phi
        # ylm = sYLM_recur(l_max=l_max, spin_max=spin_max, beta=jnp.atleast_1d(beta)) #reduces memory, but slower.
        alm[s] = (
            alm[s].at[:, :].add(Gmy * ylm[s][:, :, ring_i - 1])
        )  # this 1 y, no jnp.sum.
    return alm


@partial(jax.jit, static_argnums=(0, 1, 2))
def map2alm(nside, l_max, spin_max, maps):
    alm = {
        s: jnp.zeros((1, l_max + 1, l_max + 1), dtype=jnp.complex64)
        for s in maps.keys()
    }
    betas = ring_beta(nside)
    ylm = sYLM_recur(l_max=l_max, spin_max=spin_max, beta=betas)
    ring2alm_i = partial(ring2alm, nside, l_max, spin_max, maps, ylm)

    pix_area = 4 * jnp.pi / 12 / nside**2

    alm = jax.lax.fori_loop(1, 4 * nside, ring2alm_i, alm)
    for s in maps.keys():
        alm[s] = alm[s].at[:].multiply(pix_area)
    return alm


# @partial(jax.jit, static_argnums=(0, 1))
def alm2ring(nside, l_max, alm, ylm, ring_i, maps):
    fft_phi, beta = fft_phi_m(nside, l_max, ring_i, 1)
    for s in alm.keys():
        Fmy = jnp.einsum("ilm,lm->im", alm[s], ylm[s][:, :, ring_i - 1])
        maps[s] = (
            maps[s].at[:, ring_i - 1, :].add(jnp.einsum("im,xm->ix", Fmy, fft_phi))
        )
        # spin2
    return maps


@partial(jax.jit, static_argnums=(0, 1, 2))
def alm2map(nside, l_max, spin_max, alm):
    spins = jnp.arange(spin_max + 1, step=2)
    for s in alm.keys():
        alm[s] = alm[s].at[:, :, 1:].multiply(2)
    nmaps = 1
    maps = {s: jnp.zeros((nmaps, 4 * nside - 1, 4 * nside)) for s in alm.keys()}
    betas = ring_beta(nside)
    ylm = sYLM_recur(l_max=l_max, spin_max=spin_max, beta=betas)
    # Fmy = {i: jnp.einsum("ilm,lmr->imr", alm[i], ylm[i]) for i in spins}
    alm2ring_i = partial(alm2ring, nside, l_max, alm, ylm)
    maps = jax.lax.fori_loop(1, 4 * nside, alm2ring_i, maps)
    return maps


# @partial(jax.jit, static_argnums=(0, 1, 2))
def map2alm_iteration(nside, l_max, spin_max, maps, iteration, alm):
    map_t = alm2map(nside, l_max, spin_max, alm)
    diff_map = {s: maps[s] - map_t[s] for s in maps.keys()}  # FIXME
    alm_diff = map2alm(nside, l_max, spin_max, diff_map)
    for s in maps.keys():
        alm[s] = alm[s].at[:].add(alm_diff[s])
    return alm


# @partial(jax.jit, static_argnums=(0, 1, 2, 3))
def map2alm_iter(nside, l_max, spin_max, niter, maps):
    """
    https://healpix.sourceforge.io/html/sub_map2alm_iterative.htm
    https://github.com/healpy/healpy/issues/676#issuecomment-990887530.
    """
    alm = map2alm(nside, l_max, spin_max, maps)
    map2alm_i = jax.tree_util.Partial(map2alm_iteration, nside, l_max, spin_max, maps)
    alm = jax.lax.fori_loop(0, niter, map2alm_i, alm)
    # for i in range(1, niter):
    #     # alm = map2alm_i(i, alm)
    #     alm = map2alm_iteration(nside, l_max, spin_max, maps, i, alm)
    return alm


def alm2cl(l_max, alm):
    alm = alm.at[:, :, 1:].multiply(jnp.sqrt(2))  # because we donot compute m<0
    Cl = jnp.real((alm * jnp.conjugate(alm)).sum(axis=2))
    Cl = Cl.at[:, :].divide(2 * jnp.arange(l_max + 1)[None, :] + 1)
    return Cl

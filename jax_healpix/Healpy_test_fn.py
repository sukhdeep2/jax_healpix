import healpy as hp
import numpy as np
import jax.numpy as jnp
from SPHT_jax import *


def hp_pix_trans_i(nside):
    npix = hp.nside2npix(nside)

    ring_i_unique_pole = np.arange(nside - 1) + 1
    z = 1 - ring_i_unique_pole**2 / 3 / nside**2
    ring_i_unique_eq = np.arange(nside, 3 * nside + 1)
    print(len(ring_i_unique_pole) * 2 + len(ring_i_unique_eq))
    z = np.append(z, 4 / 3 - 2 * ring_i_unique_eq / 3 / nside)
    repeats = np.append(
        ring_i_unique_pole * 4, 4 * nside * np.ones_like(ring_i_unique_eq)
    )
    z = np.append(z, z[:-1][::-1] * -1)
    repeats = np.append(repeats, repeats[:-1][::-1])

    fft_phi = np.zeros(npix)

    l1 = nside * (nside - 1) * 2  # *4/2

    fft_phi[:l1] += np.hstack(
        [
            2 * np.pi * (np.arange(4 * i)) / 4 / i * 0 + np.pi / 4 / i
            for i in np.arange(nside - 1) + 1
        ]
    )

    fft_phi[-l1:] = np.hstack(
        [
            2 * np.pi * (np.arange(4 * i)) / 4 / i * 0 + np.pi / 4 / i
            for i in np.arange(nside - 1)[::-1] + 1
        ]
    )

    l2 = npix - 2 * l1
    ring_i_unique_eq = np.arange(nside, 3 * nside + 1)
    s = (ring_i_unique_eq - nside + 1) % 2
    s[s == 0] = 2  # FIXME: To match with healpy
    fft_phi[l1:-l1] = np.hstack(
        [
            2 * np.pi * (np.arange(4 * nside)) / 4 / nside * 0
            + np.pi / 2 / nside * (1.0 - i / 2)
            for i in s
        ]
    )

    return z, repeats, fft_phi


def hp_pix2ang(nside):
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)

    ring_i = np.zeros(npix)
    ring_j = np.zeros(npix)

    z = np.zeros(npix)
    phi = np.zeros(npix)

    l1 = nside * (nside - 1) * 2  # *4/2

    ring_i[:l1] = np.repeat(np.arange(nside - 1) + 1, (1 + np.arange(nside - 1)) * 4)
    ring_j[:l1] = np.hstack([np.arange(4 * i) + 1 for i in np.arange(nside - 1) + 1])
    z[:l1] = 1 - ring_i[:l1] ** 2 / 3 / nside**2
    phi[:l1] = np.pi / 2 / ring_i[:l1] * (ring_j[:l1] - 0.5)

    ring_i[-l1:] = ring_i[:l1][::-1]
    ring_j[-l1:] = np.hstack(
        [np.arange(4 * i) + 1 for i in np.arange(nside - 1)[::-1] + 1]
    )

    z[-l1:] = z[:l1][::-1] * -1
    phi[-l1:] = np.pi / 2 / ring_i[-l1:] * (ring_j[-l1:] - 0.5)

    l2 = npix - 2 * l1
    ring_i[l1:-l1] = np.repeat(np.arange(nside, 3 * nside + 1), 4 * nside)
    ring_j[l1:-l1] = np.tile(np.arange(4 * nside) + 1, 2 * nside + 1)

    z[l1:-l1] = 4.0 / 3 - 2.0 / 3 * ring_i[l1:-l1] / nside

    s = (ring_i[l1:-l1] - nside + 1) % 2
    s[s == 0] = 2  # FIXME: To match with healpy
    phi[l1:-l1] = np.pi / 2 / nside * (ring_j[l1:-l1] - s / 2)

    return z, phi, ring_i, ring_j


def get_phi0_beta(nside):
    z, phi, ring_i2, ring_j = hp_pix2ang(nside)
    beta = ring_beta(nside)
    phi2 = np.zeros(12 * nside**2)
    j = 0
    for i in ring_i2:
        ring_i = i
        phi_0, npix, _ = jnp.where(
            jnp.logical_or(ring_i < nside, ring_i > 3 * nside),
            ring_pol(nside, ring_i),
            ring_eq(nside, ring_i),
        )
        phi2[j] += phi_0
        x = ring_i2 == i
        phi[x] = phi[x][0]
        j += 1
    print(np.all(np.isclose(phi2, phi)))
    print(np.all(np.isclose(np.unique(beta), np.unique(z))))
    return phi2, phi


# z, repeats, fft_phi = hp_pix_trans_i(nside)
# fft_phi2, fft_phi22 = get_phi0_beta(nside)
# np.isclose(np.unique(fft_phi2), np.unique(fft_phi))

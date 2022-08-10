from SPHT_jax import *

"""
Utils to convert 1-D healpy array into 2-D arrays needed for SPHT_jax functions
and vice-versa.
"""
####################################################################
#                       alm
####################################################################
def set_m_indxs(l_max, m, indxs):
    """util for alm_indxs. sets right indxs for all ell and a given m"""
    i = jnp.arange(l_max + 1)
    indxs = indxs.at[:, m].set(
        jnp.where(i >= m, indxs[:, m] - (m * (m - 1) / 2) - m, -1)
    )
    return indxs


def alm_indxs(l_max):
    """indxs to covert 1-D alms to 2-D alm array"""
    indxs = jnp.arange((l_max + 1) ** 2).reshape(l_max + 1, l_max + 1).T
    set_m_indxs_i = partial(set_m_indxs, l_max)
    indxs = jax.lax.fori_loop(1, l_max + 1, set_m_indxs_i, indxs)
    return indxs  # m-order. for l use ii.T


def reshape_alm(l_max, alm):
    """Convert 1-D alm to 2-D alm"""
    indxs = alm_indxs(l_max)
    alm2 = alm[:, indxs]
    alm2 = jnp.where(
        indxs >= 0,
        alm2[
            :,
        ],
        0,
    )
    return alm2


def l_stack_mask(l_max):
    """
    Ylm are computed in l_max X l_max matrix. Half of those elements are zero.
    This function generates a mask to stack Ylm in 1-D array ordered by l.
    """
    m_mat = jnp.tile(jnp.arange(l_max + 1), l_max + 1).reshape(l_max + 1, l_max + 1)
    l_mat = m_mat.T
    mask = m_mat <= l_mat
    return mask


def m_stack_mask(l_max):
    """
    Similar to l_stack_mask, this function generates a mask to stack Ylm in
    1-D array ordered by m (similar to healpy).
    """
    l_mat = jnp.tile(jnp.arange(l_max + 1), l_max + 1).reshape(l_max + 1, l_max + 1)
    m_mat = l_mat.T
    mask = m_mat <= l_mat
    return mask


def stack_alm(l_max, alm, order="l", in_place=False):
    """
    Stack 2-D alm into 1-D alm arrays
    """
    if order == "l":
        mask = l_stack_mask(l_max)
    if order == "m":
        mask = m_stack_mask(l_max)
    if in_place:
        for k in alm.keys():
            if order == "l":
                alm[k] = alm[k][:, mask]
            else:
                alm[k] = alm[k].transpose(0, 2, 1)[:, mask]
        return ylm
    else:
        alm_s = {}
        for k in alm.keys():
            if order == "l":
                alm_s[k] = alm[k][:, mask]
            else:
                alm_s[k] = alm[k].transpose(0, 2, 1)[:, mask]
        return alm_s


@partial(jax.jit, static_argnums=(0, 2))
def stack_ylm(l_max, ylm, order="l", in_place=False):
    """
    Get the mask and then stack all the Ylm
    """
    if order == "l":
        mask = l_stack_mask(l_max)
    if order == "m":
        mask = m_stack_mask(l_max)
    if in_place:
        for k in ylm.keys():
            ylm[k] = ylm[k][mask, :]
        return ylm
    else:
        ylm_s = {}
        for k in ylm.keys():
            ylm_s[k] = ylm[k][mask, :]
        return ylm_s


# l_mat=jnp.tile(jnp.arange(l_max+1),l_max+1).reshape(l_max+1,l_max+1)
# l_mat=l_mat.T
# m_mat=l_mat.T
# mask=m_stack_mask(l_max)
# m_mat.T[mask]
# l_h, m_h = hp.Alm.getlm(lmax=lmax)


####################################################################
#                       maps
####################################################################


@partial(jax.jit, static_argnums=(0))
def ring_pol_mask(nside, ring_i, mask):
    j = jnp.arange(4 * nside) + 1
    mask = mask.at[ring_i - 1, :].set(
        jnp.where(j <= 4 * ring_i, mask[ring_i - 1, :], False)
    )
    mask = mask.at[4 * nside - ring_i - 1, :].set(
        jnp.where(j <= 4 * ring_i, mask[4 * nside - ring_i - 1, :], False)
    )
    return mask


@partial(jax.jit, static_argnums=(0))
def stack_maps(nside, maps):
    mask = jnp.ones((4 * nside - 1, 4 * nside), dtype="bool")
    ring_pol_mask_i = partial(ring_pol_mask, nside)
    mask = jax.lax.fori_loop(1, nside, ring_pol_mask_i, mask)
    # for i in range(1, nside):
    #     mask = ring_pol_mask_i(i, mask)
    return mask


# hmap2=reshape_maps(nside,hmap)
# hmap_t=stack_maps(nside,hmap2)
# np.all(hmap==hmap_t)


@partial(jax.jit, static_argnums=(0))
def pol_ring_indxs(nside, ring_i, indxs):
    i = ring_i
    ii = jnp.arange(4 * nside) + 2 * i * (i + 1)
    ii = jnp.where(ii < 4 * (i + 1) + 2 * i * (i + 1), ii, -1)
    indxs = indxs.at[i, :].set(ii)

    s_pol_fact_i = 12 * nside**2 - 2 * i * (i + 1) - 4 * (i + 1)
    ii = jnp.arange(4 * nside) + s_pol_fact_i
    ii = jnp.where(ii < 4 * (i + 1) + s_pol_fact_i, ii, -1)

    indxs = indxs.at[4 * nside - i - 2, :].set(ii)
    return indxs


@partial(jax.jit, static_argnums=(0))
def eq_ring_indxs(nside, ring_i, indxs):
    i = ring_i
    eq_fact = 2 * nside * (nside - 1)
    eq_fact += (i - nside + 1) * 4 * nside
    indxs = indxs.at[i, :].set(jnp.arange(4 * nside) + eq_fact)  # +2*i*(i+1)
    return indxs


@partial(jax.jit, static_argnums=(0))
def reshape_maps(nside, maps):
    indxs = jnp.zeros((4 * nside - 1, 4 * nside), dtype="int")
    eq_ring_indxs_i = jax.jit(partial(eq_ring_indxs, nside))
    pol_ring_indxs_i = jax.jit(partial(pol_ring_indxs, nside))
    #     pol_ring_indxs_i=partial(pol_ring_indxs,nside)
    #     eq_ring_indxs_i=partial(eq_ring_indxs,nside)
    #     for i in range(nside-1):
    #         indxs=pol_ring_indxs_i(i,indxs)
    #     for i in range(nside-1,3*nside):
    #         indxs=eq_ring_indxs_i(i,indxs)
    indxs = jax.lax.fori_loop(0, nside - 1, pol_ring_indxs_i, indxs)
    indxs = jax.lax.fori_loop(nside - 1, 3 * nside, eq_ring_indxs_i, indxs)
    maps2 = maps[:, indxs]
    maps2 = jnp.where(
        indxs >= 0,
        maps2[
            :,
        ],
        0,
    )
    #     maps2=maps2.at[:,mask].set(0)
    return maps2, indxs

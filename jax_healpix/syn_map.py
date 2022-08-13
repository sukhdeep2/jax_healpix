from SPHT_jax import *
from utils import *
from reshape_utils import *
from jax.tree_util import Partial as partial


@partial(jax.jit, static_argnums=(0))
def get_alm_cov(n_tracers, cl_i):
    alm_cov = jnp.zeros((n_tracers, n_tracers))
    alm_cov = alm_cov.at[jnp.triu_indices(n_tracers, k=0)].set(cl_i)
    alm_cov = alm_cov + alm_cov.T
    alm_cov = alm_cov.at[jnp.diag_indices(n_tracers)].divide(2)
    return alm_cov


@partial(jax.jit, static_argnums=(0, 1, 2))
def syn_alm_l(n_maps, n_tracers, l_max, cl, m, rand_seed, l):
    rand_key = jax.random.PRNGKey(rand_seed + l)
    #     subkeys = jax.random.split(rand_key, num=l_max)
    #     rand_key=subkeys[l]
    cl_i = cl[:, l]
    alm_cov = get_alm_cov(n_tracers, cl_i)
    alm_rand = jax.random.multivariate_normal(
        rand_key, mean=jnp.zeros(n_tracers), cov=alm_cov, shape=(n_maps, l_max + 1)
    ).transpose(
        0, 2, 1
    )  # need to be complex
    new_key, subkey = jax.random.split(rand_key)
    phase = jax.random.uniform(
        subkey, shape=alm_rand.shape, minval=0, maxval=jnp.pi * 2
    )
    #     alm_sim=alm_sim.at[:,:,l,:].set(alm_rand*jnp.exp(1j*phase))
    #     return alm_sim
    return alm_rand * jnp.exp(1j * phase)


@partial(jax.jit, static_argnums=(0, 1, 2))
def syn_alm(n_maps, n_tracers, l_max, cl, rand_seed):
    m = jnp.arange(l_max + 1)
    alm_sim = jnp.zeros((n_maps, n_tracers, l_max + 1, l_max + 1), dtype=jnp.complex_)
    syn_alm_l_i = partial(syn_alm_l, n_maps, n_tracers, l_max, cl, m, rand_seed)
    syn_alm_l_i = jax.vmap(syn_alm_l_i, in_axes=(0,))
    #     alm_sim=jax.lax.map(syn_alm_l_i,jnp.arange(l_max + 1)) #slower than vmap
    alm_sim = syn_alm_l_i(jnp.arange(l_max + 1))

    alm_sim = alm_sim.transpose(1, 2, 0, 3)
    m = m[None, None, None, :]
    l = jnp.arange(l_max + 1)[None, None, :, None]
    alm_sim = alm_sim.at[:, :, :, :].set(jnp.where(l >= m, alm_sim, 0))
    return alm_sim


# @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def synfast(nside, l_max, spin_max, n_tracers, n_maps, cl, rand_seed):
    n_tracers_total = sum(n_tracers)
    alm_sim0 = syn_alm(n_maps, n_tracers_total, l_max, cl, rand_seed)

    alm_sim = {}
    ni = 0
    si = 0
    for s in range(0, spin_max + 1, 2):
        alm_sim[s] = alm_sim0[:, ni : n_tracers[si], :, :].reshape(
            n_maps * n_tracers[s], l_max + 1, l_max + 1
        )
        ni += n_tracers[si]
        si += 1
    del alm_sim0
    print("synfast got all alms")
    maps = alm2map(nside, l_max, spin_max, alm_sim)

    for s in maps.keys():
        maps[s] = maps[s].reshape(n_maps, n_tracers[s], 4 * nside - 1, 4 * nside)
    return maps

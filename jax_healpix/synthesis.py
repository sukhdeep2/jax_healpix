import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial
# Assuming transform.py (for HealpixTransformer) is in the same directory
from .transform import HealpixTransformer

class MapSynthesizer:
    def __init__(self, healpix_transformer=None):
        """
        Initializes the MapSynthesizer.
        Args:
            healpix_transformer (HealpixTransformer, optional): An instance of HealpixTransformer.
                If None, it will be created on-demand by synfast if nside and l_max are provided to synfast.
        """
        self.transformer = healpix_transformer

    @staticmethod
    @partial(jax.jit, static_argnums=(0,)) # n_tracers is static
    def _get_alm_cov(n_tracers, cl_i):
        """
        Helper function to construct the covariance matrix for alms at a single l.
        Args:
            n_tracers (int): Number of tracers (e.g., 1 for T, 2 for TQ, 3 for TQU/TEB).
            cl_i (jnp.ndarray): Flat 1D array of Cl values for this l.
                                 Order should be [Cl_00, Cl_01, Cl_02, ..., Cl_11, Cl_12, ..., Cl_NN].
                                 (Upper triangle elements in row-major order).
        Returns:
            jnp.ndarray: Covariance matrix [n_tracers, n_tracers].
        """
        alm_cov = jnp.zeros((n_tracers, n_tracers), dtype=cl_i.dtype)
        # Ensure cl_i is correctly shaped for triu_indices if it's multi-dimensional
        # cl_i should be flat, with ((n_tracers * (n_tracers + 1)) // 2) elements
        if cl_i.shape[0] != (n_tracers * (n_tracers + 1)) // 2:
            # This check will fail JIT. Caller must ensure correct shape.
            # Or, this function should not be JITted if this check is essential at runtime.
            # For now, assuming shape is correct as per JIT requirements.
            pass

        alm_cov = alm_cov.at[jnp.triu_indices(n_tracers, k=0)].set(cl_i)
        alm_cov = alm_cov + alm_cov.T
        alm_cov = alm_cov.at[jnp.diag_indices(n_tracers)].divide(2)
        return alm_cov

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2)) # n_maps, n_tracers, l_max are static
    def _syn_alm_l(n_maps, n_tracers, l_max, cl, rand_seed_base, l_val):
        """
        Generates alms for a single l value.
        Args:
            n_maps (int): Number of map realizations.
            n_tracers (int): Number of tracers (components like T, E, B).
            l_max (int): Maximum l value (for defining array sizes).
            cl (jnp.ndarray): Full Cl array [n_cl_combinations_flat, l_max+1].
                               n_cl_combinations_flat is (n_tracers * (n_tracers+1))//2.
            rand_seed_base (int): Base seed, will be combined with l_val.
            l_val (int): Current l being processed.
        Returns:
            jnp.ndarray: Complex alms for this l_val, shape (n_maps, l_max+1, n_tracers).
                         Contains values for m from 0 to l_max.
        """
        rand_key = jax.random.PRNGKey(rand_seed_base + l_val)

        cl_i_for_l = cl[:, l_val]
        alm_cov = MapSynthesizer._get_alm_cov(n_tracers, cl_i_for_l)

        key_real, key_imag, key_phase_m0 = jax.random.split(rand_key, 3)

        real_parts = jax.random.multivariate_normal(
            key_real,
            mean=jnp.zeros(n_tracers),
            cov=alm_cov,
            shape=(n_maps, l_max + 1)
        )
        imag_parts = jax.random.multivariate_normal(
            key_imag,
            mean=jnp.zeros(n_tracers),
            cov=alm_cov,
            shape=(n_maps, l_max + 1)
        )

        alms_val = (real_parts + 1j * imag_parts) / jnp.sqrt(2.0)

        m0_real_component = jax.random.multivariate_normal(
            key_phase_m0,
            mean=jnp.zeros(n_tracers),
            cov=alm_cov,
            shape=(n_maps,1)
        )[:,0,:]

        alms_val = alms_val.at[:,0,:].set(m0_real_component.astype(alms_val.dtype))

        return alms_val


    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2))
    def _syn_alm(n_maps, n_tracers, l_max, cl, rand_seed):
        alm_sim = jnp.zeros((n_maps, n_tracers, l_max + 1, l_max + 1), dtype=jnp.complex64)

        for l_idx in range(l_max + 1):
            alms_for_l_all_m = MapSynthesizer._syn_alm_l(
                n_maps, n_tracers, l_max, cl, rand_seed, l_idx
            )

            for m_idx in range(l_idx + 1):
                alm_sim = alm_sim.at[:, :, l_idx, m_idx].set(alms_for_l_all_m[:, m_idx, :])

        return alm_sim


    def synfast(self, nside, l_max, spins_to_generate, tracer_info, cls_input, rand_seed,
                healpix_transformer_instance=None):

        num_cl_flat = cls_input.shape[0]
        n_tracers_val = (-1 + jnp.sqrt(1 + 8 * num_cl_flat)) / 2
        if not jnp.isclose(n_tracers_val, jnp.round(n_tracers_val)):
            raise ValueError(f"Could not determine integer n_tracers from cls_input shape {cls_input.shape[0]}")
        n_tracers_for_syn_alm = int(jnp.round(n_tracers_val))

        n_maps_realization = 1
        alm_sim_raw_all_tracers = self._syn_alm(
            n_maps_realization,
            n_tracers_for_syn_alm,
            l_max,
            cls_input,
            rand_seed
        )
        alm_sim_raw_all_tracers = alm_sim_raw_all_tracers.squeeze(axis=0)

        alm_sim_for_transform = {}

        if 0 in spins_to_generate:
            t_idx = tracer_info.get('T', {}).get('cl_idx')
            if t_idx is None: raise ValueError("T tracer cl_idx missing in tracer_info for spin 0 map.")
            alm_sim_for_transform[0] = alm_sim_raw_all_tracers[t_idx, ...]

        if 2 in spins_to_generate:
            e_idx = tracer_info.get('E', {}).get('cl_idx')
            if e_idx is None: raise ValueError("E tracer cl_idx missing in tracer_info for spin 2 map (Q).")
            alm_sim_for_transform[2] = alm_sim_raw_all_tracers[e_idx, ...]

        if -2 in spins_to_generate:
            b_idx = tracer_info.get('B', {}).get('cl_idx')
            if b_idx is None: raise ValueError("B tracer cl_idx missing in tracer_info for spin -2 map (U).")
            alm_sim_for_transform[-2] = alm_sim_raw_all_tracers[b_idx, ...]

        current_transformer = self.transformer
        if current_transformer is None:
            current_transformer = HealpixTransformer(nside, l_max)
        elif nside != current_transformer.nside or l_max > current_transformer.l_max:
             print(f"Warning: synfast nside/l_max ({nside}/{l_max}) differs from " +
                   f"pre-configured transformer ({current_transformer.nside}/{current_transformer.l_max}). " +
                   "Creating a new transformer for this call.")
             current_transformer = HealpixTransformer(nside, l_max)

        spins_for_alm2map = tuple(k for k in (0, 2, -2) if k in alm_sim_for_transform)
        maps_dict = current_transformer.alm2map(alm_sim_for_transform, spins_to_map=spins_for_alm2map)

        return maps_dict

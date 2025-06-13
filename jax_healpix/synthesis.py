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

        # For m=0, a_l0 is real. For m>0, a_lm is complex.
        # Common practice: generate real N(0, C_l) for a_l0.
        # For m>0, generate two real N(0, C_l/2), combine into (N1+iN2).
        # Or, generate complex N(0, C_l).
        # JAX's multivariate_normal generates real values.
        # If alm_cov is C_l, then sqrt(C_l) gives stddev.
        # Let's generate real and imaginary parts separately for m>0.

        # Generate random numbers for m=0 (real part only)
        # alm_cov_m0 should be just C_l_TT for a_l0_T, or a matrix for multiple tracers if a_l0s are correlated.
        # For simplicity, assume _get_alm_cov provides the correct covariance for the components.
        # If a_l0 is real, its variance is C_l.
        # If a_lm (m>0) is (x+iy)/sqrt(2) and Var(x)=Var(y)=C_l, then Var(a_lm)=C_l.
        # The syn_map.py used `alm_rand * exp(1j*phase)` where alm_rand from multivariate_normal.
        # This implied alm_rand was a magnitude.

        # Let's follow the structure of generating a complex number directly from multivariate_normal
        # if we assume the covariance matrix `alm_cov` is for the complex components.
        # However, `multivariate_normal` in JAX produces real samples.
        # A standard approach for complex Gaussian a_lm ~ CN(0, C_l):
        # Real part ~ N(0, C_l/2), Imag part ~ N(0, C_l/2) for m > 0
        # Real part ~ N(0, C_l) for m = 0

        # The original code `alm_rand * jnp.exp(1j*phase)` suggests magnitudes and random phases.
        # This is for |a_lm| drawn from some distribution (e.g. Rayleigh if underlying Gaussian).
        # If alm_cov is the variance, then sqrt(alm_cov) is std.
        # Samples from multivariate_normal are correlated Gaussians.

        # Replicating `alm_rand_complex_flat = jax.random.multivariate_normal(...)`
        # then `alms_for_l_all_m = alm_rand_complex_flat * jnp.exp(1j * phase)`
        # implies `alm_rand_complex_flat` are magnitudes.
        # This is correct if `alm_cov` is for the magnitudes squared (power).
        # Standard multivariate normal gives samples, not magnitudes.

        # Let's assume `multivariate_normal` generates real components (x_i) for each tracer.
        # These need to be combined into complex a_lm.
        # For m > 0: a_lm = (x_real + i * x_imag) / sqrt(2)
        # For m = 0: a_l0 = x_real (this must be real)

        # Generate (n_maps, l_max+1, n_tracers) for real parts
        # Generate (n_maps, l_max+1, n_tracers) for imag parts (only for m>0)
        key_real, key_imag, key_phase_m0 = jax.random.split(rand_key, 3)

        real_parts = jax.random.multivariate_normal(
            key_real,
            mean=jnp.zeros(n_tracers),
            cov=alm_cov, # Assumes alm_cov is C_l or C_l/2 as appropriate
            shape=(n_maps, l_max + 1)
        )
        imag_parts = jax.random.multivariate_normal(
            key_imag,
            mean=jnp.zeros(n_tracers),
            cov=alm_cov, # Assumes alm_cov is C_l or C_l/2 as appropriate
            shape=(n_maps, l_max + 1)
        )

        # Combine into complex alms
        # For m > 0, alms = (real_parts + 1j * imag_parts) / sqrt(2) (if alm_cov = C_l)
        # For m = 0, alms = real_parts (if alm_cov = C_l for m=0 part)
        # This depends on how Cl diagonal (auto-power) and off-diagonal (cross-power)
        # are defined w.r.t real/imaginary parts.

        # A common simplification consistent with healpy's synalm:
        # Generate complex noise N(0,1), then scale by sqrt(C_l).
        # N(0,1) complex = (N_r(0,1) + i N_i(0,1))/sqrt(2)
        # So, a_lm = sqrt(C_l) * (N_r + i N_i)/sqrt(2)
        # This means Var(Re(a_lm)) = Var(Im(a_lm)) = C_l/2.
        # The multivariate_normal with cov=alm_cov (where alm_cov has C_l on diag)
        # generates samples x where E[x*xT] = alm_cov.
        # So, samples are already scaled by sqrt(C_l).

        # Let samples_real = real_parts, samples_imag = imag_parts from above.
        # These are already scaled by sqrt of variances in alm_cov.

        alms_val = (samples_real + 1j * samples_imag) / jnp.sqrt(2.0)

        # For m=0 (real alms): take real part of a N(0,C_l) sample.
        # If alm_cov is C_l, then samples_real[m=0] is N(0,C_l).
        # So, alms_val[:,0,:] should be real.
        # The above makes m=0 complex. Forcing real:
        m0_real_component = jax.random.multivariate_normal(
            key_phase_m0, # Use a different key for m=0 part
            mean=jnp.zeros(n_tracers),
            cov=alm_cov, # This cov should be for m=0 (real) alms.
            shape=(n_maps,1) # (n_maps, 1 for m=0, n_tracers)
        )[:,0,:] # Squeeze m-dim -> (n_maps, n_tracers)

        alms_val = alms_val.at[:,0,:].set(m0_real_component.astype(alms_val.dtype)) # Ensure m=0 is real
                                                                    # And ensure dtype matches

        return alms_val # Shape: (n_maps, l_max+1, n_tracers)


    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2)) # n_maps, n_tracers, l_max are static
    def _syn_alm(n_maps, n_tracers, l_max, cl, rand_seed):
        alm_sim = jnp.zeros((n_maps, n_tracers, l_max + 1, l_max + 1), dtype=jnp.complex64)

        # Vmap _syn_alm_l over l_val
        # _syn_alm_l args: n_maps, n_tracers, l_max, cl, rand_seed_base, l_val
        # cl shape: [n_cl_combinations_flat, l_max+1]
        # Output of _syn_alm_l: (n_maps, l_max+1_m_values, n_tracers)

        # To use vmap, _syn_alm_l needs to be defined to take l_val and return the slice for that l.
        # The cl argument needs to be handled carefully. Cl has an l_max+1 dim.
        # vmap can iterate over one axis of cl if needed.

        # Iterative approach (as in request, for clarity first)
        for l_idx in range(l_max + 1):
            alms_for_l_all_m = MapSynthesizer._syn_alm_l(
                n_maps, n_tracers, l_max, cl, rand_seed, l_idx
            ) # Result (n_maps, l_max+1 for m, n_tracers)

            for m_idx in range(l_idx + 1):
                # alm_sim is (N, T, L, M)
                # alms_for_l_all_m is (N, M_potential, T)
                # We need alm_sim[:, :, l_idx, m_idx] to be (N,T)
                # So, take alms_for_l_all_m[:, m_idx, :] which is (N,T)
                alm_sim = alm_sim.at[:, :, l_idx, m_idx].set(alms_for_l_all_m[:, m_idx, :])

        # Ensure a_lm = (-1)^m conj(a_l,-m) symmetry if needed by transformer.
        # Healpy alms are m>=0. Transformer typically expects this.
        # The loop m_idx up to l_idx populates only m>=0.
        return alm_sim


    def synfast(self, nside, l_max, spins_to_generate, tracer_info, cls_input, rand_seed,
                healpix_transformer_instance=None):
        """
        Generates synthetic maps.
        Args:
            nside (int): HEALPix nside for the output map.
            l_max (int): Maximum l for alm generation.
            spins_to_generate (tuple): Target map spins (e.g., (0,) for T; (0,2,-2) for T,Q,U).
            tracer_info (dict): Defines how tracers in Cl relate to output alms.
                                Example: {'T': {'cl_idx': 0}, 'E': {'cl_idx': 1}, 'B': {'cl_idx': 2}}
                                Assumes Cls are provided for T, E, B separately if they are all generated.
                                Number of tracers for _syn_alm derived from max cl_idx + 1.
            cls_input (jnp.ndarray): Power spectra array [n_cl_components_flat, l_max+1].
                                     n_cl_components_flat = (n_tracers * (n_tracers+1))//2.
                                     The order of Cls must match how _get_alm_cov expects them.
            rand_seed (int): Random seed.
            healpix_transformer_instance (HealpixTransformer, optional): Use if provided.
        Returns:
            dict: {spin_map_type: map_array}. E.g. {0: Tmap, 2: Qmap, -2: Umap}
        """

        # Determine n_tracers for _syn_alm from tracer_info (e.g., max index used)
        # This is simplified; a full setup would explicitly define n_tracers for syn_alm.
        n_tracers_for_syn_alm = 0
        if not tracer_info: # Should not happen if any spin is generated
            raise ValueError("tracer_info must be provided.")

        # Infer n_tracers from the number of unique cl_idx values mentioned.
        # Or, more simply, user should specify n_tracers for syn_alm.
        # For this version, let's assume n_tracers_for_syn_alm is derivable or fixed.
        # If cls_input is [C_TT, C_TE, C_TB, C_EE, C_EB, C_BB], n_tracers=3 (T,E,B).
        # The shape of cls_input implies n_tracers.
        # (N * (N+1))/2 = cls_input.shape[0]. Solve for N.
        num_cl_flat = cls_input.shape[0]
        # N^2 + N - 2*num_cl_flat = 0. N = (-1 + sqrt(1 + 8*num_cl_flat))/2
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
        # Shape: (n_tracers_for_syn_alm, l_max+1, l_max+1)

        alm_sim_for_transform = {}

        # Map from raw tracer alms to T, E, B alms based on tracer_info
        if 0 in spins_to_generate: # Temperature
            t_idx = tracer_info.get('T', {}).get('cl_idx')
            if t_idx is None: raise ValueError("T tracer cl_idx missing in tracer_info for spin 0 map.")
            alm_sim_for_transform[0] = alm_sim_raw_all_tracers[t_idx, ...]

        # For polarization, alm2map expects E-alm at key 2, B-alm at key -2
        if 2 in spins_to_generate: # Q map (implies E-alm needed)
            e_idx = tracer_info.get('E', {}).get('cl_idx')
            if e_idx is None: raise ValueError("E tracer cl_idx missing in tracer_info for spin 2 map (Q).")
            alm_sim_for_transform[2] = alm_sim_raw_all_tracers[e_idx, ...]

        if -2 in spins_to_generate: # U map (implies B-alm needed)
            b_idx = tracer_info.get('B', {}).get('cl_idx')
            if b_idx is None: raise ValueError("B tracer cl_idx missing in tracer_info for spin -2 map (U).")
            # If E was also generated and B uses same cl_idx as E (e.g. from auto-power), this is wrong.
            # Assume distinct cl_idx for E and B if both are present.
            alm_sim_for_transform[-2] = alm_sim_raw_all_tracers[b_idx, ...]

        current_transformer = self.transformer
        if current_transformer is None:
            current_transformer = HealpixTransformer(nside, l_max)
        elif nside != current_transformer.nside or l_max > current_transformer.l_max:
             print(f"Warning: synfast nside/l_max ({nside}/{l_max}) differs from " +
                   f"pre-configured transformer ({current_transformer.nside}/{current_transformer.l_max}). " +
                   "Creating a new transformer for this call.")
             current_transformer = HealpixTransformer(nside, l_max)

        spins_for_alm2map = tuple(k for k in (0, 2, -2) if k in alm_sim_for_transform) # Ensure correct order/presence
        maps_dict = current_transformer.alm2map(alm_sim_for_transform, spins_in_alm=spins_for_alm2map)

        return maps_dict

```

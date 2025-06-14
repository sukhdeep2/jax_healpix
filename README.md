# JAX-HEALPix: Spherical Harmonic Transforms with JAX

This library provides tools for performing Spherical Harmonic Transforms (SHTs) on HEALPix maps using JAX, enabling hardware acceleration (CPU/GPU/TPU) and automatic differentiation.

## Overview

The library is structured into the following main components:

*   **`jax_healpix.grid` (`HealpixGrid` class):**
    *   Handles HEALPix grid geometry, pixel-to-angle conversions, and ring property calculations.
*   **`jax_healpix.ylm` (`SphericalHarmonics` class):**
    *   Computes spin-weighted spherical harmonics ($sY_{lm}$) required for the transforms. Supports standard and numerically stable logarithmic recurrence relations. Includes utilities for caching and reshaping YLM arrays.
*   **`jax_healpix.transform` (`HealpixTransformer` class):**
    *   The core class for performing SHTs: `map2alm` (map to spherical harmonic coefficients) and `alm2map` (coefficients to map).
    *   Provides utilities for reshaping HEALPix maps (1D flat RING order <-> 2D ring-ordered format) and $a_{lm}$ coefficients (1D healpy-ordered <-> 2D (l,m) array).
    *   Includes a method to compute power spectra ($C_l$) from $a_{lm}$ coefficients.
*   **`jax_healpix.synthesis` (`MapSynthesizer` class):**
    *   Generates synthetic sky maps from input power spectra, utilizing the `HealpixTransformer`.

## Installation

Ensure you have Python 3.8+ and JAX installed according to the official JAX installation guide for your specific hardware (CPU/GPU/TPU). You will also need Healpy and NumPy:

```bash
pip install numpy healpy
# Follow JAX installation instructions: https://github.com/google/jax#installation
# e.g., pip install --upgrade "jax[cpu]"
```

The `jax_healpix` module should then be importable if it's in your Python path.

## Basic Usage Examples

```python
import jax
import jax.numpy as jnp
import numpy as np # For initial data
from jax_healpix.transform import HealpixTransformer
from jax_healpix.synthesis import MapSynthesizer

# JAX global precision (optional, tests use float64)
# from jax.config import config
# config.update('jax_enable_x64', True)

# --- Basic Setup ---
nside = 16
lmax = 2 * nside - 1 # Example lmax
key = jax.random.PRNGKey(42)

# Create a transformer instance
transformer = HealpixTransformer(nside=nside, l_max=lmax)

# --- Example 1: map2alm and alm2map (Temperature) ---

# Create a dummy 1D HEALPix map (RING ordered)
npix = transformer.grid.npix
dummy_map_1d_T = jnp.arange(npix, dtype=jnp.float64) / npix

# Convert 1D map to 2D ring format required by transformer
map_2d_T = transformer.convert_map_1d_to_2d(dummy_map_1d_T)

# Perform map2alm
# Input: dict {spin: map_data}. For Temperature, spin is 0.
# Output: dict {spin: alm_data_2D}, where alm_data is (lmax+1, lmax+1)
alm_output_T = transformer.map2alm(maps_dict={0: map_2d_T}, spins_in_map=(0,))
alm_T_2d = alm_output_T[0]
print(f"Shape of 2D T alms: {alm_T_2d.shape}")

# Perform alm2map
# Input: dict {spin: alm_data_2D}
# Output: dict {spin: map_data_2D}
map_reconv_2d_T_dict = transformer.alm2map(alm_dict={0: alm_T_2d}, spins_to_map=(0,))
map_reconv_2d_T = map_reconv_2d_T_dict[0]

# Convert 2D reconverted map back to 1D
map_reconv_1d_T = transformer.convert_map_2d_to_1d(map_reconv_2d_T)
print(f"Shape of reconverted 1D T map: {map_reconv_1d_T.shape}")
# Check if map_reconv_1d_T is close to dummy_map_1d_T (within precision)

# --- Example 2: Compute Power Spectrum (Cl) ---
# Using alm_T_2d from above
cl_TT = transformer.compute_cl(alm_T_2d) # Input can be single array or dict
print(f"Shape of C_l^TT: {cl_TT.shape}")

# --- Example 3: Synthesize a Map ---
# Define a simple Cl_TT for synthesis
input_cl_tt = jnp.ones(lmax + 1, dtype=jnp.float64) * 1e-5
input_cl_tt = input_cl_tt.at[0:2].set(0) # Zero monopole/dipole
cls_for_synthesis = input_cl_tt[None, :] # Expected shape [n_components, lmax+1]

synthesizer = MapSynthesizer(healpix_transformer=transformer) # Can reuse transformer

# For T-only synthesis:
# spins_to_generate=(0,): Generate alms for spin 0 (Temperature)
# tracer_counts_per_spin={0:1}: The first (and only) component in cls_for_synthesis is for spin 0
# The tracer_info argument name in synfast was used in tests.
# The example below uses tracer_counts_per_spin, ensure this matches the actual API if different.
# Assuming tracer_info is the correct argument based on last test setup:
tracer_info_synth = {'T': {'cl_idx': 0}}
synth_maps = synthesizer.synfast(
    nside=nside,
    l_max=lmax,
    spins_to_generate=(0,),
    tracer_info=tracer_info_synth, 
    cls_input=cls_for_synthesis, # Renamed from cls_dict in test to cls_input
    rand_seed=42 # Direct seed value
)
synth_map_T_2d = synth_maps[0]
print(f"Shape of synthesized Temperature map (2D): {synth_map_T_2d.shape}")
```

## Known Issues & Current Status

The library has undergone a significant refactoring into a class-based structure. A comprehensive test suite has been written (`tests/`). However, users should be aware of the following outstanding issues:

1.  **Critical File Corruption in `jax_healpix/transform.py` (Manual Fix Required):**
    *   An erroneous line `[end of jax_healpix/transform.py]` may exist at the very end of the `jax_healpix/transform.py` file due to a tooling issue during development. This causes a `SyntaxError`.
    *   **Action:** Please manually inspect and remove this line if present. *Self-correction attempt during development confirmed this marker was indeed present and then removed, but this should be double-checked.*

2.  **JAX JIT Compilation Issue for `_compute_cl_from_single_alm_set` (Manual Fix Recommended):**
    *   The method `HealpixTransformer._compute_cl_from_single_alm_set` (responsible for Cl calculations) has a problematic JAX JIT decorator when used as a `@staticmethod`. This can lead to `TypeError` during test collection or runtime (e.g. `TypeError: jit() missing 1 required positional argument: 'fun'` or `TypeError: argument is not a string: 0`).
    *   **Recommended Fix:** Convert this static method into a private, module-level helper function within `transform.py` (e.g., name it `_jit_compute_cl_logic(l_max, ...)`). This helper should be JIT-compiled (e.g., with `@jit(static_argnames='l_max')`). The class static method `_compute_cl_from_single_alm_set` should then become a simple wrapper calling this module-level JITted function. (This fix was attempted but failed due to tooling issues during development, specifically reliably modifying the decorator line).

3.  **Runtime Errors Identified by Tests:**
    *   The most recent `pytest tests/` run (after fixing file corruption and attempting JIT decorator changes) revealed several runtime errors that need debugging:
        *   **`tests/test_grid.py`:** `jax.errors.ConcretizationTypeError` in `HealpixGrid.pixel_to_angle_ring_colatitude_longitude` due to `int(tracer)` conversion of a JAX tracer (likely `npix_ring`) within JITted loops. This needs refactoring of loops or how dynamic values are handled in JIT.
        *   **`tests/test_synthesis.py`:** `TypeError` in `MapSynthesizer.synfast` due to incorrect keyword argument (`spins_in_alm` vs `spins_to_map`) when calling `transformer.alm2map`.
        *   **`tests/test_transform.py`:** `NameError` (e.g. `r_idx_0_based` typo) in `HealpixTransformer.convert_map_2d_to_1d`. Multiple `TypeError`s in `_map2alm_ring_contribution` from `jnp.zeros_like(None)` when optional spin components are not provided.
        *   **`tests/test_ylm.py`:** `AttributeError` in `SphericalHarmonics._sYLM_recur_log_internal` from calling `.astype` on a Python `int`. `AttributeError` in `SphericalHarmonics.process_ylm_stacking` test due to a test case passing a string, which is not caught early by a type check before `v_array.shape` is accessed.
    *   These errors prevent the full test suite from passing and indicate bugs in the library code that need to be addressed.

Due to these issues, the library is currently **experimental**. Once these are addressed, the tests should provide good validation of the core functionalities.

## Contributing

(Placeholder for future contribution guidelines)

## License

(Placeholder - e.g., MIT License, Apache 2.0)

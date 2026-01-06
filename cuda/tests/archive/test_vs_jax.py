#!/usr/bin/env python3
"""Compare CUDA SPHT results against JAX reference implementation

Usage:
    python test_vs_jax.py [nside] [l_max] [precision]

    nside: HEALPix resolution parameter (default: 256)
    l_max: Maximum multipole (default: 3*nside)
    precision: 'float64' or 'float32' (default: float64)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')

from spht_cuda import (set_phase1_method, get_phase1_method_name,
                       PHASE1_DFT, PHASE1_FFT_EQUATORIAL, PHASE1_BLUESTEIN)

# Global parameters (can be overridden via command line)
NSIDE = 256
L_MAX = None  # If None, defaults to 3*NSIDE
PRECISION = 'float64'  # 'float64' or 'float32'


def setup_precision(precision):
    """Configure JAX and CUDA precision settings."""
    import jax

    if precision == 'float64':
        jax.config.update("jax_enable_x64", True)
        np_dtype = np.float64
        np_complex = np.complex128
    else:
        jax.config.update("jax_enable_x64", False)
        np_dtype = np.float32
        np_complex = np.complex64

    return np_dtype, np_complex


def compare_map2alm(nside, l_max, precision):
    """Compare map2alm results between CUDA and JAX"""
    from spht_cuda import SPHTCuda
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm

    np_dtype, np_complex = setup_precision(precision)

    print(f"\n=== Comparing map2alm: nside={nside}, l_max={l_max}, precision={precision} ===\n")

    n_rings = 4 * nside - 1
    n_maps = 1  # Single map as array of size 1

    # Create a random map array [n_maps, n_rings, 4*nside]
    np.random.seed(42)
    map_array = np.random.randn(n_maps, n_rings, 4 * nside).astype(np_dtype)

    # JAX result
    map_jax = {0: jnp.array(map_array)}
    alm_jax_dict = jax_map2alm(nside, l_max, (0,), map_jax)
    alm_jax = np.array(alm_jax_dict[0][0])

    # CUDA results - test different precision and Phase 1 method combinations
    # Format: (name, storage, recurrence, phase1_method)
    if precision == 'float32':
        cuda_configs = [
            ('f32_f64_dft', 'float32', 'float64', PHASE1_DFT),
            ('f32_f32_dft', 'float32', 'float32', PHASE1_DFT),
            ('f32_f32_fft', 'float32', 'float32', PHASE1_FFT_EQUATORIAL),
            ('f32_f32_bluestein', 'float32', 'float32', PHASE1_BLUESTEIN),
        ]
    else:
        cuda_configs = [
            ('f64_f64_dft', 'float64', 'float64', PHASE1_DFT),
            ('f64_f64_fft', 'float64', 'float64', PHASE1_FFT_EQUATORIAL),
            ('f64_f64_bluestein', 'float64', 'float64', PHASE1_BLUESTEIN),
        ]

    results = {}
    for name, storage, recurrence, phase1_method in cuda_configs:
        set_phase1_method(phase1_method)
        spht_cuda = SPHTCuda(nside, l_max, version='v6',
                            storage_precision=storage,
                            recurrence_precision=recurrence)
        cuda_result = spht_cuda.map2alm({0: map_array}, spins=(0,))
        alm_cuda = cuda_result[0][0]

        # Convert to same dtype for comparison
        if alm_cuda.dtype != alm_jax.dtype:
            alm_cuda_cmp = alm_cuda.astype(alm_jax.dtype)
        else:
            alm_cuda_cmp = alm_cuda

        # Compute statistics
        diff = np.abs(alm_cuda_cmp - alm_jax)
        rel_diff = diff / (np.abs(alm_jax) + 1e-15)

        # Only consider valid (l,m) pairs where l >= m
        # For float32, limit to reliable range l <= nside to avoid Ylm instability
        l_reliable = nside if precision == 'float32' else l_max
        valid_mask = np.zeros_like(diff, dtype=bool)
        for l in range(min(l_max + 1, l_reliable + 1)):
            for m in range(l + 1):
                valid_mask[l, m] = True

        max_rel_diff = np.max(rel_diff[valid_mask])
        mean_rel_diff = np.mean(rel_diff[valid_mask])
        max_abs_diff = np.max(diff[valid_mask])

        results[name] = {
            'alm': alm_cuda,
            'max_rel': max_rel_diff,
            'mean_rel': mean_rel_diff,
            'max_abs': max_abs_diff,
            'all_finite': np.all(np.isfinite(alm_cuda)),
        }

        method_name = get_phase1_method_name()
        l_range_str = f" (l≤{l_reliable})" if precision == 'float32' else ""
        print(f"CUDA {name} ({storage} storage, {recurrence} recurrence, {method_name}):")
        print(f"  Max relative diff vs JAX{l_range_str}: {max_rel_diff:.6e}")
        print(f"  Mean relative diff{l_range_str}: {mean_rel_diff:.6e}")
        print(f"  All finite: {results[name]['all_finite']}")

    # Show sample values
    print(f"\nSample ALM values (first 4 l values):")
    print(f"  l,m |      JAX             ", end="")
    for name in results:
        print(f"|      CUDA {name}       ", end="")
    print()
    print("-" * (25 + 25 * len(results)))

    for l in range(min(4, l_max + 1)):
        for m in range(l + 1):
            jax_val = alm_jax[l, m]
            print(f"  {l},{m} | {jax_val.real:9.3e}+{jax_val.imag:9.3e}j", end="")
            for name in results:
                cuda_val = results[name]['alm'][l, m]
                print(f" | {cuda_val.real:9.3e}+{cuda_val.imag:9.3e}j", end="")
            print()

    # Check success criteria (looser for float32 due to Ylm instability at high l)
    # For float32, we only check that results are reasonable at low l
    threshold = 1e-4 if precision == 'float64' else 1e-1
    all_pass = True
    for name, r in results.items():
        passed = r['all_finite'] and r['max_rel'] < threshold
        if not passed:
            all_pass = False
        print(f"\n  {name}: {'PASS' if passed else 'FAIL'} (threshold: {threshold:.0e})")

    return all_pass


def compare_cell(nside, l_max, precision):
    """Compare angular power spectrum C_ell between CUDA and JAX"""
    from spht_cuda import SPHTCuda
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2cl as jax_alm2cl

    np_dtype, np_complex = setup_precision(precision)

    print(f"\n=== Comparing C_ell: nside={nside}, l_max={l_max}, precision={precision} ===\n")

    n_rings = 4 * nside - 1
    n_maps = 1  # Single map as array of size 1

    # Create a random map array [n_maps, n_rings, 4*nside]
    np.random.seed(42)
    map_array = np.random.randn(n_maps, n_rings, 4 * nside).astype(np_dtype)

    # JAX alm and C_ell
    map_jax = {0: jnp.array(map_array)}
    alm_jax_dict = jax_map2alm(nside, l_max, (0,), map_jax)
    alm_jax = np.array(alm_jax_dict[0][0])
    cell_jax = np.array(jax_alm2cl(l_max, jnp.array(alm_jax[None, :, :])))[0]

    # CUDA configs - test different Phase 1 methods
    # Format: (name, storage, recurrence, phase1_method)
    if precision == 'float32':
        cuda_configs = [
            ('f32_f64_dft', 'float32', 'float64', PHASE1_DFT),
            ('f32_f32_dft', 'float32', 'float32', PHASE1_DFT),
            ('f32_f32_bluestein', 'float32', 'float32', PHASE1_BLUESTEIN),
        ]
    else:
        cuda_configs = [
            ('f64_f64_dft', 'float64', 'float64', PHASE1_DFT),
            ('f64_f64_bluestein', 'float64', 'float64', PHASE1_BLUESTEIN),
        ]

    results = {}
    for name, storage, recurrence, phase1_method in cuda_configs:
        set_phase1_method(phase1_method)
        spht_cuda = SPHTCuda(nside, l_max, version='v6',
                            storage_precision=storage,
                            recurrence_precision=recurrence)
        cuda_result = spht_cuda.map2alm({0: map_array}, spins=(0,))
        alm_cuda = cuda_result[0][0]

        # Compute C_ell
        alm_for_cl = alm_cuda.astype(np.complex128) if alm_cuda.dtype == np.complex64 else alm_cuda
        cell_cuda = np.array(jax_alm2cl(l_max, jnp.array(alm_for_cl[None, :, :])))[0]

        diff = np.abs(cell_cuda - cell_jax)
        rel_diff = diff / (np.abs(cell_jax) + 1e-15)

        # Only consider C_ell up to l ~ nside (reliable regime for HEALPix)
        l_reliable = nside
        results[name] = {
            'cell': cell_cuda,
            'max_rel': np.max(rel_diff[:l_reliable + 1]),
            'mean_rel': np.mean(rel_diff[:l_reliable + 1]),
            'max_rel_all': np.max(rel_diff),  # For reference
        }

        print(f"CUDA {name}: Max rel diff (l≤{l_reliable}) = {results[name]['max_rel']:.6e}")

    # Show C_ell comparison at key ℓ values
    l_reliable = nside
    sample_ells = [0, 2, 10, nside//4, nside//2, nside]
    sample_ells = [l for l in sample_ells if l <= l_max]

    print(f"\nC_ell comparison (sample ℓ values, reliable range: ℓ≤{l_reliable}):")
    print(f"  l  |    JAX          ", end="")
    for name in results:
        print(f"|    CUDA {name}     ", end="")
    print()
    print("-" * (20 + 20 * len(results)))

    for l in sample_ells:
        print(f"{l:4d} | {cell_jax[l]:14.6e}", end="")
        for name in results:
            print(f" | {results[name]['cell'][l]:14.6e}", end="")
        print()

    # Check success (only for reliable ℓ range)
    threshold = 1e-4 if precision == 'float64' else 1e-2
    all_pass = True
    for name, r in results.items():
        passed = r['max_rel'] < threshold
        if not passed:
            all_pass = False

    return all_pass


def main(nside=None, l_max=None, precision=None):
    # Use global defaults if not specified
    if nside is None:
        nside = NSIDE
    if l_max is None:
        l_max = L_MAX if L_MAX is not None else 3 * nside
    if precision is None:
        precision = PRECISION

    print("=" * 70)
    print("CUDA SPHT vs JAX Reference Comparison")
    print("=" * 70)
    print(f"  nside     = {nside}")
    print(f"  l_max     = {l_max}")
    print(f"  precision = {precision}")
    print("  Phase 1 methods: DFT, FFT_EQUATORIAL, BLUESTEIN")
    if precision == 'float32':
        print("  CUDA modes: f32_f64 (recommended), f32_f32 (fastest)")
    print("=" * 70)

    map2alm_match = compare_map2alm(nside, l_max, precision)
    cell_match = compare_cell(nside, l_max, precision)

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  map2alm match: {'PASS' if map2alm_match else 'FAIL'}")
    print(f"  C_ell match: {'PASS' if cell_match else 'FAIL'}")
    print("=" * 70)

    return map2alm_match and cell_match


if __name__ == "__main__":
    # Parse command line arguments
    nside = int(sys.argv[1]) if len(sys.argv) > 1 else NSIDE
    l_max = int(sys.argv[2]) if len(sys.argv) > 2 else None
    precision = sys.argv[3] if len(sys.argv) > 3 else PRECISION

    success = main(nside, l_max, precision)
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""Compare CUDA alm2map_v6 results against JAX reference implementation

Usage:
    python test_alm2map_vs_jax.py [nside] [l_max] [precision]

    nside: HEALPix resolution parameter (default: 64)
    l_max: Maximum multipole (default: 3*nside)
    precision: 'float64' or 'float32' (default: float64)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')

# Global parameters (can be overridden via command line)
NSIDE = 64
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


def get_valid_pixel_mask(nside, n_rings):
    """Create mask for valid pixels per ring."""
    max_ring_pixels = 4 * nside
    valid_mask = np.zeros((n_rings, max_ring_pixels), dtype=bool)

    for r in range(n_rings):
        ring_i = r + 1
        if ring_i < nside:
            n_pix = 4 * ring_i
        elif ring_i > 3 * nside:
            n_pix = 4 * (4 * nside - ring_i)
        else:
            n_pix = 4 * nside
        valid_mask[r, :n_pix] = True

    return valid_mask


def compare_alm2map(nside, l_max, precision):
    """Compare alm2map results between CUDA and JAX"""
    from spht_cuda import SPHTCuda
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2map as jax_alm2map

    np_dtype, np_complex = setup_precision(precision)

    print(f"\n=== Comparing alm2map: nside={nside}, l_max={l_max}, precision={precision} ===\n")

    n_rings = 4 * nside - 1

    # Create a random map to get realistic alm coefficients
    np.random.seed(42)
    map_random = np.random.randn(1, n_rings, 4 * nside).astype(np_dtype)

    # Get alm using JAX map2alm (gold standard)
    map_jax = {0: jnp.array(map_random)}
    alm_jax_dict = jax_map2alm(nside, l_max, (0,), map_jax)
    alm_jax = np.array(alm_jax_dict[0])

    # JAX alm2map result (gold standard)
    alm_jax_in = {0: jnp.array(alm_jax)}
    map_jax_result = jax_alm2map(nside, l_max, (0,), alm_jax_in)
    map_jax_out = np.array(map_jax_result[0][0])

    # CUDA results - test recommended precision combinations
    # Note: f32 recurrence modes (f64_f32, f32_f32) are faster but lose accuracy at high l
    # They are not recommended for l_max > ~100-200
    if precision == 'float32':
        cuda_configs = [
            ('f32_f64', 'float32', 'float64'),  # f32 storage, f64 recurrence (recommended)
        ]
    else:
        cuda_configs = [
            ('f64_f64', 'float64', 'float64'),  # full f64 (highest accuracy, recommended)
        ]

    results = {}
    valid_mask = get_valid_pixel_mask(nside, n_rings)

    for name, storage, recurrence in cuda_configs:
        spht_cuda = SPHTCuda(nside, l_max, version='v6',
                            storage_precision=storage,
                            recurrence_precision=recurrence)

        # Convert alm to appropriate dtype for CUDA
        if storage == 'float32':
            alm_cuda_in = {0: alm_jax.astype(np.complex64)}
        else:
            alm_cuda_in = {0: alm_jax}

        cuda_result = spht_cuda.alm2map(alm_cuda_in, spins=(0,))
        map_cuda = cuda_result[0][0]

        # Convert to same dtype for comparison
        if map_cuda.dtype != map_jax_out.dtype:
            map_cuda_cmp = map_cuda.astype(map_jax_out.dtype)
        else:
            map_cuda_cmp = map_cuda

        # Compute statistics only on valid pixels
        diff = np.abs(map_cuda_cmp - map_jax_out)
        rel_diff = diff / (np.abs(map_jax_out) + 1e-15)

        max_rel_diff = np.max(rel_diff[valid_mask])
        mean_rel_diff = np.mean(rel_diff[valid_mask])
        max_abs_diff = np.max(diff[valid_mask])

        results[name] = {
            'map': map_cuda,
            'max_rel': max_rel_diff,
            'mean_rel': mean_rel_diff,
            'max_abs': max_abs_diff,
            'all_finite': np.all(np.isfinite(map_cuda)),
        }

        print(f"CUDA {name} ({storage} storage, {recurrence} recurrence):")
        print(f"  Max relative diff vs JAX: {max_rel_diff:.6e}")
        print(f"  Mean relative diff: {mean_rel_diff:.6e}")
        print(f"  Max absolute diff: {max_abs_diff:.6e}")
        print(f"  All finite: {results[name]['all_finite']}")

    # Show sample values
    print(f"\nSample map values (ring 0, ring {n_rings//2}, ring {n_rings-1}):")
    sample_rings = [0, n_rings//2, n_rings-1]

    for r_idx in sample_rings:
        ring_i = r_idx + 1
        if ring_i < nside:
            n_pix = min(4, 4 * ring_i)
        elif ring_i > 3 * nside:
            n_pix = min(4, 4 * (4 * nside - ring_i))
        else:
            n_pix = 4

        print(f"\n  Ring {r_idx} (first {n_pix} pixels):")
        print(f"    JAX:  {map_jax_out[r_idx, :n_pix]}")
        for name in results:
            print(f"    CUDA {name}: {results[name]['map'][r_idx, :n_pix]}")

    # Check success criteria - use appropriate thresholds per precision config
    # f32 recurrence has higher errors at high l due to limited mantissa bits
    thresholds = {
        'f64_f64': 1e-6,   # Highest accuracy
        'f64_f32': 2e-1,   # f32 recurrence has higher errors at high l
        'f32_f64': 1e-3,   # f32 storage, f64 recurrence
        'f32_f32': 2e-1,   # Lowest accuracy (both f32)
    }

    all_pass = True
    for name, r in results.items():
        threshold = thresholds.get(name, 1e-3)
        passed = r['all_finite'] and r['max_rel'] < threshold
        if not passed:
            all_pass = False
        print(f"\n  {name}: {'PASS' if passed else 'FAIL'} (threshold: {threshold:.0e})")

    return all_pass


def compare_roundtrip(nside, l_max, precision):
    """Compare roundtrip: map -> alm -> map"""
    from spht_cuda import SPHTCuda
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2map as jax_alm2map

    np_dtype, np_complex = setup_precision(precision)

    print(f"\n=== Roundtrip Test (map->alm->map): nside={nside}, l_max={l_max} ===\n")

    n_rings = 4 * nside - 1

    # Create random map
    np.random.seed(123)
    map_orig = np.random.randn(1, n_rings, 4 * nside).astype(np_dtype)

    # CUDA roundtrip
    spht_cuda = SPHTCuda(nside, l_max, version='v6',
                         storage_precision=precision,
                         recurrence_precision='float64')

    alm_cuda = spht_cuda.map2alm({0: map_orig}, spins=(0,))
    map_cuda_recon = spht_cuda.alm2map(alm_cuda, spins=(0,))

    # JAX roundtrip
    map_jax = {0: jnp.array(map_orig)}
    alm_jax = jax_map2alm(nside, l_max, (0,), map_jax)
    map_jax_recon = jax_alm2map(nside, l_max, (0,), alm_jax)

    # Compare reconstructed maps
    valid_mask = get_valid_pixel_mask(nside, n_rings)

    map_cuda_out = map_cuda_recon[0][0]
    map_jax_out = np.array(map_jax_recon[0][0])

    # Correlation between CUDA and JAX reconstructions
    corr_cuda_jax = np.corrcoef(
        map_cuda_out[valid_mask].flatten(),
        map_jax_out[valid_mask].flatten()
    )[0, 1]

    # Correlation with original (limited by l_max truncation)
    corr_cuda_orig = np.corrcoef(
        map_cuda_out[valid_mask].flatten(),
        map_orig[0][valid_mask].flatten()
    )[0, 1]

    corr_jax_orig = np.corrcoef(
        map_jax_out[valid_mask].flatten(),
        map_orig[0][valid_mask].flatten()
    )[0, 1]

    print(f"Correlation CUDA vs JAX reconstruction: {corr_cuda_jax:.10f}")
    print(f"Correlation CUDA reconstruction vs original: {corr_cuda_orig:.6f}")
    print(f"Correlation JAX reconstruction vs original: {corr_jax_orig:.6f}")

    # Check that CUDA and JAX give essentially the same result
    diff = np.abs(map_cuda_out - map_jax_out)
    rel_diff = diff / (np.abs(map_jax_out) + 1e-15)
    max_rel = np.max(rel_diff[valid_mask])

    print(f"Max relative diff CUDA vs JAX: {max_rel:.6e}")

    threshold = 1e-6 if precision == 'float64' else 1e-3
    passed = corr_cuda_jax > 0.9999 and max_rel < threshold
    print(f"\nRoundtrip test: {'PASS' if passed else 'FAIL'}")

    return passed


def main(nside=None, l_max=None, precision=None):
    # Use global defaults if not specified
    if nside is None:
        nside = NSIDE
    if l_max is None:
        l_max = L_MAX if L_MAX is not None else 3 * nside
    if precision is None:
        precision = PRECISION

    print("=" * 70)
    print("CUDA alm2map_v6 vs JAX Reference Comparison")
    print("=" * 70)
    print(f"  nside     = {nside}")
    print(f"  l_max     = {l_max}")
    print(f"  precision = {precision}")
    if precision == 'float32':
        print("  CUDA modes: f32_f64 (recommended), f32_f32 (fastest)")
    else:
        print("  CUDA modes: f64_f64 (highest accuracy), f64_f32")
    print("=" * 70)

    alm2map_match = compare_alm2map(nside, l_max, precision)
    roundtrip_match = compare_roundtrip(nside, l_max, precision)

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  alm2map match: {'PASS' if alm2map_match else 'FAIL'}")
    print(f"  Roundtrip match: {'PASS' if roundtrip_match else 'FAIL'}")
    print("=" * 70)

    return alm2map_match and roundtrip_match


if __name__ == "__main__":
    # Parse command line arguments
    nside = int(sys.argv[1]) if len(sys.argv) > 1 else NSIDE
    l_max = int(sys.argv[2]) if len(sys.argv) > 2 else None
    precision = sys.argv[3] if len(sys.argv) > 3 else PRECISION

    success = main(nside, l_max, precision)
    sys.exit(0 if success else 1)

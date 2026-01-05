#!/usr/bin/env python3
"""Compare CUDA SPHT results against JAX reference implementation

Usage:
    python test_vs_jax.py [nside] [l_max]

    nside: HEALPix resolution parameter (default: 8)
    l_max: Maximum multipole (default: 3*nside)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')

# Enable 64-bit precision in JAX (must be done before importing jax.numpy)
import jax
jax.config.update("jax_enable_x64", True)

from spht_cuda import SPHTCuda, compute_ylm_cuda, get_ring_geometry

# Import JAX SPHT reference
import jax.numpy as jnp
from YLM_jax_log import sYLM_recur_log
from SPHT_jax import map2alm as jax_map2alm, alm2cl as jax_alm2cl

# Global parameters (can be overridden via command line)
NSIDE = 256
L_MAX = None  # If None, defaults to 3*NSIDE


def compare_ylm(nside, l_max):
    """Compare YLM values against JAX reference"""
    n_rings = 2 * nside  # North hemisphere + equator

    print(f"\n=== Comparing YLM: nside={nside}, l_max={l_max}, n_rings={n_rings} ===\n")

    # Get CUDA geometry
    cuda_log_beta, cuda_beta_sign = get_ring_geometry(nside)

    # Compute CUDA YLM
    ylm_cuda = compute_ylm_cuda(l_max, cuda_log_beta[:n_rings])

    # Compute JAX YLM using log_beta directly
    log_beta_jax = jnp.array(cuda_log_beta[:n_rings])

    # Call JAX YLM computation
    ylm_jax_dict = sYLM_recur_log(l_max, (0,), log_beta_jax)
    ylm_jax = np.array(ylm_jax_dict[0])  # Shape [l_max+1, l_max+1, n_rings]

    # Compare
    diff = np.abs(ylm_cuda - ylm_jax)
    max_diff = np.max(diff)
    print(f"Max absolute difference: {max_diff:.6e}")

    # Show some values
    print("\nSample YLM values (first 5 l values, ring 0):")
    for l in range(min(5, l_max + 1)):
        for m in range(l + 1):
            cuda_val = ylm_cuda[l, m, 0]
            jax_val = ylm_jax[l, m, 0]
            d = abs(cuda_val - jax_val)
            status = "OK" if d < 1e-10 else f"DIFF={d:.2e}"
            print(f"  Y_{l},{m}(ring 0): CUDA={cuda_val:12.6e}  JAX={jax_val:12.6e}  {status}")

    # Show any large differences
    if max_diff > 1e-10:
        print("\nLarge differences (showing first 20):")
        large_diffs = np.argwhere(diff > 1e-10)
        for idx in large_diffs[:20]:
            l, m, r = idx
            print(f"  Y_{l},{m}(ring {r}): CUDA={ylm_cuda[l,m,r]:.6e}  JAX={ylm_jax[l,m,r]:.6e}  diff={diff[l,m,r]:.2e}")

    # 1e-4 is acceptable for numerical computations
    return max_diff < 1e-4


def compare_map2alm(nside, l_max):
    """Compare map2alm results between CUDA and JAX"""
    print(f"\n=== Comparing map2alm: nside={nside}, l_max={l_max} ===\n")

    # Initialize CUDA implementation
    spht_cuda = SPHTCuda(nside, l_max)

    n_rings = 4 * nside - 1

    # Create a random map
    np.random.seed(42)
    map_random = np.random.randn(1, n_rings, 4 * nside)

    # CUDA result
    cuda_result = spht_cuda.map2alm({0: map_random}, spins=(0,))
    alm_cuda = cuda_result[0][0]  # First map, shape [l_max+1, l_max+1]

    # JAX result - use actual JAX implementation
    map_jax = {0: jnp.array(map_random)}
    alm_jax_dict = jax_map2alm(nside, l_max, (0,), map_jax)
    alm_jax = np.array(alm_jax_dict[0][0])  # First map, shape [l_max+1, l_max+1]

    # Compare alm values
    print("ALM comparison (first 6 l values):")
    print(f"  l,m |      a_lm (CUDA)       |       a_lm (JAX)       |   Rel Diff")
    print("-" * 80)
    for l in range(min(6, l_max + 1)):
        for m in range(l + 1):
            cuda_val = alm_cuda[l, m]
            jax_val = alm_jax[l, m]
            rel = abs(cuda_val - jax_val) / (abs(jax_val) + 1e-15)
            print(f"  {l},{m} | {cuda_val.real:10.4e}+{cuda_val.imag:10.4e}j | "
                  f"{jax_val.real:10.4e}+{jax_val.imag:10.4e}j | {rel:10.2e}")

    # Compute overall statistics
    diff = np.abs(alm_cuda - alm_jax)
    rel_diff = diff / (np.abs(alm_jax) + 1e-15)

    # Only consider valid (l,m) pairs where l >= m
    valid_mask = np.zeros_like(diff, dtype=bool)
    for l in range(l_max + 1):
        for m in range(l + 1):
            valid_mask[l, m] = True

    max_rel_diff = np.max(rel_diff[valid_mask])
    mean_rel_diff = np.mean(rel_diff[valid_mask])
    max_abs_diff = np.max(diff[valid_mask])

    print(f"\nStatistics (all valid l,m):")
    print(f"  Max relative difference: {max_rel_diff:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff:.6e}")
    print(f"  Max absolute difference: {max_abs_diff:.6e}")

    # Success criteria
    all_finite = np.all(np.isfinite(alm_cuda)) and np.all(np.isfinite(alm_jax))
    close_match = max_rel_diff < 1e-4  # Within 0.01%

    print(f"\n  All finite: {all_finite}")
    print(f"  Close match (< 0.01% rel diff): {close_match}")

    return all_finite and close_match


def compare_cell(nside, l_max):
    """Compare angular power spectrum C_ell between CUDA and JAX"""
    print(f"\n=== Comparing C_ell: nside={nside}, l_max={l_max} ===\n")

    # Initialize CUDA implementation
    spht_cuda = SPHTCuda(nside, l_max)

    n_rings = 4 * nside - 1

    # Create a random map
    np.random.seed(42)
    map_random = np.random.randn(1, n_rings, 4 * nside)

    # CUDA alm
    cuda_result = spht_cuda.map2alm({0: map_random}, spins=(0,))
    alm_cuda = cuda_result[0][0]  # Shape [l_max+1, l_max+1]

    # JAX alm - use actual JAX implementation
    map_jax = {0: jnp.array(map_random)}
    alm_jax_dict = jax_map2alm(nside, l_max, (0,), map_jax)
    alm_jax = np.array(alm_jax_dict[0][0])  # Shape [l_max+1, l_max+1]

    # Compute C_ell using JAX function for both
    # The JAX alm2cl expects shape [n_maps, l_max+1, l_max+1]
    cell_jax = np.array(jax_alm2cl(l_max, jnp.array(alm_jax[None, :, :])))[0]

    # Compute C_ell for CUDA result using same formula
    cell_cuda = np.array(jax_alm2cl(l_max, jnp.array(alm_cuda[None, :, :])))[0]

    # Compare
    diff = np.abs(cell_cuda - cell_jax)
    rel_diff = diff / (np.abs(cell_jax) + 1e-15)

    print("C_ell comparison (CUDA vs JAX):")
    print(f"  l |    C_l (CUDA)   |    C_l (JAX)    |   Abs Diff   |  Rel Diff")
    print("-" * 75)
    for l in range(min(15, l_max + 1)):
        print(f"  {l:2d} | {cell_cuda[l]:14.6e} | {cell_jax[l]:14.6e} | {diff[l]:12.2e} | {rel_diff[l]:10.2e}")

    # Overall statistics
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)

    print(f"\nStatistics:")
    print(f"  Max relative difference: {max_rel_diff:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff:.6e}")
    print(f"  Max absolute difference: {np.max(diff):.6e}")

    # Check that values are close
    all_finite = np.all(np.isfinite(cell_cuda)) and np.all(np.isfinite(cell_jax))
    close_match = max_rel_diff < 1e-4  # Within 0.01%

    print(f"\n  All finite: {all_finite}")
    print(f"  Close match (< 0.01% rel diff): {close_match}")

    return all_finite and close_match


def main(nside=None, l_max=None):
    # Use global defaults if not specified
    if nside is None:
        nside = NSIDE
    if l_max is None:
        l_max = L_MAX if L_MAX is not None else 3 * nside

    print("=" * 60)
    print("CUDA SPHT vs JAX Reference Comparison")
    print(f"  nside = {nside}")
    print(f"  l_max = {l_max}")
    print("=" * 60)

    ylm_match = compare_ylm(nside, l_max)
    map2alm_match = compare_map2alm(nside, l_max)
    cell_match = compare_cell(nside, l_max)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  YLM match: {'PASS' if ylm_match else 'FAIL'}")
    print(f"  map2alm match: {'PASS' if map2alm_match else 'FAIL'}")
    print(f"  C_ell match: {'PASS' if cell_match else 'FAIL'}")
    print("=" * 60)

    return ylm_match and map2alm_match and cell_match


if __name__ == "__main__":
    # Parse command line arguments
    nside = int(sys.argv[1]) if len(sys.argv) > 1 else NSIDE
    l_max = int(sys.argv[2]) if len(sys.argv) > 2 else None

    success = main(nside, l_max)
    sys.exit(0 if success else 1)

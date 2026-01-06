#!/usr/bin/env python3
"""Compare CUDA spin-2 transforms (Q,U <-> E,B) against JAX reference implementation.

Usage:
    python test_spin2_vs_jax.py [nside] [l_max]

    nside: HEALPix resolution parameter (default: 32)
    l_max: Maximum multipole (default: 3*nside)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')

# Default parameters
NSIDE = 32
L_MAX = None  # If None, defaults to 3*NSIDE
PRECISION = 'float64'


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


def create_random_qu_maps(nside, n_maps, dtype):
    """Create random Q and U polarization maps."""
    n_rings = 4 * nside - 1
    max_pix = 4 * nside

    np.random.seed(42)
    Q_map = np.random.randn(n_maps, n_rings, max_pix).astype(dtype)
    U_map = np.random.randn(n_maps, n_rings, max_pix).astype(dtype)

    return Q_map, U_map


def cuda_map_to_jax_format(Q_map, U_map, dtype):
    """Convert CUDA map format to JAX spin-2 format.

    CUDA: Q_map [n_maps, n_rings, max_pix], U_map [n_maps, n_rings, max_pix] (real)
    JAX: maps[2] = Q (real), maps[-2] = U (real)
    """
    import jax.numpy as jnp

    # JAX expects Q in maps[2], U in maps[-2]
    maps_jax = {
        2: jnp.array(Q_map, dtype=dtype),
        -2: jnp.array(U_map, dtype=dtype)
    }
    return maps_jax


def jax_alm_to_cuda_format(alm_jax, dtype_complex):
    """Convert JAX alm format to CUDA spin-2 format.

    JAX: alm[2] = E_alm, alm[-2] = B_alm
    CUDA: alm[..., 0] = E, alm[..., 1] = B
    """
    E_alm = np.array(alm_jax[2], dtype=dtype_complex)
    B_alm = np.array(alm_jax[-2], dtype=dtype_complex)

    # Stack E and B
    alm_cuda = np.stack([E_alm, B_alm], axis=-1)
    return alm_cuda


def cuda_alm_to_jax_format(alm_cuda, dtype_complex):
    """Convert CUDA alm format to JAX spin-2 format.

    CUDA: alm[..., 0] = E, alm[..., 1] = B
    JAX: alm[2] = E_alm, alm[-2] = B_alm
    """
    import jax.numpy as jnp

    E_alm = alm_cuda[..., 0]
    B_alm = alm_cuda[..., 1]

    return {2: jnp.array(E_alm), -2: jnp.array(B_alm)}


def jax_map_to_cuda_format(maps_jax, dtype):
    """Convert JAX map format to CUDA spin-2 format.

    JAX: maps[2] = Q (real part is actual Q), maps[-2] = U (needs real() since Q, U are real)
    CUDA: map[..., 0] = Q, map[..., 1] = U

    After alm2map in JAX: maps[-2] *= 1j and maps[2] *= -1
    So maps[2].real = -Q and maps[-2].imag = U
    """
    Q_map = -np.real(np.array(maps_jax[2], dtype=dtype))  # Undo the -1 multiplication
    U_map = np.imag(np.array(maps_jax[-2]))  # Extract U from imaginary part
    U_map = U_map.astype(dtype)

    return Q_map, U_map


def compare_map2alm_spin2(nside, l_max, precision):
    """Compare spin-2 map2alm results between CUDA and JAX."""
    from spht_cuda import SPHTCuda
    import jax
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm

    np_dtype, np_complex = setup_precision(precision)

    print(f"\n=== Comparing spin-2 map2alm: nside={nside}, l_max={l_max} ===\n")

    n_rings = 4 * nside - 1
    n_maps = 1

    # Create random Q, U maps
    Q_map, U_map = create_random_qu_maps(nside, n_maps, np_dtype)

    # JAX map2alm
    maps_jax = cuda_map_to_jax_format(Q_map, U_map, np_dtype)
    alm_jax = jax_map2alm(nside, l_max, (2,), maps_jax)
    jax.block_until_ready(alm_jax)

    # Convert JAX alm to numpy for comparison
    E_jax = np.array(alm_jax[2])
    B_jax = np.array(alm_jax[-2])

    # CUDA map2alm
    spht_cuda = SPHTCuda(nside, l_max, version='v6',
                         storage_precision=precision,
                         recurrence_precision='float64')

    # Create CUDA input format: [n_maps, n_rings, max_pix, 2]
    maps_cuda_in = np.stack([Q_map, U_map], axis=-1)
    alm_cuda = spht_cuda.map2alm({2: maps_cuda_in}, spins=(2,))

    E_cuda = alm_cuda[2][..., 0]
    B_cuda = alm_cuda[2][..., 1]

    # Compare E-mode
    diff_E = np.abs(E_cuda - E_jax)
    rel_diff_E = diff_E / (np.abs(E_jax) + 1e-15)
    max_rel_E = np.max(rel_diff_E)
    mean_rel_E = np.mean(rel_diff_E)

    # Compare B-mode
    diff_B = np.abs(B_cuda - B_jax)
    rel_diff_B = diff_B / (np.abs(B_jax) + 1e-15)
    max_rel_B = np.max(rel_diff_B)
    mean_rel_B = np.mean(rel_diff_B)

    print(f"E-mode comparison:")
    print(f"  Max relative diff: {max_rel_E:.6e}")
    print(f"  Mean relative diff: {mean_rel_E:.6e}")

    print(f"\nB-mode comparison:")
    print(f"  Max relative diff: {max_rel_B:.6e}")
    print(f"  Mean relative diff: {mean_rel_B:.6e}")

    # Show sample values
    print(f"\nSample E-mode alm values (l=2, first 3 m values):")
    print(f"  JAX:  {E_jax[0, 2, :3]}")
    print(f"  CUDA: {E_cuda[0, 2, :3]}")

    print(f"\nSample B-mode alm values (l=2, first 3 m values):")
    print(f"  JAX:  {B_jax[0, 2, :3]}")
    print(f"  CUDA: {B_cuda[0, 2, :3]}")

    # Check pass/fail
    threshold = 1e-6 if precision == 'float64' else 1e-3
    passed_E = max_rel_E < threshold
    passed_B = max_rel_B < threshold

    print(f"\n  E-mode: {'PASS' if passed_E else 'FAIL'} (threshold: {threshold:.0e})")
    print(f"  B-mode: {'PASS' if passed_B else 'FAIL'} (threshold: {threshold:.0e})")

    return passed_E and passed_B, (E_cuda, B_cuda), (E_jax, B_jax)


def compare_alm2map_spin2(nside, l_max, precision):
    """Compare spin-2 alm2map results between CUDA and JAX."""
    from spht_cuda import SPHTCuda
    import jax
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2map as jax_alm2map

    np_dtype, np_complex = setup_precision(precision)

    print(f"\n=== Comparing spin-2 alm2map: nside={nside}, l_max={l_max} ===\n")

    n_rings = 4 * nside - 1
    n_maps = 1

    # Create random Q, U maps to get realistic E, B alm
    Q_map, U_map = create_random_qu_maps(nside, n_maps, np_dtype)

    # Get alm using JAX map2alm
    maps_jax = cuda_map_to_jax_format(Q_map, U_map, np_dtype)
    alm_jax = jax_map2alm(nside, l_max, (2,), maps_jax)
    jax.block_until_ready(alm_jax)

    # JAX alm2map
    maps_jax_result = jax_alm2map(nside, l_max, (2,), alm_jax)
    jax.block_until_ready(maps_jax_result)

    # Convert JAX maps to Q, U
    Q_jax, U_jax = jax_map_to_cuda_format(maps_jax_result, np_dtype)

    # CUDA alm2map
    spht_cuda = SPHTCuda(nside, l_max, version='v6',
                         storage_precision=precision,
                         recurrence_precision='float64')

    # Convert JAX alm to CUDA format
    alm_cuda_in = jax_alm_to_cuda_format(alm_jax, np_complex)
    maps_cuda = spht_cuda.alm2map({2: alm_cuda_in}, spins=(2,))

    Q_cuda = maps_cuda[2][..., 0]
    U_cuda = maps_cuda[2][..., 1]

    # Get valid pixel mask
    valid_mask = get_valid_pixel_mask(nside, n_rings)

    # Compare Q maps
    diff_Q = np.abs(Q_cuda[0] - Q_jax[0])
    rel_diff_Q = diff_Q / (np.abs(Q_jax[0]) + 1e-15)
    max_rel_Q = np.max(rel_diff_Q[valid_mask])
    mean_rel_Q = np.mean(rel_diff_Q[valid_mask])

    # Compare U maps
    diff_U = np.abs(U_cuda[0] - U_jax[0])
    rel_diff_U = diff_U / (np.abs(U_jax[0]) + 1e-15)
    max_rel_U = np.max(rel_diff_U[valid_mask])
    mean_rel_U = np.mean(rel_diff_U[valid_mask])

    print(f"Q map comparison:")
    print(f"  Max relative diff: {max_rel_Q:.6e}")
    print(f"  Mean relative diff: {mean_rel_Q:.6e}")

    print(f"\nU map comparison:")
    print(f"  Max relative diff: {max_rel_U:.6e}")
    print(f"  Mean relative diff: {mean_rel_U:.6e}")

    # Show sample values
    mid_ring = n_rings // 2
    print(f"\nSample Q map values (ring {mid_ring}, first 4 pixels):")
    print(f"  JAX:  {Q_jax[0, mid_ring, :4]}")
    print(f"  CUDA: {Q_cuda[0, mid_ring, :4]}")

    print(f"\nSample U map values (ring {mid_ring}, first 4 pixels):")
    print(f"  JAX:  {U_jax[0, mid_ring, :4]}")
    print(f"  CUDA: {U_cuda[0, mid_ring, :4]}")

    # Check pass/fail
    threshold = 1e-6 if precision == 'float64' else 1e-3
    passed_Q = max_rel_Q < threshold
    passed_U = max_rel_U < threshold

    print(f"\n  Q map: {'PASS' if passed_Q else 'FAIL'} (threshold: {threshold:.0e})")
    print(f"  U map: {'PASS' if passed_U else 'FAIL'} (threshold: {threshold:.0e})")

    return passed_Q and passed_U


def compare_roundtrip_spin2(nside, l_max, precision):
    """Compare roundtrip: (Q,U) -> (E,B) -> (Q,U) between CUDA and JAX."""
    from spht_cuda import SPHTCuda
    import jax
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2map as jax_alm2map

    np_dtype, np_complex = setup_precision(precision)

    print(f"\n=== Roundtrip Test spin-2 (Q,U -> E,B -> Q,U): nside={nside}, l_max={l_max} ===\n")

    n_rings = 4 * nside - 1
    n_maps = 1

    # Create random Q, U maps
    np.random.seed(123)
    Q_orig = np.random.randn(n_maps, n_rings, 4 * nside).astype(np_dtype)
    U_orig = np.random.randn(n_maps, n_rings, 4 * nside).astype(np_dtype)

    # CUDA roundtrip
    spht_cuda = SPHTCuda(nside, l_max, version='v6',
                         storage_precision=precision,
                         recurrence_precision='float64')

    maps_cuda_in = np.stack([Q_orig, U_orig], axis=-1)
    alm_cuda = spht_cuda.map2alm({2: maps_cuda_in}, spins=(2,))
    maps_cuda_out = spht_cuda.alm2map(alm_cuda, spins=(2,))

    Q_cuda_recon = maps_cuda_out[2][..., 0]
    U_cuda_recon = maps_cuda_out[2][..., 1]

    # JAX roundtrip
    maps_jax_in = cuda_map_to_jax_format(Q_orig, U_orig, np_dtype)
    alm_jax = jax_map2alm(nside, l_max, (2,), maps_jax_in)
    maps_jax_out = jax_alm2map(nside, l_max, (2,), alm_jax)
    jax.block_until_ready(maps_jax_out)

    Q_jax_recon, U_jax_recon = jax_map_to_cuda_format(maps_jax_out, np_dtype)

    # Get valid pixel mask
    valid_mask = get_valid_pixel_mask(nside, n_rings)

    # Correlation between CUDA and JAX reconstructions (Q)
    corr_Q_cuda_jax = np.corrcoef(
        Q_cuda_recon[0][valid_mask].flatten(),
        Q_jax_recon[0][valid_mask].flatten()
    )[0, 1]

    # Correlation between CUDA and JAX reconstructions (U)
    corr_U_cuda_jax = np.corrcoef(
        U_cuda_recon[0][valid_mask].flatten(),
        U_jax_recon[0][valid_mask].flatten()
    )[0, 1]

    # Correlation with original (limited by l_max truncation)
    corr_Q_cuda_orig = np.corrcoef(
        Q_cuda_recon[0][valid_mask].flatten(),
        Q_orig[0][valid_mask].flatten()
    )[0, 1]

    corr_U_cuda_orig = np.corrcoef(
        U_cuda_recon[0][valid_mask].flatten(),
        U_orig[0][valid_mask].flatten()
    )[0, 1]

    print(f"Correlation CUDA vs JAX (Q reconstruction): {corr_Q_cuda_jax:.10f}")
    print(f"Correlation CUDA vs JAX (U reconstruction): {corr_U_cuda_jax:.10f}")
    print(f"Correlation CUDA reconstruction vs original Q: {corr_Q_cuda_orig:.6f}")
    print(f"Correlation CUDA reconstruction vs original U: {corr_U_cuda_orig:.6f}")

    # Check that CUDA and JAX give essentially the same result
    diff_Q = np.abs(Q_cuda_recon - Q_jax_recon)
    rel_diff_Q = diff_Q / (np.abs(Q_jax_recon) + 1e-15)
    max_rel_Q = np.max(rel_diff_Q[0][valid_mask])

    diff_U = np.abs(U_cuda_recon - U_jax_recon)
    rel_diff_U = diff_U / (np.abs(U_jax_recon) + 1e-15)
    max_rel_U = np.max(rel_diff_U[0][valid_mask])

    print(f"\nMax relative diff CUDA vs JAX (Q): {max_rel_Q:.6e}")
    print(f"Max relative diff CUDA vs JAX (U): {max_rel_U:.6e}")

    threshold = 1e-6 if precision == 'float64' else 1e-3
    passed = (corr_Q_cuda_jax > 0.9999 and corr_U_cuda_jax > 0.9999 and
              max_rel_Q < threshold and max_rel_U < threshold)
    print(f"\nRoundtrip test: {'PASS' if passed else 'FAIL'}")

    return passed


def main(nside=None, l_max=None, precision=None):
    """Run all spin-2 comparison tests."""
    # Use global defaults if not specified
    if nside is None:
        nside = NSIDE
    if l_max is None:
        l_max = L_MAX if L_MAX is not None else 3 * nside
    if precision is None:
        precision = PRECISION

    print("=" * 70)
    print("CUDA Spin-2 Transforms vs JAX Reference Comparison")
    print("=" * 70)
    print(f"  nside     = {nside}")
    print(f"  l_max     = {l_max}")
    print(f"  precision = {precision}")
    print("=" * 70)

    map2alm_pass, _, _ = compare_map2alm_spin2(nside, l_max, precision)
    alm2map_pass = compare_alm2map_spin2(nside, l_max, precision)
    roundtrip_pass = compare_roundtrip_spin2(nside, l_max, precision)

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  map2alm spin-2: {'PASS' if map2alm_pass else 'FAIL'}")
    print(f"  alm2map spin-2: {'PASS' if alm2map_pass else 'FAIL'}")
    print(f"  Roundtrip spin-2: {'PASS' if roundtrip_pass else 'FAIL'}")
    print("=" * 70)

    return map2alm_pass and alm2map_pass and roundtrip_pass


if __name__ == "__main__":
    # Parse command line arguments
    nside = int(sys.argv[1]) if len(sys.argv) > 1 else NSIDE
    l_max = int(sys.argv[2]) if len(sys.argv) > 2 else None

    success = main(nside, l_max)
    sys.exit(0 if success else 1)

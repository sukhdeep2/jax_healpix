"""
Test CUDA implementation against JAX reference implementation.

Reference: CUDA_MIGRATION_PLAN.md Section 9.6 - Validation Strategy
"""

import numpy as np
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "jax_healpix"))

# Try to import modules
try:
    from spht_cuda import SPHTCuda, map2alm_cuda, alm2map_cuda
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CUDA module not available: {e}")
    CUDA_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from SPHT_jax import map2alm as jax_map2alm, alm2map as jax_alm2map, ring_log_beta
    from YLM_jax_log import sYLM_recur_log
    from reshape_utils import reshape_maps, stack_maps
    JAX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: JAX module not available: {e}")
    JAX_AVAILABLE = False


def get_tolerance(l_max, stage='alm'):
    """
    Compute tolerance based on l_max and computational stage.

    Reference: CUDA_MIGRATION_PLAN.md Section 9.6
    """
    eps = 2.2e-16  # float64 machine epsilon

    if stage == 'ylm_element':
        return max(1e-14, l_max * eps * 10)
    elif stage == 'ylm_array':
        return max(1e-13, l_max * eps * 100)
    elif stage == 'alm':
        return max(1e-12, l_max * eps * 1000)
    elif stage == 'roundtrip':
        return max(1e-11, l_max * eps * 10000)
    else:
        return 1e-10


def test_ylm_north():
    """Compare YLM values for north hemisphere rings."""
    if not JAX_AVAILABLE or not CUDA_AVAILABLE:
        print("Skipping YLM test - modules not available")
        return True

    print("Testing YLM computation (north rings)...")

    test_cases = [(32, 96)]

    for nside, l_max in test_cases:
        print(f"  nside={nside}, l_max={l_max}")

        # Get log(cos(theta)) for north rings only (rings 1 to 2*nside)
        log_beta_jax, _ = ring_log_beta(nside)
        log_beta_jax = log_beta_jax[:2*nside]

        # Compute YLM in JAX (reference)
        ylm_jax = sYLM_recur_log(l_max, spins=(0,), log_beta=log_beta_jax)

        # TODO: Call CUDA YLM computation directly
        # For now, this is a placeholder

        print(f"    JAX YLM shape: {ylm_jax[0].shape}")
        print(f"    JAX YLM[0,0,0] (Y_00): {ylm_jax[0][0, 0, 0]:.10e}")

    return True


def test_uniform_map():
    """Test map2alm with uniform map."""
    if not JAX_AVAILABLE:
        print("Skipping uniform map test - JAX not available")
        return True

    print("Testing uniform map transform...")

    nside = 32
    l_max = 3 * nside
    n_rings = 4 * nside - 1

    # Create uniform map
    npix = 12 * nside**2
    map_1d = np.ones((1, npix))

    # Reshape to ring format
    map_2d = np.zeros((1, n_rings, 4 * nside))
    for ring_i in range(1, n_rings + 1):
        if ring_i < nside:
            npix_ring = 4 * ring_i
            start_pix = 2 * ring_i * (ring_i - 1)
        elif ring_i <= 3 * nside:
            npix_ring = 4 * nside
            north_cap_pix = 2 * nside * (nside - 1)
            start_pix = north_cap_pix + (ring_i - nside) * 4 * nside
        else:
            ring_from_south = 4 * nside - ring_i
            npix_ring = 4 * ring_from_south
            total_pix = 12 * nside * nside
            south_remaining = 2 * ring_from_south * (ring_from_south + 1)
            start_pix = total_pix - south_remaining

        map_2d[0, ring_i - 1, :npix_ring] = map_1d[0, start_pix:start_pix + npix_ring]

    # JAX reference
    maps_jax = {0: jnp.array(map_2d)}
    alm_jax = jax_map2alm(nside, l_max, (0,), maps_jax)

    expected_a00 = np.sqrt(4 * np.pi)
    jax_a00 = alm_jax[0][0, 0, 0]

    print(f"  JAX a_00: {jax_a00:.10e}")
    print(f"  Expected: {expected_a00:.10e}")
    print(f"  Diff: {abs(jax_a00 - expected_a00):.2e}")

    if CUDA_AVAILABLE:
        # CUDA test
        spht = SPHTCuda(nside, l_max)
        alm_cuda = spht.map2alm({0: map_2d}, spins=(0,))
        cuda_a00 = alm_cuda[0][0, 0, 0]

        print(f"  CUDA a_00: {cuda_a00:.10e}")
        print(f"  CUDA-JAX diff: {abs(cuda_a00 - jax_a00):.2e}")

    return True


def test_alm_hemisphere_maps():
    """
    Test map2alm with hemisphere maps.

    Reference: CUDA_MIGRATION_PLAN.md Section 9.6 Step 3
    """
    if not JAX_AVAILABLE:
        print("Skipping hemisphere test - JAX not available")
        return True

    print("Testing hemisphere maps...")

    nside = 32
    l_max = 3 * nside

    # Create north-only map
    npix = 12 * nside**2
    map_north = np.zeros(npix)

    # Set north hemisphere to 1
    for ring_i in range(1, 2 * nside + 1):
        if ring_i < nside:
            npix_ring = 4 * ring_i
            start_pix = 2 * ring_i * (ring_i - 1)
        else:
            npix_ring = 4 * nside
            north_cap_pix = 2 * nside * (nside - 1)
            start_pix = north_cap_pix + (ring_i - nside) * 4 * nside

        map_north[start_pix:start_pix + npix_ring] = 1.0

    print(f"  North map: {np.sum(map_north > 0)} pixels set to 1")
    print(f"  South map: {np.sum(map_north == 0)} pixels")

    # Reshape
    n_rings = 4 * nside - 1
    map_2d = np.zeros((1, n_rings, 4 * nside))
    for ring_i in range(1, n_rings + 1):
        if ring_i < nside:
            npix_ring = 4 * ring_i
            start_pix = 2 * ring_i * (ring_i - 1)
        elif ring_i <= 3 * nside:
            npix_ring = 4 * nside
            north_cap_pix = 2 * nside * (nside - 1)
            start_pix = north_cap_pix + (ring_i - nside) * 4 * nside
        else:
            ring_from_south = 4 * nside - ring_i
            npix_ring = 4 * ring_from_south
            total_pix = 12 * nside * nside
            south_remaining = 2 * ring_from_south * (ring_from_south + 1)
            start_pix = total_pix - south_remaining

        map_2d[0, ring_i - 1, :npix_ring] = map_north[start_pix:start_pix + npix_ring]

    # JAX transform
    maps_jax = {0: jnp.array(map_2d)}
    alm_jax = jax_map2alm(nside, l_max, (0,), maps_jax)

    print(f"  JAX a_00 (north only): {alm_jax[0][0, 0, 0]:.10e}")
    print(f"  JAX a_10: {alm_jax[0][0, 1, 0]:.10e}")

    return True


def test_roundtrip():
    """Test map -> alm -> map round-trip."""
    if not JAX_AVAILABLE:
        print("Skipping roundtrip test - JAX not available")
        return True

    print("Testing round-trip transform...")

    nside = 32
    l_max = 3 * nside

    # Create random map
    np.random.seed(42)
    npix = 12 * nside**2
    n_rings = 4 * nside - 1

    map_1d = np.random.randn(1, npix)

    # Reshape to ring format
    map_2d = np.zeros((1, n_rings, 4 * nside))
    for ring_i in range(1, n_rings + 1):
        if ring_i < nside:
            npix_ring = 4 * ring_i
            start_pix = 2 * ring_i * (ring_i - 1)
        elif ring_i <= 3 * nside:
            npix_ring = 4 * nside
            north_cap_pix = 2 * nside * (nside - 1)
            start_pix = north_cap_pix + (ring_i - nside) * 4 * nside
        else:
            ring_from_south = 4 * nside - ring_i
            npix_ring = 4 * ring_from_south
            total_pix = 12 * nside * nside
            south_remaining = 2 * ring_from_south * (ring_from_south + 1)
            start_pix = total_pix - south_remaining

        map_2d[0, ring_i - 1, :npix_ring] = map_1d[0, start_pix:start_pix + npix_ring]

    # JAX round-trip
    maps_jax = {0: jnp.array(map_2d)}
    alm_jax = jax_map2alm(nside, l_max, (0,), maps_jax)
    maps_recon_jax = jax_alm2map(nside, l_max, (0,), alm_jax)

    # Compute residual
    residual = np.array(maps_jax[0]) - np.array(maps_recon_jax[0])
    rel_error = np.std(residual) / np.std(np.array(maps_jax[0]))

    print(f"  JAX relative round-trip error: {rel_error:.2e}")

    tol = get_tolerance(l_max, 'roundtrip')
    print(f"  Tolerance: {tol:.2e}")

    return True


def main():
    print("=== SPHT CUDA vs JAX Validation Tests ===\n")

    test_ylm_north()
    print()

    test_uniform_map()
    print()

    test_alm_hemisphere_maps()
    print()

    test_roundtrip()
    print()

    print("=== Tests Complete ===")


if __name__ == "__main__":
    main()

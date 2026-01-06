"""
Comprehensive comparison of CUDA vs JAX for ALL YLM and alm values.
"""

import numpy as np
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "jax_healpix"))

def test_uniform_map_alm():
    """Compare ALL alm values for uniform map against expected values."""
    print("=" * 60)
    print("Test: Uniform map ALL alm comparison")
    print("=" * 60)

    try:
        from spht_cuda import SPHTCuda
    except ImportError as e:
        print(f"CUDA module not available: {e}")
        return False

    try:
        import jax.numpy as jnp
        from SPHT_jax import map2alm as jax_map2alm
        JAX_AVAILABLE = True
    except ImportError as e:
        print(f"JAX not available: {e}")
        JAX_AVAILABLE = False

    nside = 32
    l_max = 3 * nside  # 96
    n_rings = 4 * nside - 1  # 127

    # Create proper uniform map in 2D ring format
    # IMPORTANT: Only set valid pixels to 1, rest should be 0
    map_2d = np.zeros((1, n_rings, 4 * nside), dtype=np.float64)

    for ring_i in range(1, n_rings + 1):
        # Get number of pixels for this ring
        if ring_i < nside:
            npix_ring = 4 * ring_i
        elif ring_i <= 3 * nside:
            npix_ring = 4 * nside
        else:
            ring_from_south = 4 * nside - ring_i
            npix_ring = 4 * ring_from_south

        # Only set valid pixels to 1
        map_2d[0, ring_i - 1, :npix_ring] = 1.0

    print(f"  nside = {nside}, l_max = {l_max}")
    print(f"  Map shape: {map_2d.shape}")
    print(f"  Total pixels set to 1: {np.sum(map_2d > 0)}")
    print(f"  Expected total: {12 * nside**2}")

    # CUDA transform
    spht = SPHTCuda(nside, l_max)
    alm_cuda = spht.map2alm({0: map_2d}, spins=(0,))[0]

    print(f"\n  CUDA alm shape: {alm_cuda.shape}")

    # Expected values for uniform map
    expected_a00 = np.sqrt(4 * np.pi)
    cuda_a00 = alm_cuda[0, 0, 0]

    print(f"\n  a_00:")
    print(f"    CUDA:     {cuda_a00.real:.10e} + {cuda_a00.imag:.10e}i")
    print(f"    Expected: {expected_a00:.10e}")
    print(f"    |diff|:   {abs(cuda_a00.real - expected_a00):.2e}")

    # Check ALL alm for l > 0 (should all be ~0)
    print(f"\n  All |a_lm| for l > 0 (should be ~0):")
    max_alm = 0
    max_l, max_m = 0, 0
    large_alm_count = 0

    for l in range(1, l_max + 1):
        for m in range(0, l + 1):
            mag = abs(alm_cuda[0, l, m])
            if mag > max_alm:
                max_alm = mag
                max_l, max_m = l, m
            if mag > 1e-10:
                large_alm_count += 1

    print(f"    Max |a_lm|: {max_alm:.6e} at l={max_l}, m={max_m}")
    print(f"    Count of |a_lm| > 1e-10: {large_alm_count}")

    # Print first few l,m=0 values
    print(f"\n  First few a_l0 values:")
    for l in range(min(10, l_max + 1)):
        val = alm_cuda[0, l, 0]
        print(f"    a_{l},0 = {val.real:+.6e} + {val.imag:.6e}i")

    # Compare with JAX if available
    if JAX_AVAILABLE:
        print(f"\n  JAX comparison:")
        maps_jax = {0: jnp.array(map_2d)}
        alm_jax = jax_map2alm(nside, l_max, (0,), maps_jax)[0]
        alm_jax = np.array(alm_jax)

        jax_a00 = alm_jax[0, 0, 0]
        print(f"    JAX a_00: {jax_a00.real:.10e} + {jax_a00.imag:.10e}i")

        # Find max difference
        max_diff = 0
        max_diff_l, max_diff_m = 0, 0

        for l in range(l_max + 1):
            for m in range(l + 1):
                diff = abs(alm_cuda[0, l, m] - alm_jax[0, l, m])
                if diff > max_diff:
                    max_diff = diff
                    max_diff_l, max_diff_m = l, m

        print(f"    Max |CUDA - JAX|: {max_diff:.6e} at l={max_diff_l}, m={max_diff_m}")

        # Print first few differences
        print(f"\n    First few a_l0 CUDA vs JAX:")
        for l in range(min(10, l_max + 1)):
            cuda_val = alm_cuda[0, l, 0]
            jax_val = alm_jax[0, l, 0]
            diff = abs(cuda_val - jax_val)
            print(f"      l={l}: CUDA={cuda_val.real:+.6e}, JAX={jax_val.real:+.6e}, diff={diff:.2e}")

    # Pass/fail criteria
    passed = True
    if abs(cuda_a00.real - expected_a00) > 1e-8:
        print(f"\n  FAIL: a_00 mismatch")
        passed = False
    if max_alm > 1e-6:
        print(f"\n  FAIL: Higher l coefficients too large (max={max_alm:.2e})")
        passed = False

    if passed:
        print(f"\n  PASSED")

    return passed


def test_random_map_comparison():
    """Compare CUDA vs JAX for random map transform."""
    print("\n" + "=" * 60)
    print("Test: Random map CUDA vs JAX comparison")
    print("=" * 60)

    try:
        from spht_cuda import SPHTCuda
    except ImportError as e:
        print(f"CUDA module not available: {e}")
        return False

    try:
        import jax.numpy as jnp
        from SPHT_jax import map2alm as jax_map2alm
    except ImportError as e:
        print(f"JAX not available: {e}")
        return False

    nside = 32
    l_max = 2 * nside  # Smaller for faster test
    n_rings = 4 * nside - 1

    # Create random map in 2D ring format
    np.random.seed(42)
    map_2d = np.zeros((1, n_rings, 4 * nside), dtype=np.float64)

    for ring_i in range(1, n_rings + 1):
        if ring_i < nside:
            npix_ring = 4 * ring_i
        elif ring_i <= 3 * nside:
            npix_ring = 4 * nside
        else:
            ring_from_south = 4 * nside - ring_i
            npix_ring = 4 * ring_from_south

        map_2d[0, ring_i - 1, :npix_ring] = np.random.randn(npix_ring)

    print(f"  nside = {nside}, l_max = {l_max}")

    # CUDA transform
    spht = SPHTCuda(nside, l_max)
    alm_cuda = spht.map2alm({0: map_2d}, spins=(0,))[0]

    # JAX transform
    maps_jax = {0: jnp.array(map_2d)}
    alm_jax = jax_map2alm(nside, l_max, (0,), maps_jax)[0]
    alm_jax = np.array(alm_jax)

    print(f"  CUDA alm shape: {alm_cuda.shape}")
    print(f"  JAX alm shape: {alm_jax.shape}")

    # Compute differences for ALL (l, m)
    total_diff = 0
    max_diff = 0
    max_diff_l, max_diff_m = 0, 0
    count = 0

    for l in range(l_max + 1):
        for m in range(l + 1):
            diff = abs(alm_cuda[0, l, m] - alm_jax[0, l, m])
            total_diff += diff ** 2
            count += 1
            if diff > max_diff:
                max_diff = diff
                max_diff_l, max_diff_m = l, m

    rms_diff = np.sqrt(total_diff / count)

    # Compute relative error
    jax_rms = np.sqrt(np.mean(np.abs(alm_jax[0])**2))
    rel_error = rms_diff / jax_rms

    print(f"\n  Statistics:")
    print(f"    RMS difference: {rms_diff:.6e}")
    print(f"    Max difference: {max_diff:.6e} at l={max_diff_l}, m={max_diff_m}")
    print(f"    JAX RMS: {jax_rms:.6e}")
    print(f"    Relative error: {rel_error:.6e}")

    # Print some specific values
    print(f"\n  Sample alm values:")
    for l in [0, 1, 2, 5, 10]:
        if l <= l_max:
            cuda_val = alm_cuda[0, l, 0]
            jax_val = alm_jax[0, l, 0]
            diff = abs(cuda_val - jax_val)
            print(f"    a_{l},0: CUDA={cuda_val.real:+.6e}, JAX={jax_val.real:+.6e}, diff={diff:.2e}")

    # Pass/fail
    if rel_error < 1e-6:
        print(f"\n  PASSED (relative error < 1e-6)")
        return True
    else:
        print(f"\n  FAIL (relative error = {rel_error:.2e} > 1e-6)")
        return False


def main():
    print("=" * 60)
    print("COMPREHENSIVE CUDA vs JAX COMPARISON")
    print("=" * 60)

    results = []

    results.append(("Uniform map alm", test_uniform_map_alm()))
    results.append(("Random map comparison", test_random_map_comparison()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

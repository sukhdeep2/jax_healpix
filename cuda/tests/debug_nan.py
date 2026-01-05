#!/usr/bin/env python3
"""Debug NaN issue in map2alm for l >= 3"""

import numpy as np
import sys
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT')

from spht_cuda import SPHTCuda, compute_ylm_cuda, get_ring_geometry, _get_lib
import ctypes
from ctypes import c_int, c_void_p

def compare_ring_geometry():
    """Compare CUDA-computed ring geometry with Python-computed values"""
    nside = 8
    n_rings_total = 4 * nside - 1  # 31 rings

    print("=== Comparing CUDA vs Python Ring Geometry ===")

    # Get CUDA-computed geometry
    cuda_log_beta, cuda_beta_sign = get_ring_geometry(nside)

    # Compute Python reference
    py_log_beta = np.zeros(n_rings_total)
    py_beta_sign = np.zeros(n_rings_total, dtype=np.int8)

    for ring_idx in range(n_rings_total):
        ring_i = ring_idx + 1  # 1-indexed

        if ring_i < nside:
            # North polar cap
            i2_over_3n2 = ring_i**2 / (3 * nside**2)
            beta = 1 - i2_over_3n2
            py_log_beta[ring_idx] = np.log(abs(beta))
            py_beta_sign[ring_idx] = 1
        elif ring_i > 3 * nside:
            # South polar cap
            mirror_i = 4 * nside - ring_i
            i2_over_3n2 = mirror_i**2 / (3 * nside**2)
            beta = 1 - i2_over_3n2
            py_log_beta[ring_idx] = np.log(abs(beta))
            py_beta_sign[ring_idx] = -1
        else:
            # Equatorial belt
            beta = 4/3 - 2*ring_i / (3*nside)
            if abs(beta) < 1e-15:
                py_log_beta[ring_idx] = -700  # Essentially -infinity
            else:
                py_log_beta[ring_idx] = np.log(abs(beta))
            py_beta_sign[ring_idx] = 1 if beta >= 0 else -1

    # Compare
    print("\nRing | CUDA log_beta | Python log_beta | Diff | CUDA sign | Python sign")
    print("-" * 80)
    max_diff = 0
    for i in range(n_rings_total):
        diff = abs(cuda_log_beta[i] - py_log_beta[i])
        if diff > max_diff:
            max_diff = diff
        sign_match = "OK" if cuda_beta_sign[i] == py_beta_sign[i] else "MISMATCH!"
        if diff > 1e-10 or cuda_beta_sign[i] != py_beta_sign[i]:
            print(f"{i:4d} | {cuda_log_beta[i]:13.6e} | {py_log_beta[i]:13.6e} | {diff:.2e} | {cuda_beta_sign[i]:9d} | {py_beta_sign[i]:11d} {sign_match}")

    print(f"\nMax log_beta diff: {max_diff:.2e}")

    # Now test YLM with CUDA geometry
    print("\n=== Testing YLM with CUDA-computed geometry ===")
    l_max = 24

    # Use only north rings (first 16)
    ylm_cuda = compute_ylm_cuda(l_max, cuda_log_beta[:16])
    print(f"YLM shape: {ylm_cuda.shape}")
    print(f"YLM has NaN: {np.any(np.isnan(ylm_cuda))}")
    print(f"YLM has Inf: {np.any(np.isinf(ylm_cuda))}")

    if np.any(np.isnan(ylm_cuda)):
        print("\nNaN locations:")
        nan_locs = np.argwhere(np.isnan(ylm_cuda))
        for loc in nan_locs[:20]:  # Show first 20
            print(f"  Y_{loc[0]},{loc[1]}(ring {loc[2]})")

    return cuda_log_beta, cuda_beta_sign


def test_ylm_with_geometry_offset():
    """Test YLM computation with ring geometry offset (as map2alm does)"""
    nside = 8
    l_max = 24
    n_rings_total = 4 * nside - 1  # 31 rings
    n_north_rings = 2 * nside  # 16 rings

    # Compute log_beta for all rings using exact same formula as CUDA
    log_beta = np.zeros(n_rings_total)
    beta_sign = np.zeros(n_rings_total, dtype=np.int8)

    for ring_idx in range(n_rings_total):
        ring_i = ring_idx + 1  # 1-indexed

        if ring_i < nside:
            # North polar cap
            i2_over_3n2 = ring_i**2 / (3 * nside**2)
            beta = 1 - i2_over_3n2
            log_beta[ring_idx] = np.log(abs(beta))
            beta_sign[ring_idx] = 1
        elif ring_i > 3 * nside:
            # South polar cap
            mirror_i = 4 * nside - ring_i
            i2_over_3n2 = mirror_i**2 / (3 * nside**2)
            beta = 1 - i2_over_3n2
            log_beta[ring_idx] = np.log(abs(beta))
            beta_sign[ring_idx] = -1
        else:
            # Equatorial belt
            beta = 4/3 - 2*ring_i / (3*nside)
            if abs(beta) < 1e-15:
                log_beta[ring_idx] = -700  # Essentially -infinity
            else:
                log_beta[ring_idx] = np.log(abs(beta))
            beta_sign[ring_idx] = 1 if beta >= 0 else -1

    print("=== Ring Geometry (first 20 rings) ===")
    for i in range(min(20, n_rings_total)):
        ring_i = i + 1
        cos_theta = np.exp(log_beta[i]) * beta_sign[i] if log_beta[i] > -100 else 0
        print(f"Ring {i} (i={ring_i}): log_beta={log_beta[i]:.4f}, sign={beta_sign[i]}, cos={cos_theta:.6f}")

    # Test 1: YLM with first batch (rings 0-15)
    print("\n=== Test 1: YLM for first batch (rings 0-15) ===")
    batch_log_beta = log_beta[:16].copy()

    # Call CUDA YLM compute
    ylm_cuda = compute_ylm_cuda(l_max, batch_log_beta)

    print(f"YLM shape: {ylm_cuda.shape}")
    print(f"YLM has NaN: {np.any(np.isnan(ylm_cuda))}")
    print(f"YLM has Inf: {np.any(np.isinf(ylm_cuda))}")

    # Check specific values
    for l in range(6):
        for m in range(l+1):
            val = ylm_cuda[l, m, 0]  # First ring
            status = "NaN" if np.isnan(val) else ("Inf" if np.isinf(val) else f"{val:.6e}")
            if np.isnan(val) or np.isinf(val):
                print(f"  Y_{l},{m}(ring 0) = {status}")

    # Test 2: Check equator ring (ring 15)
    print("\n=== Test 2: YLM at equator (ring 15) ===")
    for l in range(6):
        for m in range(l+1):
            val = ylm_cuda[l, m, 15]  # Equator ring
            status = "NaN" if np.isnan(val) else ("Inf" if np.isinf(val) else f"{val:.6e}")
            if np.isnan(val) or np.isinf(val) or l <= 3:
                print(f"  Y_{l},{m}(ring 15) = {status}")

    # Test 3: Full map2alm
    print("\n=== Test 3: Full map2alm ===")
    spht = SPHTCuda(nside, l_max)

    # Create uniform map
    n_rings = 4 * nside - 1
    map_2d = np.ones((1, n_rings, 4 * nside), dtype=np.float64)

    result = spht.map2alm({0: map_2d}, spins=(0,))
    alm = result[0][0]  # First map

    print(f"alm shape: {alm.shape}")
    print(f"alm has NaN: {np.any(np.isnan(alm))}")

    # Print first 10 alm values
    print("\nFirst 10 alm values:")
    for l in range(10):
        for m in range(l+1):
            val = alm[l, m]
            status = f"({val.real:.6e}, {val.imag:.6e})"
            if np.isnan(val.real) or np.isnan(val.imag):
                status = "NaN"
            print(f"  a_{l},{m} = {status}")
        print()


def test_log_arithmetic_edge_cases():
    """Test log arithmetic with edge cases"""
    print("\n=== Log Arithmetic Edge Cases ===")

    # Test what happens when cos(theta) = 0 (equator)
    log_beta_equator = -700  # log(0) ≈ -infinity

    # sin^2 = 1 - cos^2 = 1 - 0 = 1
    # log(sin^2) = log(1) = 0
    # log(sin) = 0

    log_cos2 = 2 * log_beta_equator  # -1400
    max_log = max(0, log_cos2)  # 0
    min_log = min(0, log_cos2)  # -1400

    # log(1 - cos^2) = log(1 - exp(2*log_beta))
    log_sin2 = max_log + np.log1p(-np.exp(min_log - max_log))
    log_sin = 0.5 * log_sin2

    print(f"log_beta_equator = {log_beta_equator}")
    print(f"log_cos2 = {log_cos2}")
    print(f"log_sin2 = {log_sin2}")
    print(f"log_sin = {log_sin}")
    print(f"sin(theta) = {np.exp(log_sin)}")


if __name__ == "__main__":
    compare_ring_geometry()
    print("\n" + "="*80 + "\n")
    test_log_arithmetic_edge_cases()
    print("\n" + "="*80 + "\n")
    test_ylm_with_geometry_offset()

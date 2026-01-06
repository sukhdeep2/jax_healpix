#!/usr/bin/env python3
"""Debug spin-2 Bluestein vs DFT for a single ring/pixel"""

import numpy as np
import sys
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')

import os
os.environ['SPHT_TIMING'] = '1'

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from spht_cuda import SPHTCuda, set_phase1_method

NSIDE = 64
L_MAX = 3 * NSIDE

def get_ring_info(ring_idx, nside):
    """Get ring geometry info"""
    ring_i = ring_idx + 1
    if ring_i < nside:
        N = 4 * ring_i
        phi0 = np.pi / (2.0 * ring_i) * 0.5
    elif ring_i > 3 * nside:
        mirror_i = 4 * nside - ring_i
        N = 4 * mirror_i
        phi0 = np.pi / (2.0 * mirror_i) * 0.5
    else:
        s = 1 if (ring_i % 2 == 0) else 2
        N = 4 * nside
        phi0 = np.pi / (2.0 * nside) * (1.0 - s / 2.0)
    return N, phi0

def manual_idft(Fmy, N, phi0, l_max):
    """Manual IDFT: map[j] = sum_m Fmy[m] * exp(i*m*(phi0 + j*2*pi/N))"""
    result = np.zeros(N, dtype=np.float64)
    for j in range(N):
        phi_j = phi0 + j * 2.0 * np.pi / N
        val = 0.0 + 0.0j
        for m in range(l_max + 1):
            val += Fmy[m] * np.exp(1j * m * phi_j)
        result[j] = val.real
    return result

def manual_bluestein_idft(Fmy, N, phi0, l_max, debug=False):
    """Manual Bluestein IDFT for debugging

    For IDFT: x[n] = sum_m Fmy[m] * exp(+2*pi*i*m*n/N)

    Bluestein converts this to convolution using:
    m*n = -(m-n)^2/2 + m^2/2 + n^2/2

    So: exp(2*pi*i*m*n/N) = exp(-pi*i*(m-n)^2/N) * exp(+pi*i*m^2/N) * exp(+pi*i*n^2/N)
    """
    M = 1
    while M < 2 * N - 1:
        M *= 2

    if debug:
        print(f"Manual Bluestein: N={N}, M={M}, phi0={phi0:.6f}, l_max={l_max}")

    # Pre-chirp: multiply Fmy[m] by exp(+i*m*phi0) * exp(+i*pi*m^2/N)
    chirped = np.zeros(M, dtype=np.complex128)
    for m in range(min(l_max + 1, M)):
        phase_corr = np.exp(1j * m * phi0)
        pre_chirp = np.exp(1j * np.pi * m * m / N)
        chirped[m] = Fmy[m] * phase_corr * pre_chirp

    if debug:
        print(f"  Chirped[0:4]: {chirped[:4]}")

    # Convolution chirp: exp(-i*pi*j^2/N) for circular convolution
    conv_chirp = np.zeros(M, dtype=np.complex128)
    for j in range(M):
        if j < N:
            j_eff = j
        elif j >= M - N + 1:
            j_eff = j - M
        else:
            j_eff = 0  # zero padding region - should stay zero?
        conv_chirp[j] = np.exp(-1j * np.pi * j_eff * j_eff / N)

    if debug:
        print(f"  Conv_chirp[0:M]: {conv_chirp}")

    # FFT convolution
    chirped_fft = np.fft.fft(chirped)
    conv_chirp_fft = np.fft.fft(conv_chirp)
    product = chirped_fft * conv_chirp_fft
    conv_result = np.fft.ifft(product)

    if debug:
        print(f"  Conv_result[0:N]: {conv_result[:N]}")

    # Post-chirp and extract
    result = np.zeros(N, dtype=np.float64)
    for n in range(N):
        post_chirp = np.exp(1j * np.pi * n * n / N)
        z = conv_result[n] * post_chirp
        result[n] = z.real
        if debug:
            print(f"    n={n}: conv={conv_result[n]:.6f}, post_chirp={post_chirp:.6f}, z={z:.6f}, result={result[n]:.6f}")

    return result

def main():
    nside = NSIDE
    l_max = L_MAX
    n_rings = 4 * nside - 1
    n_north_rings = 2 * nside

    # Test ring (polar cap with small N)
    test_ring = 0  # Ring 0 has N=4
    N, phi0 = get_ring_info(test_ring, nside)
    print(f"Testing ring {test_ring}: N={N}, phi0={phi0:.6f}")

    # Create random E/B alm in combined format [n_maps, lp1, lp1, 2]
    np.random.seed(42)
    lp1 = l_max + 1
    alm_EB = np.zeros((1, lp1, lp1, 2), dtype=np.complex128)
    alm_EB[..., 0] = np.random.randn(1, lp1, lp1) + 1j * np.random.randn(1, lp1, lp1)  # E
    alm_EB[..., 1] = np.random.randn(1, lp1, lp1) + 1j * np.random.randn(1, lp1, lp1)  # B

    # Make lower triangle zero (l < m)
    for l in range(lp1):
        for m in range(l + 1, lp1):
            alm_EB[0, l, m, :] = 0

    # Run CUDA with DFT
    print("\n--- Running CUDA DFT ---")
    set_phase1_method(0)  # DFT
    spht_dft = SPHTCuda(nside, l_max, version='v6',
                        storage_precision='float64',
                        recurrence_precision='float64')

    alm_dict = {2: alm_EB}
    result_dft = spht_dft.alm2map(alm_dict, spins=(2,))
    print(f"DFT result keys: {result_dft.keys()}")
    print(f"DFT result[2] shape: {result_dft[2].shape if hasattr(result_dft[2], 'shape') else type(result_dft[2])}")
    # Result is [n_maps, n_rings, max_pix] for Q, and same for U
    # They might be stacked or separate
    if isinstance(result_dft[2], tuple) or (hasattr(result_dft[2], 'shape') and len(result_dft[2].shape) == 4):
        Q_dft = result_dft[2][..., 0]
        U_dft = result_dft[2][..., 1]
    else:
        # Assume result is just Q, and -2 key is U
        Q_dft = result_dft[2][0]
        U_dft = result_dft.get(-2, result_dft[2])[0] if -2 in result_dft else result_dft[2][0]

    # Run CUDA with Bluestein
    print("\n--- Running CUDA Bluestein ---")
    set_phase1_method(2)  # Bluestein
    spht_blu = SPHTCuda(nside, l_max, version='v6',
                        storage_precision='float64',
                        recurrence_precision='float64')

    result_blu = spht_blu.alm2map({2: alm_EB}, spins=(2,))
    if isinstance(result_blu[2], tuple) or (hasattr(result_blu[2], 'shape') and len(result_blu[2].shape) == 4):
        Q_blu = result_blu[2][..., 0]
        U_blu = result_blu[2][..., 1]
    else:
        Q_blu = result_blu[2][0]
        U_blu = result_blu.get(-2, result_blu[2])[0] if -2 in result_blu else result_blu[2][0]

    # Shape is [n_maps, n_rings, max_pix, 2] -> Q/U are [n_maps, n_rings, max_pix]
    # Access: Q[map_idx, ring_idx, pixel_idx]

    # Compare for test ring
    print(f"\n=== Ring {test_ring} (N={N}) comparison ===")
    print(f"\nQ map - pixel values for ring {test_ring}:")
    for j in range(N):
        q_dft = Q_dft[0, test_ring, j]
        q_blu = Q_blu[0, test_ring, j]
        diff = q_blu - q_dft
        print(f"  pixel {j}: DFT={q_dft:12.6f}  Bluestein={q_blu:12.6f}  diff={diff:12.6f}")

    print(f"\nU map - pixel values for ring {test_ring}:")
    for j in range(N):
        u_dft = U_dft[0, test_ring, j]
        u_blu = U_blu[0, test_ring, j]
        diff = u_blu - u_dft
        print(f"  pixel {j}: DFT={u_dft:12.6f}  Bluestein={u_blu:12.6f}  diff={diff:12.6f}")

    # Check a few more rings
    print("\n=== Summary for multiple rings ===")
    for ring in [0, 1, 2, 10, nside, n_rings//2, n_rings-1]:
        N_r, _ = get_ring_info(ring, nside)
        q_diff = np.max(np.abs(Q_blu[0, ring, :N_r] - Q_dft[0, ring, :N_r]))
        u_diff = np.max(np.abs(U_blu[0, ring, :N_r] - U_dft[0, ring, :N_r]))
        print(f"Ring {ring:3d} (N={N_r:3d}): max Q diff = {q_diff:.6e}, max U diff = {u_diff:.6e}")

    # Reset to DFT
    set_phase1_method(0)

    # Also test scalar to confirm it works for small N
    print("\n\n=== SCALAR ALM2MAP COMPARISON ===")
    np.random.seed(42)
    alm_scalar = np.random.randn(1, lp1, lp1) + 1j * np.random.randn(1, lp1, lp1)
    alm_scalar = alm_scalar.astype(np.complex128)
    for l in range(lp1):
        for m in range(l + 1, lp1):
            alm_scalar[0, l, m] = 0

    print("--- Scalar DFT ---")
    set_phase1_method(0)
    spht_s_dft = SPHTCuda(nside, l_max, version='v6',
                          storage_precision='float64',
                          recurrence_precision='float64')
    result_s_dft = spht_s_dft.alm2map({0: alm_scalar}, spins=(0,))
    map_s_dft = result_s_dft[0][0]

    print("--- Scalar Bluestein ---")
    set_phase1_method(2)
    spht_s_blu = SPHTCuda(nside, l_max, version='v6',
                          storage_precision='float64',
                          recurrence_precision='float64')
    result_s_blu = spht_s_blu.alm2map({0: alm_scalar}, spins=(0,))
    map_s_blu = result_s_blu[0][0]

    print("\nScalar ring comparison:")
    for ring in [0, 1, 2, 10, nside, n_rings//2]:
        N_r, _ = get_ring_info(ring, nside)
        diff = np.max(np.abs(map_s_blu[ring, :N_r] - map_s_dft[ring, :N_r]))
        print(f"Ring {ring:3d} (N={N_r:3d}): max diff = {diff:.6e}")

    set_phase1_method(0)

    # Test manual Python Bluestein vs DFT
    print("\n\n=== MANUAL PYTHON BLUESTEIN vs DFT TEST ===")
    N_test = 4
    phi0_test = np.pi / (2.0 * 1) * 0.5  # phi0 for ring 0

    # Create simple Fmy (just first few m values)
    Fmy_test = np.zeros(l_max + 1, dtype=np.complex128)
    Fmy_test[0] = 1.0 + 0.5j
    Fmy_test[1] = 0.3 - 0.2j
    Fmy_test[2] = 0.1 + 0.1j
    Fmy_test[3] = -0.05 + 0.05j

    print(f"Test with N={N_test}, phi0={phi0_test:.6f}")
    print(f"Fmy[0:4] = {Fmy_test[:4]}")

    dft_result = manual_idft(Fmy_test, N_test, phi0_test, l_max)
    blu_result = manual_bluestein_idft(Fmy_test, N_test, phi0_test, l_max, debug=True)

    print(f"\nDirect DFT result: {dft_result}")
    print(f"Bluestein result:  {blu_result}")
    print(f"Difference:        {blu_result - dft_result}")

    # Also test with larger N to verify
    print("\n--- Testing with N=256 ---")
    N_test2 = 256
    phi0_test2 = 0.0
    dft_result2 = manual_idft(Fmy_test, N_test2, phi0_test2, l_max)
    blu_result2 = manual_bluestein_idft(Fmy_test, N_test2, phi0_test2, l_max)
    print(f"Max difference: {np.max(np.abs(blu_result2 - dft_result2)):.6e}")

    # Check M size requirements
    print("\n\n=== M SIZE ANALYSIS ===")
    print("For Bluestein: M >= input_length + output_length - 1")
    print(f"l_max = {l_max}, so input_length = l_max + 1 = {l_max + 1}")
    print("")
    for ring in [0, 1, 2, 10, nside, n_rings//2]:
        N_r, _ = get_ring_info(ring, nside)
        M_actual = 1
        while M_actual < 2 * N_r - 1:
            M_actual *= 2
        M_needed = l_max + 1 + N_r - 1

        M_correct = 1
        while M_correct < M_needed:
            M_correct *= 2

        print(f"Ring {ring:3d}: N={N_r:3d}, M_actual={M_actual:4d}, M_needed>={M_needed:4d}, M_correct={M_correct:4d}, OK={M_actual >= M_needed}")

if __name__ == "__main__":
    main()

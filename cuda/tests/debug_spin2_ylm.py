#!/usr/bin/env python3
"""Step-by-step diagnostic for spin-2 transforms.
Compare Ylm values between JAX and CUDA formulas.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from YLM_jax import sYLM_recur, alpha_lm
from jax.scipy.special import gammaln as loggamma


def compute_spin2_ylm_cuda_formula(l, m, cos_theta, sin_theta, Y0_l, Y0_l_prev):
    """
    Compute spin-2 Ylm using the CUDA formula (from astro-ph/0502469 Eq A7).

    Y0_l = Y_{l,m}(theta), Y0_l_prev = Y_{l-1,m}(theta) (spin-0)
    """
    if l < 2:
        return 0.0, 0.0

    # Normalization: sqrt((l-2)!/(l+2)!) = 1/sqrt((l-1)*l*(l+1)*(l+2))
    norm = 1.0 / np.sqrt((l-1) * l * (l+1) * (l+2))

    # alpha_{l,m} = sqrt((2l+1)(l²-m²)/(2l-1))
    alpha = np.sqrt((2*l + 1) * (l**2 - m**2) / (2*l - 1)) if l > 1 else 0.0

    sin_sq = sin_theta**2
    inv_sin_sq = 1.0 / sin_sq if sin_sq > 1e-20 else 0.0

    # ₂Y formula from CUDA (Eq A7):
    # ₂Y = norm × [(2(m²-l)/sin² - l(l-1)) × Y + 2α×cos/sin² × Y_{l-1}]
    coeff1 = 2 * (m**2 - l) * inv_sin_sq - l * (l - 1)
    coeff2 = 2 * alpha * cos_theta * inv_sin_sq
    Y2 = norm * (coeff1 * Y0_l + coeff2 * Y0_l_prev)

    # ₋₂Y formula from CUDA:
    # ₋₂Y = norm × 2m/sin² × (α × Y_{l-1} - (l-1)cos × Y)
    inner = alpha * Y0_l_prev - (l - 1) * cos_theta * Y0_l
    Ym2 = norm * 2 * m * inv_sin_sq * inner

    return Y2, Ym2


def compute_spin2_ylm_jax_formula(l, m, cos_theta, sin_theta, Y0_l, Y0_l_prev):
    """
    Compute spin-2 Ylm using the JAX formula from YLM_jax.py (sYLM_l2).
    """
    if l < 2:
        return 0.0, 0.0

    # log_factorial_norm = 0.5 * (loggamma(l - 2 + 1) - loggamma(l + 2 + 1))
    log_factorial_norm = 0.5 * (float(loggamma(l - 2 + 1)) - float(loggamma(l + 2 + 1)))
    factorial_norm = np.exp(log_factorial_norm)

    # alpha from JAX
    alpha = np.sqrt((2*l + 1) * (l**2 - m**2) / (2*l - 1)) if l > 1 else 0.0

    sin_sq = sin_theta**2

    # Y2 (spin +2) from JAX:
    # (2 * (m² - l) / sin² - l * (l - 1)) * Y0_l + 2 * (cos/sin²) * alpha * Y0_{l-1}
    # Then multiply by factorial_norm
    Y2 = (2 * (m**2 - l) / sin_sq - l * (l - 1)) * Y0_l
    Y2 += 2 * (cos_theta / sin_sq) * alpha * Y0_l_prev
    Y2 *= factorial_norm

    # Ym2 (spin -2) from JAX:
    # First: -(l-1)*cos*Y0_l + alpha*Y0_{l-1}
    # Then: multiply by 2*m/sin²
    # Then: multiply by factorial_norm
    Ym2 = -(l - 1) * cos_theta * Y0_l + alpha * Y0_l_prev
    Ym2 *= 2 * m / sin_sq
    Ym2 *= factorial_norm

    return Y2, Ym2


def test_ylm_formulas():
    """Test that CUDA and JAX formulas give the same spin-2 Ylm values."""
    print("=" * 70)
    print("Testing spin-2 Ylm formulas: CUDA vs JAX")
    print("=" * 70)

    # Test at a few theta values
    theta_vals = [0.3, 0.5, np.pi/4, np.pi/2, 1.0, 1.5, 2.0]
    l_max = 10

    # Get JAX Ylm reference
    cos_thetas = np.array([np.cos(t) for t in theta_vals])
    ylm_jax = sYLM_recur(l_max, (0, 2), jnp.array(cos_thetas))

    print("\nComparing spin-2 Ylm at theta=pi/4:")
    theta_idx = 2  # pi/4
    theta = theta_vals[theta_idx]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    print(f"\ntheta = {theta:.4f}, cos(theta) = {cos_theta:.6f}, sin(theta) = {sin_theta:.6f}")
    print()
    print(f"{'l':>3} {'m':>3} | {'JAX Y2':>14} {'CUDA Y2':>14} {'diff':>12} | {'JAX Ym2':>14} {'CUDA Ym2':>14} {'diff':>12}")
    print("-" * 100)

    all_match = True
    for l in range(2, min(l_max+1, 8)):
        for m in range(0, l+1):
            # Get JAX values
            Y2_jax = float(ylm_jax[2][l, m, theta_idx])
            Ym2_jax = float(ylm_jax[-2][l, m, theta_idx])

            # Get spin-0 values for CUDA formula
            Y0_l = float(ylm_jax[0][l, m, theta_idx])
            Y0_l_prev = float(ylm_jax[0][l-1, m, theta_idx]) if l > m else 0.0

            # Compute using CUDA formula
            Y2_cuda, Ym2_cuda = compute_spin2_ylm_cuda_formula(l, m, cos_theta, sin_theta, Y0_l, Y0_l_prev)

            # Also compute using JAX formula directly to verify
            Y2_jax_direct, Ym2_jax_direct = compute_spin2_ylm_jax_formula(l, m, cos_theta, sin_theta, Y0_l, Y0_l_prev)

            diff_Y2 = abs(Y2_cuda - Y2_jax)
            diff_Ym2 = abs(Ym2_cuda - Ym2_jax)

            status = ""
            if diff_Y2 > 1e-10 or diff_Ym2 > 1e-10:
                status = " <-- MISMATCH"
                all_match = False

            print(f"{l:3d} {m:3d} | {Y2_jax:14.8f} {Y2_cuda:14.8f} {diff_Y2:12.2e} | {Ym2_jax:14.8f} {Ym2_cuda:14.8f} {diff_Ym2:12.2e}{status}")

    print()
    if all_match:
        print("SUCCESS: CUDA and JAX spin-2 Ylm formulas match!")
    else:
        print("FAILURE: Formulas don't match, check the implementation!")

    return all_match


def test_ylm_at_poles():
    """Test spin-2 Ylm behavior near poles (sin_theta -> 0)."""
    print("\n" + "=" * 70)
    print("Testing spin-2 Ylm near poles")
    print("=" * 70)

    # Near north pole
    theta_vals = [0.01, 0.05, 0.1]
    l_max = 6

    cos_thetas = np.array([np.cos(t) for t in theta_vals])
    ylm_jax = sYLM_recur(l_max, (0, 2), jnp.array(cos_thetas))

    print("\nNear north pole (small theta):")
    for ti, theta in enumerate(theta_vals):
        print(f"\ntheta = {theta:.4f}:")
        print(f"  l=2,m=0: Y2={float(ylm_jax[2][2,0,ti]):12.6e}, Ym2={float(ylm_jax[-2][2,0,ti]):12.6e}")
        print(f"  l=2,m=1: Y2={float(ylm_jax[2][2,1,ti]):12.6e}, Ym2={float(ylm_jax[-2][2,1,ti]):12.6e}")
        print(f"  l=2,m=2: Y2={float(ylm_jax[2][2,2,ti]):12.6e}, Ym2={float(ylm_jax[-2][2,2,ti]):12.6e}")


def test_parity_rules():
    """Test north-south parity rules for spin-2 Ylm."""
    print("\n" + "=" * 70)
    print("Testing spin-2 parity rules (north vs south)")
    print("=" * 70)

    l_max = 6
    theta_north = np.pi / 4
    theta_south = np.pi - theta_north  # Mirror about equator

    cos_thetas = np.array([np.cos(theta_north), np.cos(theta_south)])
    ylm_jax = sYLM_recur(l_max, (0, 2), jnp.array(cos_thetas))

    print(f"\nNorth: theta={theta_north:.4f}, South: theta={theta_south:.4f}")
    print()
    print(f"{'l':>3} {'m':>3} | {'Y2_north':>12} {'Y2_south':>12} {'ratio':>8} | {'Ym2_north':>12} {'Ym2_south':>12} {'ratio':>8}")
    print("-" * 90)

    for l in range(2, l_max+1):
        for m in range(0, l+1):
            Y2_n = float(ylm_jax[2][l, m, 0])
            Y2_s = float(ylm_jax[2][l, m, 1])
            Ym2_n = float(ylm_jax[-2][l, m, 0])
            Ym2_s = float(ylm_jax[-2][l, m, 1])

            # Expected parity: Y2(south) = (-1)^(l+m) * Y2(north)
            #                  Ym2(south) = (-1)^(l+m+1) * Ym2(north)
            expected_Y2_parity = (-1)**(l+m)
            expected_Ym2_parity = (-1)**(l+m+1)

            actual_Y2_ratio = Y2_s / Y2_n if abs(Y2_n) > 1e-15 else 0
            actual_Ym2_ratio = Ym2_s / Ym2_n if abs(Ym2_n) > 1e-15 else 0

            print(f"{l:3d} {m:3d} | {Y2_n:12.6f} {Y2_s:12.6f} {actual_Y2_ratio:8.2f} | {Ym2_n:12.6f} {Ym2_s:12.6f} {actual_Ym2_ratio:8.2f}  (expect {expected_Y2_parity:+d}, {expected_Ym2_parity:+d})")


def test_alm2map_formula():
    """Test the alm2map formula: Q = ₂Y×E + i×₋₂Y×B, U = ₋₂Y×E + i×₂Y×B"""
    print("\n" + "=" * 70)
    print("Testing alm2map spin-2 formula")
    print("=" * 70)

    l_max = 6
    theta = np.pi / 4
    cos_theta = np.cos(theta)

    ylm_jax = sYLM_recur(l_max, (0, 2), jnp.array([cos_theta]))

    # Test with simple E, B alm values
    E_alm = np.zeros((l_max+1, l_max+1), dtype=np.complex128)
    B_alm = np.zeros((l_max+1, l_max+1), dtype=np.complex128)

    # Set E_22 = 1 + 0j
    E_alm[2, 2] = 1.0 + 0j

    print(f"\nTest: E_22 = 1, all other alm = 0")
    print(f"theta = {theta:.4f}")

    # Compute Q and U at this theta (single pixel)
    # For a single l,m: Q = Y2*E + i*Ym2*B, U = Ym2*E + i*Y2*B
    l, m = 2, 2
    Y2 = float(ylm_jax[2][l, m, 0])
    Ym2 = float(ylm_jax[-2][l, m, 0])

    E_lm = E_alm[l, m]
    B_lm = B_alm[l, m]

    # Q = Y2*E + i*Ym2*B (complex)
    Q_complex = Y2 * E_lm + 1j * Ym2 * B_lm
    # U = Ym2*E + i*Y2*B (complex)
    U_complex = Ym2 * E_lm + 1j * Y2 * B_lm

    print(f"\nBefore post-processing:")
    print(f"  Y2 = {Y2:.8f}, Ym2 = {Ym2:.8f}")
    print(f"  Q_complex = {Q_complex}")
    print(f"  U_complex = {U_complex}")

    # Post-processing: Q *= -1, U *= i
    Q_final = -Q_complex
    U_final = 1j * U_complex

    print(f"\nAfter post-processing (Q *= -1, U *= i):")
    print(f"  Q_final = {Q_final}")
    print(f"  U_final = {U_final}")
    print(f"  Re(Q_final) = {Q_final.real:.8f}")
    print(f"  Re(U_final) = {U_final.real:.8f}")


def main():
    test_ylm_formulas()
    test_ylm_at_poles()
    test_parity_rules()
    test_alm2map_formula()


if __name__ == "__main__":
    main()

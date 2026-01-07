#!/usr/bin/env python3
"""
Tests for healpix_ops CUDA functions against healpy reference.

Tests:
- gauss_beam: Gaussian beam window function
- smoothalm: Apply beam smoothing to alm
- almxfl: Multiply alm by filter function
- pixwin: Pixel window function
- synalm: Generate random alm from power spectrum
- resize_alm: Resize alm to new l_max

Reference: healpy library functions

Usage:
    python test_healpix_ops.py                    # Default: f64, nside=256
    python test_healpix_ops.py --precision f32   # Test with float32
    python test_healpix_ops.py --nside 128       # Test with nside=128
    python test_healpix_ops.py --precision f32 --nside 512
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Add the python directory to path
python_dir = str(Path(__file__).parent.parent / "python")
sys.path.insert(0, python_dir)

import healpy as hp

# Try to import our CUDA implementation
try:
    import healpix_ops
    from healpix_ops import gauss_beam, smoothalm, almxfl, pixwin, resize_alm, synalm
    import spht_cuda
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import healpix_ops: {e}")
    print(f"  Python path includes: {python_dir}")
    CUDA_AVAILABLE = False


# Global test configuration
class TestConfig:
    precision = "float64"  # "float32" or "float64"
    nside = 256
    l_max = None  # Derived from nside if not set

    @classmethod
    def set_precision(cls, prec):
        cls.precision = prec
        spht_cuda.set_precision(storage=prec, recurrence=prec)

    @classmethod
    def get_dtype(cls):
        return np.float32 if cls.precision == "float32" else np.float64

    @classmethod
    def get_complex_dtype(cls):
        return np.complex64 if cls.precision == "float32" else np.complex128

    @classmethod
    def get_threshold(cls, base_threshold):
        """Get threshold adjusted for precision."""
        if cls.precision == "float32":
            return max(base_threshold, 1e-5)  # f32 has ~7 digits precision
        return base_threshold


def alm_2d_to_healpy(alm_2d):
    """Convert our [l_max+1, l_max+1] alm format to healpy's 1D format."""
    l_max = alm_2d.shape[0] - 1
    n_alm = hp.Alm.getsize(l_max)
    alm_hp = np.zeros(n_alm, dtype=np.complex128)

    for l in range(l_max + 1):
        for m in range(l + 1):
            idx = hp.Alm.getidx(l_max, l, m)
            alm_hp[idx] = alm_2d[l, m]

    return alm_hp


def alm_healpy_to_2d(alm_hp, l_max):
    """Convert healpy's 1D alm format to our [l_max+1, l_max+1] format."""
    alm_2d = np.zeros((l_max + 1, l_max + 1), dtype=np.complex128)

    for l in range(l_max + 1):
        for m in range(l + 1):
            idx = hp.Alm.getidx(l_max, l, m)
            alm_2d[l, m] = alm_hp[idx]

    return alm_2d


def test_gauss_beam():
    """Test Gaussian beam window function against healpy."""
    print("\n" + "="*70)
    print("TEST: gauss_beam")
    print("="*70)

    # Test parameters
    fwhm_arcmin = 30.0  # 30 arcmin beam
    l_max = TestConfig.l_max or 3 * TestConfig.nside - 1

    # Convert to radians
    fwhm_rad = np.radians(fwhm_arcmin / 60.0)

    # Healpy reference (always f64)
    hp_beam = hp.gauss_beam(fwhm_rad, lmax=l_max, pol=False)

    # CUDA implementation
    cuda_beam = gauss_beam(fwhm_rad, l_max)

    # Compare ALL ell values
    diff = np.abs(hp_beam - cuda_beam)
    max_diff = diff.max()
    mean_diff = diff.mean()
    max_diff_l = np.argmax(diff)

    print(f"\nFWHM = {fwhm_arcmin} arcmin, l_max = {l_max}")
    print(f"Precision: {TestConfig.precision}")
    print(f"\nComparison across ALL {l_max + 1} ell values:")
    print(f"  Max absolute difference: {max_diff:.2e} (at l={max_diff_l})")
    print(f"  Mean absolute difference: {mean_diff:.2e}")

    # Show a few sample values
    sample_ls = [0, l_max//4, l_max//2, 3*l_max//4, l_max]
    print(f"\n{'l':>6} {'healpy':>15} {'CUDA':>15} {'diff':>12}")
    print("-" * 50)
    for l in sample_ls:
        print(f"{l:>6} {hp_beam[l]:>15.6e} {cuda_beam[l]:>15.6e} {diff[l]:>12.2e}")

    # Pass/fail threshold
    threshold = TestConfig.get_threshold(1e-10)
    passed = max_diff < threshold
    print(f"\nStatus: {'PASS' if passed else 'FAIL'} (threshold={threshold})")

    return passed


def test_pixwin():
    """Test pixel window function against healpy.

    Note: Our implementation uses a Gaussian approximation which differs
    from healpy's exact precomputed tables. We test for reasonable behavior.
    """
    print("\n" + "="*70)
    print("TEST: pixwin (approximate Gaussian model)")
    print("="*70)

    nside = TestConfig.nside
    l_max = 3 * nside - 1

    # CUDA implementation (approximate)
    cuda_pixwin = pixwin(nside, l_max)

    # Healpy reference
    hp_pixwin = hp.pixwin(nside, lmax=l_max)

    print(f"\nnside={nside}, l_max={l_max}")
    print(f"Precision: {TestConfig.precision}")

    # Test 1: pixwin[0] should be 1.0
    passed1 = abs(cuda_pixwin[0] - 1.0) < 1e-10
    print(f"  pixwin[0] = {cuda_pixwin[0]:.6f} (expected 1.0): {'PASS' if passed1 else 'FAIL'}")

    # Test 2: pixwin should be monotonically decreasing
    diff = np.diff(cuda_pixwin)
    passed2 = np.all(diff <= 1e-10)
    print(f"  Monotonic decreasing: {'PASS' if passed2 else 'FAIL'}")

    # Test 3: pixwin at l=nside should be in reasonable range
    val_at_nside = cuda_pixwin[nside]
    passed3 = 0.1 < val_at_nside < 1.0
    print(f"  pixwin[nside={nside}] = {val_at_nside:.4f} (should be 0.1-1.0): {'PASS' if passed3 else 'FAIL'}")

    # Test 4: pixwin at l=3*nside should be close to 0
    val_at_3nside = cuda_pixwin[-1]
    passed4 = val_at_3nside < 0.3
    print(f"  pixwin[3*nside-1] = {val_at_3nside:.4f} (should be < 0.3): {'PASS' if passed4 else 'FAIL'}")

    # Compare to healpy (informational)
    print(f"\n  Comparison to healpy (we use Gaussian approx, healpy uses exact tables):")
    sample_ls = [0, nside//2, nside, 2*nside, 3*nside-1]
    print(f"  {'l':>6} {'CUDA':>12} {'healpy':>12}")
    print("  " + "-" * 35)
    for l in sample_ls:
        if l <= l_max:
            print(f"  {l:>6} {cuda_pixwin[l]:>12.4f} {hp_pixwin[l]:>12.4f}")

    passed = passed1 and passed2 and passed3 and passed4
    print(f"\nStatus: {'PASS' if passed else 'FAIL'}")

    return passed


def test_almxfl():
    """Test almxfl (multiply alm by filter) against healpy."""
    print("\n" + "="*70)
    print("TEST: almxfl")
    print("="*70)

    np.random.seed(42)
    l_max = TestConfig.l_max or min(500, 3 * TestConfig.nside - 1)

    # Create random alm in healpy format
    n_alm = hp.Alm.getsize(l_max)
    alm_hp = np.random.randn(n_alm) + 1j * np.random.randn(n_alm)
    # Make m=0 modes real
    for l in range(l_max + 1):
        idx = hp.Alm.getidx(l_max, l, 0)
        alm_hp[idx] = alm_hp[idx].real

    # Create filter function
    fl = np.exp(-np.arange(l_max + 1) / 50.0)

    # Healpy reference
    hp_result = hp.almxfl(alm_hp.copy(), fl)

    # Convert to our format and run CUDA
    alm_2d = alm_healpy_to_2d(alm_hp, l_max)
    cuda_result_2d = almxfl(alm_2d, fl)
    cuda_result = alm_2d_to_healpy(cuda_result_2d)

    # Compare using ABSOLUTE error (alm has mean ~0)
    abs_diff = np.abs(hp_result - cuda_result)
    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()
    max_idx = np.argmax(abs_diff)

    print(f"\nl_max = {l_max}")
    print(f"Precision: {TestConfig.precision}")
    print(f"\nComparison across ALL {n_alm} alm coefficients (healpy format):")
    print(f"  Max absolute error: {max_abs_diff:.2e} (at index {max_idx})")
    print(f"  Mean absolute error: {mean_abs_diff:.2e}")

    threshold = TestConfig.get_threshold(1e-10)
    passed = max_abs_diff < threshold
    print(f"\nStatus: {'PASS' if passed else 'FAIL'} (threshold={threshold})")

    return passed


def test_synalm():
    """Test synalm (random alm from Cl) statistics."""
    print("\n" + "="*70)
    print("TEST: synalm (statistical)")
    print("="*70)

    l_max = TestConfig.l_max or min(200, 3 * TestConfig.nside - 1)
    l = np.arange(l_max + 1)
    cl = 1.0 / (1 + l)**2
    cl[0] = 0  # monopole = 0

    n_realizations = 50
    measured_cl_sum = np.zeros(l_max + 1)

    print(f"\nl_max = {l_max}, n_realizations = {n_realizations}")
    print(f"Precision: {TestConfig.precision}")

    for i in range(n_realizations):
        # CUDA synalm
        alm_2d = synalm(cl, lmax=l_max, seed=i*1000+42)

        # Convert to healpy format and compute Cl
        alm_hp = alm_2d_to_healpy(alm_2d)
        measured_cl_sum += hp.alm2cl(alm_hp, lmax=l_max)

    measured_cl = measured_cl_sum / n_realizations

    # Check ratio for ALL l values (excluding l=0)
    valid_mask = cl > 1e-10
    valid_mask[0] = False
    ratios = measured_cl[valid_mask] / cl[valid_mask]

    print(f"\nStatistics across ALL {np.sum(valid_mask)} valid ell values:")
    print(f"  Mean ratio (measured/input): {np.mean(ratios):.3f} (expected: ~1.0)")
    print(f"  Std of ratios: {np.std(ratios):.3f}")
    print(f"  Min ratio: {np.min(ratios):.3f}")
    print(f"  Max ratio: {np.max(ratios):.3f}")

    # Show sample values
    sample_ls = [1, 5, 10, 20, 50, 100, min(150, l_max), min(200, l_max)]
    sample_ls = [l_val for l_val in sample_ls if l_val <= l_max]
    print(f"\n{'l':>6} {'input Cl':>12} {'measured':>12} {'ratio':>10}")
    print("-" * 45)
    for l_val in sample_ls:
        if cl[l_val] > 1e-10:
            ratio = measured_cl[l_val] / cl[l_val]
            print(f"{l_val:>6} {cl[l_val]:>12.4e} {measured_cl[l_val]:>12.4e} {ratio:>10.3f}")

    mean_ratio = np.mean(ratios)
    passed = 0.7 < mean_ratio < 1.3
    print(f"\nStatus: {'PASS' if passed else 'FAIL'}")

    return passed


def test_resize_alm():
    """Test resize_alm against healpy."""
    print("\n" + "="*70)
    print("TEST: resize_alm")
    print("="*70)

    np.random.seed(42)
    l_max_in = min(50, TestConfig.nside)
    l_max_out = min(100, 2 * TestConfig.nside)

    # Create random alm in healpy format
    n_alm_in = hp.Alm.getsize(l_max_in)
    alm_hp_in = np.random.randn(n_alm_in) + 1j * np.random.randn(n_alm_in)
    for l in range(l_max_in + 1):
        idx = hp.Alm.getidx(l_max_in, l, 0)
        alm_hp_in[idx] = alm_hp_in[idx].real

    # Healpy resize (using ud_grade on alm is not direct, so we compare coefficients)
    # healpy doesn't have a direct resize_alm, so we just verify our implementation preserves data

    # Convert to our format
    alm_2d_in = alm_healpy_to_2d(alm_hp_in, l_max_in)

    # CUDA resize up
    alm_2d_out = resize_alm(alm_2d_in, l_max_out)

    print(f"\nResize: l_max {l_max_in} -> {l_max_out}")
    print(f"Precision: {TestConfig.precision}")
    print(f"Input shape: {alm_2d_in.shape}")
    print(f"Output shape: {alm_2d_out.shape}")

    # Check ALL preserved coefficients
    max_preserved_diff = 0
    max_idx = (0, 0)
    for l in range(l_max_in + 1):
        for m in range(l + 1):
            diff = abs(alm_2d_in[l, m] - alm_2d_out[l, m])
            if diff > max_preserved_diff:
                max_preserved_diff = diff
                max_idx = (l, m)

    print(f"\nPreserved coefficients:")
    print(f"  Max absolute error: {max_preserved_diff:.2e} (at l={max_idx[0]}, m={max_idx[1]})")

    # Check ALL new coefficients are zero
    max_new = 0
    for l in range(l_max_in + 1, l_max_out + 1):
        for m in range(l + 1):
            max_new = max(max_new, abs(alm_2d_out[l, m]))

    print(f"\nNew (zero-padded) coefficients:")
    print(f"  Max absolute value: {max_new:.2e}")

    threshold = TestConfig.get_threshold(1e-14)
    passed = max_preserved_diff < threshold and max_new < threshold

    # Test resize down (round-trip)
    print(f"\n--- Resize down: l_max {l_max_out} -> {l_max_in} ---")
    alm_2d_back = resize_alm(alm_2d_out, l_max_in)

    back_diff = np.abs(alm_2d_in - alm_2d_back)
    max_back_diff = back_diff.max()

    print(f"Max absolute error after round-trip: {max_back_diff:.2e}")

    passed = passed and max_back_diff < threshold
    print(f"\nOverall status: {'PASS' if passed else 'FAIL'} (threshold={threshold})")

    return passed


def test_smoothalm():
    """Test smoothalm (Gaussian beam smoothing) against healpy."""
    print("\n" + "="*70)
    print("TEST: smoothalm")
    print("="*70)

    np.random.seed(42)
    l_max = TestConfig.l_max or min(100, 3 * TestConfig.nside - 1)

    # Create random alm in healpy format
    n_alm = hp.Alm.getsize(l_max)
    alm_hp = np.random.randn(n_alm) + 1j * np.random.randn(n_alm)
    for l in range(l_max + 1):
        idx = hp.Alm.getidx(l_max, l, 0)
        alm_hp[idx] = alm_hp[idx].real

    fwhm_rad = np.radians(60.0 / 60.0)  # 1 degree beam

    # Healpy reference
    hp_result = hp.smoothalm(alm_hp.copy(), fwhm=fwhm_rad)

    # Convert to our format and run CUDA
    alm_2d = alm_healpy_to_2d(alm_hp, l_max)
    cuda_result_2d = smoothalm(alm_2d, fwhm=fwhm_rad)
    cuda_result = alm_2d_to_healpy(cuda_result_2d)

    # Compare using ABSOLUTE error (alm has mean ~0)
    abs_diff = np.abs(hp_result - cuda_result)
    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()
    max_idx = np.argmax(abs_diff)

    # Find l,m of max error
    for l in range(l_max + 1):
        for m in range(l + 1):
            if hp.Alm.getidx(l_max, l, m) == max_idx:
                max_lm = (l, m)
                break

    print(f"\nl_max = {l_max}, FWHM = 60 arcmin")
    print(f"Precision: {TestConfig.precision}")
    print(f"\nComparison against healpy.smoothalm across ALL {n_alm} coefficients:")
    print(f"  Max absolute error: {max_abs_diff:.2e} (at l={max_lm[0]}, m={max_lm[1]})")
    print(f"  Mean absolute error: {mean_abs_diff:.2e}")

    # Show sample comparisons
    sample_ls = [0, 10, 30, 50, 70, l_max]
    print(f"\n{'l':>4} {'m':>4} {'healpy':>15} {'CUDA':>15} {'diff':>12}")
    print("-" * 55)
    for l in sample_ls:
        if l <= l_max:
            m = min(l, 5)  # Show m=5 or m=l if l<5
            idx = hp.Alm.getidx(l_max, l, m)
            print(f"{l:>4} {m:>4} {hp_result[idx]:>15.6e} {cuda_result[idx]:>15.6e} {abs_diff[idx]:>12.2e}")

    threshold = TestConfig.get_threshold(1e-8)
    passed = max_abs_diff < threshold
    print(f"\nStatus: {'PASS' if passed else 'FAIL'} (threshold={threshold})")

    return passed


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test healpix_ops CUDA functions")
    parser.add_argument("--precision", "-p", choices=["f32", "f64", "float32", "float64"],
                        default="f64", help="Precision to use (default: f64)")
    parser.add_argument("--nside", "-n", type=int, default=256,
                        help="HEALPix nside parameter (default: 256)")
    parser.add_argument("--lmax", "-l", type=int, default=None,
                        help="Maximum l (default: 3*nside-1)")
    args = parser.parse_args()

    # Normalize precision
    prec = args.precision
    if prec in ("f32", "float32"):
        prec = "float32"
    else:
        prec = "float64"

    # Set configuration
    TestConfig.nside = args.nside
    TestConfig.l_max = args.lmax

    print("="*70)
    print("HEALPIX_OPS TEST SUITE")
    print("Testing CUDA implementations against healpy reference")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Precision: {prec}")
    print(f"  nside: {TestConfig.nside}")
    print(f"  l_max: {TestConfig.l_max or f'auto (3*nside-1 = {3*TestConfig.nside-1})'}")

    if not CUDA_AVAILABLE:
        print("\nERROR: CUDA healpix_ops module not available. Build first!")
        return 1

    # Set precision
    TestConfig.set_precision(prec)

    results = {}

    # Run tests
    results['gauss_beam'] = test_gauss_beam()
    results['pixwin'] = test_pixwin()
    results['almxfl'] = test_almxfl()
    results['synalm'] = test_synalm()
    results['resize_alm'] = test_resize_alm()
    results['smoothalm'] = test_smoothalm()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Precision: {prec}, nside: {TestConfig.nside}")
    print()

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s}: {status}")
        all_passed = all_passed and passed

    print("="*70)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

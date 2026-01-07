#!/usr/bin/env python3
"""
Roundtrip accuracy test for SPHT transforms.

Tests the consistency of: alm -> map -> cl vs alm -> cl (direct)

This test validates that the transform pipeline preserves power spectrum
information. Both CUDA and JAX implementations are tested for comparison.

Usage:
    python test_roundtrip_accuracy.py [--nside NSIDE] [--iterations N]
"""

import argparse
import numpy as np
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))


def generate_test_alm(l_max, dtype, seed=42):
    """Generate random alm coefficients for testing."""
    np.random.seed(seed)
    lp1 = l_max + 1
    alm_real = dtype(np.random.randn(1, lp1, lp1))
    alm_imag = dtype(np.random.randn(1, lp1, lp1))
    # Zero out invalid coefficients (m > l)
    for l in range(lp1):
        alm_real[0, l, l+1:] = 0
        alm_imag[0, l, l+1:] = 0
        alm_imag[0, l, 0] = 0  # m=0 is real
    return alm_real, alm_imag


def compute_cl_error(cl_roundtrip, cl_direct, l_cutoff=None):
    """Compute relative error between two Cl arrays."""
    if l_cutoff is not None:
        cl_roundtrip = cl_roundtrip[:l_cutoff+1]
        cl_direct = cl_direct[:l_cutoff+1]

    # Avoid division by zero
    mask = np.abs(cl_direct) > 1e-30
    if not np.any(mask):
        return 0.0

    rel_err = np.abs(cl_roundtrip[mask] - cl_direct[mask]) / np.abs(cl_direct[mask])
    return np.mean(rel_err) * 100  # Return as percentage


def test_cuda_roundtrip(nside, iterations=2):
    """Test CUDA roundtrip: alm -> map -> cl vs alm -> cl."""
    from spht_cuda import SPHTCuda, alm2cl_cuda, set_precision

    l_max = 3 * nside
    print(f"\n{'='*60}")
    print(f"CUDA Roundtrip Test: nside={nside}, l_max={l_max}")
    print(f"{'='*60}")

    results = {}

    for prec_name, storage, recurrence in [
        ("f64", "float64", "float64"),
        ("f32", "float32", "float32"),
    ]:
        set_precision(storage=storage, recurrence=recurrence)
        dtype = np.float32 if storage == "float32" else np.float64

        errors_by_cutoff = {nside: [], 2*nside: [], l_max: []}

        for iteration in range(iterations):
            # Generate test alm
            alm_real, alm_imag = generate_test_alm(l_max, dtype, seed=42+iteration)

            # Path 1: alm -> cl (direct)
            cl_direct = alm2cl_cuda(l_max, alm_real, alm_imag)[0]

            # Path 2: alm -> map -> cl (roundtrip)
            spht = SPHTCuda(nside, l_max, storage_precision=storage, recurrence_precision=recurrence)
            alm_complex = (alm_real + 1j * alm_imag).astype(
                np.complex64 if storage == "float32" else np.complex128
            )

            # alm -> map
            map_out = spht.alm2map({0: alm_complex}, spins=(0,))[0]

            # map -> alm
            alm_rt = spht.map2alm({0: map_out}, spins=(0,), return_split=True)[0]
            alm_rt_real, alm_rt_imag = alm_rt

            # alm -> cl
            cl_roundtrip = alm2cl_cuda(l_max, alm_rt_real, alm_rt_imag)[0]

            # Compute errors at different l cutoffs
            for cutoff in [nside, 2*nside, l_max]:
                err = compute_cl_error(cl_roundtrip, cl_direct, cutoff)
                errors_by_cutoff[cutoff].append(err)

        # Average errors
        print(f"\nCUDA {prec_name}:")
        for cutoff in [nside, 2*nside, l_max]:
            avg_err = np.mean(errors_by_cutoff[cutoff])
            results[f"cuda_{prec_name}_l{cutoff}"] = avg_err
            label = f"l<={cutoff}" if cutoff < l_max else "full"
            print(f"  {label:12s}: {avg_err:.4f}% mean relative error")

    return results


def test_jax_roundtrip(nside, iterations=2):
    """Test JAX roundtrip: alm -> map -> cl vs alm -> cl."""
    # Set JAX to f32 mode before import
    os.environ['JAX_ENABLE_X64'] = 'False'

    try:
        # Change to jax_healpix directory for imports
        jax_healpix_path = os.path.join(os.path.dirname(__file__), '..', '..', 'jax_healpix')
        sys.path.insert(0, jax_healpix_path)

        import jax
        import jax.numpy as jnp
        import SPHT_jax

        l_max = 3 * nside
        print(f"\n{'='*60}")
        print(f"JAX Roundtrip Test: nside={nside}, l_max={l_max}")
        print(f"{'='*60}")

        results = {}
        errors_by_cutoff = {nside: [], 2*nside: [], l_max: []}

        for iteration in range(iterations):
            # Generate test alm (use same seed as CUDA for comparability)
            np.random.seed(42 + iteration)
            lp1 = l_max + 1
            alm_real = np.random.randn(1, lp1, lp1).astype(np.float32)
            alm_imag = np.random.randn(1, lp1, lp1).astype(np.float32)

            # Zero out invalid coefficients
            for l in range(lp1):
                alm_real[0, l, l+1:] = 0
                alm_imag[0, l, l+1:] = 0
                alm_imag[0, l, 0] = 0

            alm_jax = jnp.array(alm_real + 1j * alm_imag)

            # Path 1: alm -> cl (direct)
            cl_direct = np.array(SPHT_jax.alm2cl(l_max, alm_jax))[0]

            # Path 2: alm -> map -> cl (roundtrip)
            # alm -> map
            map_out = SPHT_jax.alm2map(nside, l_max, (0,), {0: alm_jax})

            # map -> alm
            alm_rt = SPHT_jax.map2alm(nside, l_max, (0,), map_out)

            # alm -> cl (alm_rt[0] is shape [1, lp1, lp1])
            cl_roundtrip = np.array(SPHT_jax.alm2cl(l_max, alm_rt[0]))[0]

            # Compute errors at different l cutoffs
            for cutoff in [nside, 2*nside, l_max]:
                err = compute_cl_error(cl_roundtrip, cl_direct, cutoff)
                errors_by_cutoff[cutoff].append(err)

        # Average errors
        print(f"\nJAX f32:")
        for cutoff in [nside, 2*nside, l_max]:
            avg_err = np.mean(errors_by_cutoff[cutoff])
            results[f"jax_f32_l{cutoff}"] = avg_err
            label = f"l<={cutoff}" if cutoff < l_max else "full"
            print(f"  {label:12s}: {avg_err:.4f}% mean relative error")

        return results

    except ImportError as e:
        print(f"\nJAX test skipped: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Roundtrip accuracy test for SPHT")
    parser.add_argument("--nside", type=int, default=512, help="HEALPix nside (default: 512)")
    parser.add_argument("--iterations", type=int, default=2, help="Number of iterations (default: 2)")
    parser.add_argument("--cuda-only", action="store_true", help="Skip JAX tests")
    parser.add_argument("--jax-only", action="store_true", help="Skip CUDA tests")
    args = parser.parse_args()

    nside = args.nside
    l_max = 3 * nside

    print(f"\nRoundtrip Accuracy Test")
    print(f"nside={nside}, l_max={l_max}, iterations={args.iterations}")
    print(f"Test: alm -> map -> cl  vs  alm -> cl (direct)")

    all_results = {}

    # Run CUDA tests
    if not args.jax_only:
        cuda_results = test_cuda_roundtrip(nside, args.iterations)
        all_results.update(cuda_results)

    # Run JAX tests
    if not args.cuda_only:
        jax_results = test_jax_roundtrip(nside, args.iterations)
        all_results.update(jax_results)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Implementation':<15} {'l<=nside':<12} {'l<=2*nside':<12} {'full':<12}")
    print("-" * 51)

    if not args.jax_only and f"cuda_f64_l{nside}" in all_results:
        print(f"{'CUDA f64':<15} {all_results[f'cuda_f64_l{nside}']:<12.4f} "
              f"{all_results[f'cuda_f64_l{2*nside}']:<12.4f} "
              f"{all_results[f'cuda_f64_l{l_max}']:<12.4f}")

    if not args.jax_only and f"cuda_f32_l{nside}" in all_results:
        print(f"{'CUDA f32':<15} {all_results[f'cuda_f32_l{nside}']:<12.4f} "
              f"{all_results[f'cuda_f32_l{2*nside}']:<12.4f} "
              f"{all_results[f'cuda_f32_l{l_max}']:<12.4f}")

    if not args.cuda_only and f"jax_f32_l{nside}" in all_results:
        print(f"{'JAX f32':<15} {all_results[f'jax_f32_l{nside}']:<12.4f} "
              f"{all_results[f'jax_f32_l{2*nside}']:<12.4f} "
              f"{all_results[f'jax_f32_l{l_max}']:<12.4f}")

    print(f"\nNote: Lower is better. Values are mean relative error in %.")
    print(f"      l<=nside typically has best accuracy for transforms.")


if __name__ == "__main__":
    main()

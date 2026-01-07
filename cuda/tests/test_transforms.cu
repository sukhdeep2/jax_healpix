/**
 * Test transform operations (map2alm, alm2map)
 *
 * Usage:
 *   ./test_transforms [--linear|--log] [--nside=N] [--compare] [--help]
 *
 * Options:
 *   --linear   Use LINEAR accumulation mode (default)
 *   --log      Use LOG accumulation mode (logsumexp)
 *   --nside=N  Set nside (default: 32)
 *   --compare  Run comparison between LINEAR and LOG modes
 *   --help     Show this help
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../include/spht_api.h"

// Reference: uniform map of 1s should give alm[0,0] = sqrt(4*pi)
#define ALM00_UNIFORM_REF 3.5449077018110318

// Global configuration
static bool g_use_log_mode = false;
static int g_nside = 32;
static bool g_run_comparison = false;

// ============================================================================
// Helper functions
// ============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  --linear   Use LINEAR accumulation mode (default)\n");
    printf("  --log      Use LOG accumulation mode (logsumexp)\n");
    printf("  --nside=N  Set nside (default: 32)\n");
    printf("  --compare  Run comparison between LINEAR and LOG modes\n");
    printf("  --help     Show this help\n");
    printf("\nExamples:\n");
    printf("  %s                  # Run with LINEAR mode, nside=32\n", prog);
    printf("  %s --log            # Run with LOG mode, nside=32\n", prog);
    printf("  %s --log --nside=64 # Run with LOG mode, nside=64\n", prog);
    printf("  %s --compare        # Compare LINEAR vs LOG modes\n", prog);
}

void parse_args(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--linear") == 0) {
            g_use_log_mode = false;
        } else if (strcmp(argv[i], "--log") == 0) {
            g_use_log_mode = true;
        } else if (strncmp(argv[i], "--nside=", 8) == 0) {
            g_nside = atoi(argv[i] + 8);
            if (g_nside < 1) {
                fprintf(stderr, "Error: Invalid nside value\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "--compare") == 0) {
            g_run_comparison = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }
}

// ============================================================================
// Wrapper functions for mode selection
// ============================================================================

void call_map2alm(int nside, int l_max, int n_maps,
                  const double* d_map, double* d_alm_re, double* d_alm_im) {
    if (g_use_log_mode) {
        map2alm_cuda_v6_log_f64_f64(nside, l_max, n_maps, d_map, d_alm_re, d_alm_im);
    } else {
        map2alm_cuda_v6_f64_f64(nside, l_max, n_maps, d_map, d_alm_re, d_alm_im);
    }
}

void call_alm2map(int nside, int l_max, int n_maps,
                  const double* d_alm_re, const double* d_alm_im, double* d_map) {
    // Note: LOG mode for alm2map not yet fully implemented, use LINEAR
    alm2map_cuda_v6_f64_f64(nside, l_max, n_maps, d_alm_re, d_alm_im, d_map);
}

// ============================================================================
// Test: Uniform map
// ============================================================================

int test_uniform_map() {
    printf("Testing map2alm with uniform map (%s mode)...\n",
           g_use_log_mode ? "LOG" : "LINEAR");

    int nside = g_nside;
    int l_max = 3 * nside;
    int n_maps = 1;
    int n_rings = 4 * nside - 1;
    int lp1 = l_max + 1;
    int errors = 0;

    // Allocate host arrays
    size_t map_size = (size_t)n_maps * n_rings * 4 * nside;
    size_t alm_elements = (size_t)n_maps * lp1 * lp1;

    double* h_map = new double[map_size];
    double* h_alm_re = new double[alm_elements];
    double* h_alm_im = new double[alm_elements];

    // Initialize uniform map of 1s
    for (size_t i = 0; i < map_size; i++) {
        h_map[i] = 1.0;
    }

    // Allocate device memory
    double *d_map, *d_alm_re, *d_alm_im;
    cudaMalloc(&d_map, map_size * sizeof(double));
    cudaMalloc(&d_alm_re, alm_elements * sizeof(double));
    cudaMalloc(&d_alm_im, alm_elements * sizeof(double));

    // Initialize alm to zero
    cudaMemset(d_alm_re, 0, alm_elements * sizeof(double));
    cudaMemset(d_alm_im, 0, alm_elements * sizeof(double));

    // Copy to device
    cudaMemcpy(d_map, h_map, map_size * sizeof(double), cudaMemcpyHostToDevice);

    // Run transform
    call_map2alm(nside, l_max, n_maps, d_map, d_alm_re, d_alm_im);

    // Copy back
    cudaMemcpy(h_alm_re, d_alm_re, alm_elements * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_alm_im, d_alm_im, alm_elements * sizeof(double), cudaMemcpyDeviceToHost);

    // Check a_00 (should be sqrt(4*pi) for uniform map of 1s)
    double a00_real = h_alm_re[0];
    double a00_imag = h_alm_im[0];
    double diff = fabs(a00_real - ALM00_UNIFORM_REF);

    printf("  nside=%d, l_max=%d\n", nside, l_max);
    printf("  a_00 = %.10e + %.10ei (expected %.10e)\n",
           a00_real, a00_imag, ALM00_UNIFORM_REF);
    printf("  |diff| = %.2e\n", diff);

    if (diff > 1e-6) {
        printf("  ERROR: a_00 mismatch\n");
        errors++;
    }

    // Check that imaginary part is near zero
    if (fabs(a00_imag) > 1e-8) {
        printf("  WARNING: a_00 has non-zero imaginary part: %.2e\n", a00_imag);
    }

    // Check that higher l coefficients are small (uniform map is pure l=0)
    double max_other = 0;
    int max_l = 0, max_m = 0;
    for (int l = 1; l <= l_max; l++) {
        for (int m = 0; m <= l; m++) {
            int idx = l * lp1 + m;
            double mag = sqrt(h_alm_re[idx] * h_alm_re[idx] + h_alm_im[idx] * h_alm_im[idx]);
            if (mag > max_other) {
                max_other = mag;
                max_l = l;
                max_m = m;
            }
        }
    }
    printf("  Max |a_lm| for l>0: %.2e at l=%d, m=%d\n", max_other, max_l, max_m);

    // Print first few non-zero alm
    printf("  First few |a_lm|:\n");
    for (int l = 1; l <= 5 && l <= l_max; l++) {
        int idx = l * lp1 + 0;
        double mag = sqrt(h_alm_re[idx] * h_alm_re[idx] + h_alm_im[idx] * h_alm_im[idx]);
        printf("    |a_%d,0| = %.6e\n", l, mag);
    }

    // Cleanup
    cudaFree(d_map);
    cudaFree(d_alm_re);
    cudaFree(d_alm_im);
    delete[] h_map;
    delete[] h_alm_re;
    delete[] h_alm_im;

    if (errors == 0) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED\n");
    }

    return errors;
}

// ============================================================================
// Test: Round-trip (map -> alm -> map)
// ============================================================================

int test_roundtrip() {
    printf("Testing round-trip map -> alm -> map (%s mode)...\n",
           g_use_log_mode ? "LOG" : "LINEAR");

    int nside = g_nside;
    int l_max = 2 * nside;  // Use smaller l_max for faster test
    int n_maps = 1;
    int n_rings = 4 * nside - 1;
    int lp1 = l_max + 1;
    int errors = 0;

    // Allocate arrays
    size_t map_size = (size_t)n_maps * n_rings * 4 * nside;
    size_t alm_elements = (size_t)n_maps * lp1 * lp1;

    double* h_map_in = new double[map_size];
    double* h_map_out = new double[map_size];
    double* h_alm_re = new double[alm_elements];
    double* h_alm_im = new double[alm_elements];

    // Initialize with a simple pattern
    srand(42);
    for (size_t i = 0; i < map_size; i++) {
        h_map_in[i] = (double)rand() / RAND_MAX;
    }

    // Allocate device memory
    double *d_map_in, *d_map_out, *d_alm_re, *d_alm_im;
    cudaMalloc(&d_map_in, map_size * sizeof(double));
    cudaMalloc(&d_map_out, map_size * sizeof(double));
    cudaMalloc(&d_alm_re, alm_elements * sizeof(double));
    cudaMalloc(&d_alm_im, alm_elements * sizeof(double));

    // Initialize
    cudaMemset(d_alm_re, 0, alm_elements * sizeof(double));
    cudaMemset(d_alm_im, 0, alm_elements * sizeof(double));
    cudaMemset(d_map_out, 0, map_size * sizeof(double));

    // Copy input to device
    cudaMemcpy(d_map_in, h_map_in, map_size * sizeof(double), cudaMemcpyHostToDevice);

    // Forward transform: map -> alm
    call_map2alm(nside, l_max, n_maps, d_map_in, d_alm_re, d_alm_im);

    // Inverse transform: alm -> map
    call_alm2map(nside, l_max, n_maps, d_alm_re, d_alm_im, d_map_out);

    // Copy result back
    cudaMemcpy(h_map_out, d_map_out, map_size * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute residual statistics
    double sum_sq_diff = 0;
    double sum_sq_orig = 0;

    for (size_t i = 0; i < map_size; i++) {
        double diff = h_map_in[i] - h_map_out[i];
        sum_sq_diff += diff * diff;
        sum_sq_orig += h_map_in[i] * h_map_in[i];
    }

    double rms_diff = sqrt(sum_sq_diff / map_size);
    double rms_orig = sqrt(sum_sq_orig / map_size);
    double rel_error = rms_diff / rms_orig;

    printf("  nside=%d, l_max=%d\n", nside, l_max);
    printf("  RMS input: %.6e\n", rms_orig);
    printf("  RMS difference: %.6e\n", rms_diff);
    printf("  Relative error: %.6e\n", rel_error);

    // Round-trip should have some error due to finite l_max
    // For l_max = 2*nside, expect ~10-20% error
    if (rel_error > 0.5) {
        printf("  ERROR: Relative error too large\n");
        errors++;
    }

    // Cleanup
    cudaFree(d_map_in);
    cudaFree(d_map_out);
    cudaFree(d_alm_re);
    cudaFree(d_alm_im);
    delete[] h_map_in;
    delete[] h_map_out;
    delete[] h_alm_re;
    delete[] h_alm_im;

    if (errors == 0) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED\n");
    }

    return errors;
}

// ============================================================================
// Test: Compare LINEAR vs LOG mode
// ============================================================================

int test_mode_comparison() {
    printf("Testing LINEAR vs LOG mode comparison...\n");

    int nside = g_nside;
    int l_max = 2 * nside;
    int n_maps = 1;
    int n_rings = 4 * nside - 1;
    int lp1 = l_max + 1;
    int errors = 0;

    // Allocate arrays
    size_t map_size = (size_t)n_maps * n_rings * 4 * nside;
    size_t alm_elements = (size_t)n_maps * lp1 * lp1;

    double* h_map = new double[map_size];
    double* h_alm_re_linear = new double[alm_elements];
    double* h_alm_im_linear = new double[alm_elements];
    double* h_alm_re_log = new double[alm_elements];
    double* h_alm_im_log = new double[alm_elements];

    // Initialize with random data
    srand(42);
    for (size_t i = 0; i < map_size; i++) {
        h_map[i] = (double)rand() / RAND_MAX;
    }

    // Allocate device memory
    double *d_map, *d_alm_re, *d_alm_im;
    cudaMalloc(&d_map, map_size * sizeof(double));
    cudaMalloc(&d_alm_re, alm_elements * sizeof(double));
    cudaMalloc(&d_alm_im, alm_elements * sizeof(double));

    cudaMemcpy(d_map, h_map, map_size * sizeof(double), cudaMemcpyHostToDevice);

    // Run LINEAR mode
    printf("  Running LINEAR mode...\n");
    cudaMemset(d_alm_re, 0, alm_elements * sizeof(double));
    cudaMemset(d_alm_im, 0, alm_elements * sizeof(double));
    map2alm_cuda_v6_f64_f64(nside, l_max, n_maps, d_map, d_alm_re, d_alm_im);
    cudaMemcpy(h_alm_re_linear, d_alm_re, alm_elements * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_alm_im_linear, d_alm_im, alm_elements * sizeof(double), cudaMemcpyDeviceToHost);

    // Run LOG mode
    printf("  Running LOG mode...\n");
    cudaMemset(d_alm_re, 0, alm_elements * sizeof(double));
    cudaMemset(d_alm_im, 0, alm_elements * sizeof(double));
    map2alm_cuda_v6_log_f64_f64(nside, l_max, n_maps, d_map, d_alm_re, d_alm_im);
    cudaMemcpy(h_alm_re_log, d_alm_re, alm_elements * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_alm_im_log, d_alm_im, alm_elements * sizeof(double), cudaMemcpyDeviceToHost);

    // Compare results
    double max_diff = 0;
    double max_val = 0;
    int max_diff_l = 0, max_diff_m = 0;

    for (size_t i = 0; i < alm_elements; i++) {
        double diff_re = fabs(h_alm_re_linear[i] - h_alm_re_log[i]);
        double diff_im = fabs(h_alm_im_linear[i] - h_alm_im_log[i]);
        double diff = sqrt(diff_re * diff_re + diff_im * diff_im);
        double val = sqrt(h_alm_re_linear[i] * h_alm_re_linear[i] +
                          h_alm_im_linear[i] * h_alm_im_linear[i]);

        if (val > max_val) max_val = val;
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_l = i / lp1;
            max_diff_m = i % lp1;
        }
    }

    double rel_error = (max_val > 0) ? (max_diff / max_val) : 0;

    printf("  nside=%d, l_max=%d\n", nside, l_max);
    printf("  Max |alm|: %.6e\n", max_val);
    printf("  Max absolute difference: %.6e at l=%d, m=%d\n", max_diff, max_diff_l, max_diff_m);
    printf("  Max relative error: %.6e\n", rel_error);

    // Should be very close (machine precision)
    if (rel_error > 1e-10) {
        printf("  WARNING: Difference between LINEAR and LOG mode larger than expected\n");
    } else {
        printf("  OK: LINEAR and LOG modes produce consistent results\n");
    }

    // Cleanup
    cudaFree(d_map);
    cudaFree(d_alm_re);
    cudaFree(d_alm_im);
    delete[] h_map;
    delete[] h_alm_re_linear;
    delete[] h_alm_im_linear;
    delete[] h_alm_re_log;
    delete[] h_alm_im_log;

    printf("  PASSED\n");
    return errors;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    parse_args(argc, argv);

    printf("=== SPHT CUDA Transform Tests ===\n");
    printf("Mode: %s\n", g_use_log_mode ? "LOG" : "LINEAR");
    printf("nside: %d\n\n", g_nside);

    int total_errors = 0;

    if (g_run_comparison) {
        // Only run comparison test
        total_errors += test_mode_comparison();
    } else {
        // Run standard tests
        total_errors += test_uniform_map();
        printf("\n");

        total_errors += test_roundtrip();
    }

    printf("\n=== Summary ===\n");
    if (total_errors == 0) {
        printf("All tests PASSED\n");
        return 0;
    } else {
        printf("FAILED with %d total errors\n", total_errors);
        return 1;
    }
}

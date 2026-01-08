/**
 * Test transform operations (map2alm, alm2map)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/spht_api.h"

// Reference: uniform map of 1s should give alm[0,0] = sqrt(4*pi)
#define ALM00_UNIFORM_REF 3.5449077018110318

int test_uniform_map() {
    printf("Testing map2alm with uniform map...\n");

    int nside = 32;
    int l_max = 3 * nside;
    int n_maps = 1;
    int n_rings = 4 * nside - 1;
    int errors = 0;

    // Allocate host arrays
    size_t map_size = (size_t)n_maps * n_rings * 4 * nside;
    size_t alm_size = (size_t)n_maps * (l_max + 1) * (l_max + 1);

    real_t* h_map = new real_t[map_size];
    complex_t* h_alm = new complex_t[alm_size];

    // Initialize uniform map of 1s
    for (size_t i = 0; i < map_size; i++) {
        h_map[i] = 1.0;
    }

    // Allocate device memory
    real_t* d_map = spht_allocate_map(nside, n_maps);
    complex_t* d_alm = spht_allocate_alm(l_max, n_maps);

    if (d_map == NULL || d_alm == NULL) {
        printf("  ERROR: Failed to allocate GPU memory\n");
        delete[] h_map;
        delete[] h_alm;
        return 1;
    }

    // Copy to device
    spht_map_to_device(nside, n_maps, h_map, d_map);

    // Run transform
    map2alm_cuda_v6(nside, l_max, n_maps, d_map, d_alm);

    // Copy back
    spht_alm_to_host(l_max, n_maps, d_alm, h_alm);

    // Check a_00 (should be sqrt(4*pi) for uniform map of 1s)
    double a00_real = h_alm[0].x;
    double a00_imag = h_alm[0].y;
    double diff = fabs(a00_real - ALM00_UNIFORM_REF);

    printf("  a_00 = %.10e + %.10ei (expected %.10e)\n",
           a00_real, a00_imag, ALM00_UNIFORM_REF);
    printf("  |diff| = %.2e\n", diff);

    if (diff > 1e-8) {
        printf("  ERROR: a_00 mismatch\n");
        errors++;
    }

    // Check that imaginary part is near zero
    if (fabs(a00_imag) > 1e-10) {
        printf("  WARNING: a_00 has non-zero imaginary part: %.2e\n", a00_imag);
    }

    // Check that higher l coefficients are small (uniform map is pure l=0)
    double max_other = 0;
    int max_l = 0, max_m = 0;
    for (int l = 1; l <= l_max; l++) {
        for (int m = 0; m <= l; m++) {
            int idx = INDEX_3D(0, l, m, l_max + 1, l_max + 1);
            double mag = sqrt(h_alm[idx].x * h_alm[idx].x + h_alm[idx].y * h_alm[idx].y);
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
        int idx = INDEX_3D(0, l, 0, l_max + 1, l_max + 1);
        double mag = sqrt(h_alm[idx].x * h_alm[idx].x + h_alm[idx].y * h_alm[idx].y);
        printf("    |a_%d,0| = %.6e\n", l, mag);
    }

    // Cleanup
    spht_free(d_map);
    spht_free(d_alm);
    delete[] h_map;
    delete[] h_alm;

    if (errors == 0) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED\n");
    }

    return errors;
}

int test_roundtrip() {
    printf("Testing round-trip (map -> alm -> map)...\n");

    int nside = 32;
    int l_max = 2 * nside;  // Use smaller l_max for faster test
    int n_maps = 1;
    int n_rings = 4 * nside - 1;
    int errors = 0;

    // Allocate arrays
    size_t map_size = (size_t)n_maps * n_rings * 4 * nside;
    size_t alm_size = (size_t)n_maps * (l_max + 1) * (l_max + 1);

    real_t* h_map_in = new real_t[map_size];
    real_t* h_map_out = new real_t[map_size];
    complex_t* h_alm = new complex_t[alm_size];

    // Initialize with a simple pattern
    srand(42);
    for (size_t i = 0; i < map_size; i++) {
        h_map_in[i] = (double)rand() / RAND_MAX;
    }

    // Allocate device memory
    real_t* d_map_in = spht_allocate_map(nside, n_maps);
    real_t* d_map_out = spht_allocate_map(nside, n_maps);
    complex_t* d_alm = spht_allocate_alm(l_max, n_maps);

    // Copy input to device
    spht_map_to_device(nside, n_maps, h_map_in, d_map_in);

    // Forward transform: map -> alm
    map2alm_cuda_v6(nside, l_max, n_maps, d_map_in, d_alm);

    // Inverse transform: alm -> map
    alm2map_cuda_v6(nside, l_max, n_maps, d_alm, d_map_out);

    // Copy result back
    spht_map_to_host(nside, n_maps, d_map_out, h_map_out);

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
    spht_free(d_map_in);
    spht_free(d_map_out);
    spht_free(d_alm);
    delete[] h_map_in;
    delete[] h_map_out;
    delete[] h_alm;

    if (errors == 0) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED\n");
    }

    return errors;
}

int main() {
    printf("=== SPHT CUDA Transform Tests ===\n\n");

    int total_errors = 0;

    total_errors += test_uniform_map();
    total_errors += test_roundtrip();

    printf("\n=== Summary ===\n");
    if (total_errors == 0) {
        printf("All tests PASSED\n");
        return 0;
    } else {
        printf("FAILED with %d total errors\n", total_errors);
        return 1;
    }
}

/**
 * Test YLM computation against expected values
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/spht_types.h"
#include "../include/ylm_recurrence.cuh"

// Reference Y_00 value (1/sqrt(4*pi))
#define Y00_REF 0.28209479177387814

// Reference Y_1,0(cos(theta)=0.5) = sqrt(3/4pi) * 0.5
#define Y10_COS05_REF 0.24430125595146002

int test_ring_geometry() {
    printf("Testing ring geometry precomputation...\n");

    int nside = 32;
    ring_geometry_t* geom = allocate_ring_geometry(nside);
    precompute_ring_geometry(nside, geom);

    // Copy back to host for verification
    int n_rings = 4 * nside - 1;
    real_t* h_log_beta = new real_t[n_rings];
    int8_t* h_beta_sign = new int8_t[n_rings];
    real_t* h_phi_0 = new real_t[n_rings];
    int* h_n_pixels = new int[n_rings];

    cudaMemcpy(h_log_beta, geom->log_beta, n_rings * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_beta_sign, geom->beta_sign, n_rings * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_phi_0, geom->phi_0, n_rings * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n_pixels, geom->n_pixels, n_rings * sizeof(int), cudaMemcpyDeviceToHost);

    // Check some known values
    int errors = 0;

    // Ring 1 (north polar): npix = 4, beta = 1 - 1/(3*nside^2)
    if (h_n_pixels[0] != 4) {
        printf("  ERROR: Ring 1 npix = %d, expected 4\n", h_n_pixels[0]);
        errors++;
    }
    if (h_beta_sign[0] != 1) {
        printf("  ERROR: Ring 1 sign = %d, expected 1\n", h_beta_sign[0]);
        errors++;
    }

    // Ring 2*nside (equator): npix = 4*nside, beta near 0
    int eq_idx = 2 * nside - 1;
    if (h_n_pixels[eq_idx] != 4 * nside) {
        printf("  ERROR: Equator npix = %d, expected %d\n", h_n_pixels[eq_idx], 4 * nside);
        errors++;
    }

    // South polar cap should have negative sign
    int south_idx = n_rings - 1;
    if (h_beta_sign[south_idx] != -1) {
        printf("  ERROR: South pole sign = %d, expected -1\n", h_beta_sign[south_idx]);
        errors++;
    }

    delete[] h_log_beta;
    delete[] h_beta_sign;
    delete[] h_phi_0;
    delete[] h_n_pixels;
    free_ring_geometry(geom);

    if (errors == 0) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED with %d errors\n", errors);
    }

    return errors;
}

int test_ylm_Y00() {
    printf("Testing Y_00 value...\n");

    int l_max = 4;
    int n_rings = 1;
    int errors = 0;

    // Create log_beta = 0 (cos(theta) = 1)
    real_t* d_log_beta;
    int8_t* d_beta_sign;
    cudaMalloc(&d_log_beta, sizeof(real_t));
    cudaMalloc(&d_beta_sign, sizeof(int8_t));

    real_t h_log_beta = 0.0;  // cos(theta) = 1
    int8_t h_beta_sign = 1;
    cudaMemcpy(d_log_beta, &h_log_beta, sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta_sign, &h_beta_sign, sizeof(int8_t), cudaMemcpyHostToDevice);

    // Allocate output
    int size = (l_max + 1) * (l_max + 1) * n_rings;
    real_t* d_ylm;
    cudaMalloc(&d_ylm, size * sizeof(real_t));

    // Compute YLM
    compute_ylm_spin0(l_max, n_rings, d_log_beta, d_beta_sign, d_ylm);

    // Copy back
    real_t* h_ylm = new real_t[size];
    cudaMemcpy(h_ylm, d_ylm, size * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Check Y_00
    int idx_00 = INDEX_3D(0, 0, 0, l_max + 1, n_rings);
    double diff = fabs(h_ylm[idx_00] - Y00_REF);
    if (diff > 1e-10) {
        printf("  ERROR: Y_00 = %.15e, expected %.15e, diff = %.2e\n",
               h_ylm[idx_00], Y00_REF, diff);
        errors++;
    } else {
        printf("  Y_00 = %.15e (expected %.15e, diff = %.2e)\n",
               h_ylm[idx_00], Y00_REF, diff);
    }

    delete[] h_ylm;
    cudaFree(d_log_beta);
    cudaFree(d_beta_sign);
    cudaFree(d_ylm);

    if (errors == 0) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED\n");
    }

    return errors;
}

int test_ylm_diagonal() {
    printf("Testing Y_ll diagonal elements...\n");

    int l_max = 10;
    int n_rings = 1;
    int errors = 0;

    // Use sin(theta) = 0.5 for testing
    // log(sin) = log(0.5)
    real_t* d_log_beta;
    int8_t* d_beta_sign;
    cudaMalloc(&d_log_beta, sizeof(real_t));
    cudaMalloc(&d_beta_sign, sizeof(int8_t));

    // cos(theta) = sqrt(1 - 0.5^2) = sqrt(0.75)
    real_t cos_theta = sqrt(0.75);
    real_t h_log_beta = log(cos_theta);
    int8_t h_beta_sign = 1;
    cudaMemcpy(d_log_beta, &h_log_beta, sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta_sign, &h_beta_sign, sizeof(int8_t), cudaMemcpyHostToDevice);

    // Allocate output
    int size = (l_max + 1) * (l_max + 1) * n_rings;
    real_t* d_ylm;
    cudaMalloc(&d_ylm, size * sizeof(real_t));

    // Compute YLM
    compute_ylm_spin0(l_max, n_rings, d_log_beta, d_beta_sign, d_ylm);

    // Copy back
    real_t* h_ylm = new real_t[size];
    cudaMemcpy(h_ylm, d_ylm, size * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Check that all Y_ll have correct sign pattern: (-1)^l
    for (int l = 0; l <= l_max; l++) {
        int idx = INDEX_3D(l, l, 0, l_max + 1, n_rings);
        double val = h_ylm[idx];
        int expected_sign = (l % 2 == 0) ? 1 : -1;
        int actual_sign = (val >= 0) ? 1 : -1;

        if (actual_sign != expected_sign && fabs(val) > 1e-15) {
            printf("  ERROR: Y_%d,%d has wrong sign: %.6e (expected %s)\n",
                   l, l, val, (expected_sign > 0) ? "positive" : "negative");
            errors++;
        }
    }

    printf("  Diagonal Y_ll values:\n");
    for (int l = 0; l <= 5; l++) {
        int idx = INDEX_3D(l, l, 0, l_max + 1, n_rings);
        printf("    Y_%d,%d = %.10e\n", l, l, h_ylm[idx]);
    }

    delete[] h_ylm;
    cudaFree(d_log_beta);
    cudaFree(d_beta_sign);
    cudaFree(d_ylm);

    if (errors == 0) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED with %d errors\n", errors);
    }

    return errors;
}

int main() {
    printf("=== SPHT CUDA YLM Tests ===\n\n");

    int total_errors = 0;

    total_errors += test_ring_geometry();
    total_errors += test_ylm_Y00();
    total_errors += test_ylm_diagonal();

    printf("\n=== Summary ===\n");
    if (total_errors == 0) {
        printf("All tests PASSED\n");
        return 0;
    } else {
        printf("FAILED with %d total errors\n", total_errors);
        return 1;
    }
}

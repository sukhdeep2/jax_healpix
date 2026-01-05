/**
 * Unit tests for FFT-based Gm computation
 *
 * Tests that FFT-based Gm matches naive DFT computation
 * and that map2alm_v2 matches map2alm (original)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../include/spht_types.h"
#include "../include/spht_api.h"
#include "../include/fft_gm.cuh"
#include "../include/ylm_recurrence.cuh"

// External declarations
extern "C" void map2alm_cuda(int nside, int l_max, int n_maps,
                              const real_t* map_in, complex_t* alm_out);
extern "C" void map2alm_cuda_v2(int nside, int l_max, int n_maps,
                                 const real_t* map_in, complex_t* alm_out);

/**
 * Compute Gm using naive DFT (reference implementation)
 * Gm[m] = sum_j map[j] * exp(-i * m * phi_j)
 */
void compute_gm_naive_host(int npix, real_t phi_0, const real_t* map,
                           int l_max, complex_t* Gm) {
    for (int m = 0; m <= l_max; m++) {
        real_t sum_real = 0.0;
        real_t sum_imag = 0.0;

        for (int j = 0; j < npix; j++) {
            real_t phi_j = phi_0 + 2.0 * PI * j / npix;
            real_t angle = -m * phi_j;
            sum_real += map[j] * cos(angle);
            sum_imag += map[j] * sin(angle);
        }

        Gm[m].x = sum_real;
        Gm[m].y = sum_imag;
    }
}

/**
 * Test that FFT-based Gm matches naive DFT computation
 */
bool test_fft_gm_accuracy() {
    printf("\n=== Test: FFT-based Gm accuracy ===\n");

    int nside = 64;
    int l_max = 3 * nside;
    int n_rings = 4 * nside - 1;
    int lp1 = l_max + 1;

    // Allocate ring geometry
    ring_geometry_t* geom = allocate_ring_geometry(nside);
    precompute_ring_geometry(nside, geom);

    // Copy geometry to host
    real_t* h_phi_0 = new real_t[n_rings];
    int* h_n_pixels = new int[n_rings];
    cudaMemcpy(h_phi_0, geom->phi_0, n_rings * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n_pixels, geom->n_pixels, n_rings * sizeof(int), cudaMemcpyDeviceToHost);

    // Create random test map
    int n_maps = 1;
    size_t map_size = (size_t)n_maps * n_rings * 4 * nside;
    real_t* h_map = new real_t[map_size];

    srand(42);
    for (size_t i = 0; i < map_size; i++) {
        h_map[i] = (real_t)rand() / RAND_MAX - 0.5;
    }

    // Copy map to device
    real_t* d_map;
    cudaMalloc(&d_map, map_size * sizeof(real_t));
    cudaMemcpy(d_map, h_map, map_size * sizeof(real_t), cudaMemcpyHostToDevice);

    // Allocate Gm output on device
    complex_t* d_Gm;
    cudaMalloc(&d_Gm, (size_t)n_maps * n_rings * lp1 * sizeof(complex_t));

    // Compute Gm using FFT
    compute_gm_all_rings(nside, l_max, n_maps, d_map, geom, d_Gm);
    cudaDeviceSynchronize();

    // Copy result to host
    complex_t* h_Gm_fft = new complex_t[n_rings * lp1];
    cudaMemcpy(h_Gm_fft, d_Gm, n_rings * lp1 * sizeof(complex_t), cudaMemcpyDeviceToHost);

    // Compute Gm using naive DFT (for a few rings)
    int test_rings[] = {0, nside/2, nside-1, nside, 2*nside-1, 2*nside, 3*nside-1, n_rings-1};
    int n_test = 8;

    real_t max_error = 0.0;
    bool passed = true;

    for (int t = 0; t < n_test; t++) {
        int r = test_rings[t];
        if (r >= n_rings) continue;

        int npix = h_n_pixels[r];
        real_t phi_0 = h_phi_0[r];

        // Compute naive DFT
        complex_t* h_Gm_naive = new complex_t[lp1];
        compute_gm_naive_host(npix, phi_0, h_map + r * 4 * nside, l_max, h_Gm_naive);

        // Compare
        for (int m = 0; m < min(npix, lp1); m++) {
            real_t err_real = fabs(h_Gm_fft[r * lp1 + m].x - h_Gm_naive[m].x);
            real_t err_imag = fabs(h_Gm_fft[r * lp1 + m].y - h_Gm_naive[m].y);
            real_t err = sqrt(err_real * err_real + err_imag * err_imag);

            if (err > max_error) max_error = err;

            if (err > 1e-10) {
                printf("  Ring %d, m=%d: FFT=(%.6e, %.6e), Naive=(%.6e, %.6e), err=%.6e\n",
                       r, m, h_Gm_fft[r * lp1 + m].x, h_Gm_fft[r * lp1 + m].y,
                       h_Gm_naive[m].x, h_Gm_naive[m].y, err);
                if (err > 1e-8) {
                    passed = false;
                }
            }
        }

        delete[] h_Gm_naive;
    }

    printf("  Max error: %.6e\n", max_error);
    printf("  Result: %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    delete[] h_map;
    delete[] h_phi_0;
    delete[] h_n_pixels;
    delete[] h_Gm_fft;
    cudaFree(d_map);
    cudaFree(d_Gm);
    free_ring_geometry(geom);

    return passed;
}

/**
 * Test full map2alm_v2 against map2alm (original)
 */
bool test_map2alm_v2_accuracy() {
    printf("\n=== Test: map2alm_v2 accuracy ===\n");

    int nside = 64;
    int l_max = 3 * nside;
    int n_maps = 1;
    int n_rings = 4 * nside - 1;
    int lp1 = l_max + 1;

    // Allocate device memory
    real_t* d_map = spht_allocate_map(nside, n_maps);
    complex_t* d_alm_v1 = spht_allocate_alm(l_max, n_maps);
    complex_t* d_alm_v2 = spht_allocate_alm(l_max, n_maps);

    if (!d_map || !d_alm_v1 || !d_alm_v2) {
        printf("  Failed to allocate memory\n");
        return false;
    }

    // Create random test map
    size_t map_size = (size_t)n_maps * n_rings * 4 * nside;
    real_t* h_map = new real_t[map_size];

    srand(42);
    for (size_t i = 0; i < map_size; i++) {
        h_map[i] = (real_t)rand() / RAND_MAX - 0.5;
    }

    // Copy to device
    spht_map_to_device(nside, n_maps, h_map, d_map);

    // Run both versions
    printf("  Running map2alm (original)...\n");
    map2alm_cuda(nside, l_max, n_maps, d_map, d_alm_v1);
    cudaDeviceSynchronize();

    printf("  Running map2alm_v2 (optimized)...\n");
    map2alm_cuda_v2(nside, l_max, n_maps, d_map, d_alm_v2);
    cudaDeviceSynchronize();

    // Copy results to host
    size_t alm_size = (size_t)n_maps * lp1 * lp1;
    complex_t* h_alm_v1 = new complex_t[alm_size];
    complex_t* h_alm_v2 = new complex_t[alm_size];

    spht_alm_to_host(l_max, n_maps, d_alm_v1, h_alm_v1);
    spht_alm_to_host(l_max, n_maps, d_alm_v2, h_alm_v2);

    // Compare results
    real_t max_error = 0.0;
    real_t max_rel_error = 0.0;
    int max_error_l = 0, max_error_m = 0;
    bool passed = true;

    for (int l = 0; l <= l_max; l++) {
        for (int m = 0; m <= l; m++) {
            int idx = l * lp1 + m;

            real_t err_real = fabs(h_alm_v1[idx].x - h_alm_v2[idx].x);
            real_t err_imag = fabs(h_alm_v1[idx].y - h_alm_v2[idx].y);
            real_t err = sqrt(err_real * err_real + err_imag * err_imag);

            real_t mag = sqrt(h_alm_v1[idx].x * h_alm_v1[idx].x +
                              h_alm_v1[idx].y * h_alm_v1[idx].y);
            real_t rel_err = (mag > 1e-15) ? err / mag : err;

            if (err > max_error) {
                max_error = err;
                max_error_l = l;
                max_error_m = m;
            }
            if (rel_err > max_rel_error) {
                max_rel_error = rel_err;
            }

            // Fail if relative error too high (for non-tiny values)
            if (mag > 1e-10 && rel_err > 0.01) {
                if (passed) {
                    printf("  First failure at l=%d, m=%d:\n", l, m);
                    printf("    v1: (%.6e, %.6e)\n", h_alm_v1[idx].x, h_alm_v1[idx].y);
                    printf("    v2: (%.6e, %.6e)\n", h_alm_v2[idx].x, h_alm_v2[idx].y);
                    printf("    rel_err: %.6e\n", rel_err);
                }
                passed = false;
            }
        }
    }

    printf("  Max absolute error: %.6e at l=%d, m=%d\n", max_error, max_error_l, max_error_m);
    printf("  Max relative error: %.6e\n", max_rel_error);
    printf("  Result: %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    delete[] h_map;
    delete[] h_alm_v1;
    delete[] h_alm_v2;
    spht_free(d_map);
    spht_free(d_alm_v1);
    spht_free(d_alm_v2);

    return passed;
}

/**
 * Simple performance benchmark
 */
void benchmark_map2alm() {
    printf("\n=== Benchmark: map2alm vs map2alm_v2 ===\n");

    int nsides[] = {64, 128, 256};
    int n_nsides = 3;
    int n_warmup = 2;
    int n_runs = 5;

    for (int i = 0; i < n_nsides; i++) {
        int nside = nsides[i];
        int l_max = 3 * nside;
        int n_maps = 1;
        int n_rings = 4 * nside - 1;

        // Allocate
        real_t* d_map = spht_allocate_map(nside, n_maps);
        complex_t* d_alm = spht_allocate_alm(l_max, n_maps);

        if (!d_map || !d_alm) {
            printf("  nside=%d: Failed to allocate\n", nside);
            continue;
        }

        // Initialize with random data
        size_t map_size = (size_t)n_maps * n_rings * 4 * nside;
        real_t* h_map = new real_t[map_size];
        for (size_t j = 0; j < map_size; j++) h_map[j] = (real_t)rand() / RAND_MAX;
        spht_map_to_device(nside, n_maps, h_map, d_map);
        delete[] h_map;

        // Benchmark v1
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int w = 0; w < n_warmup; w++) {
            map2alm_cuda(nside, l_max, n_maps, d_map, d_alm);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int r = 0; r < n_runs; r++) {
            map2alm_cuda(nside, l_max, n_maps, d_map, d_alm);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_v1_ms;
        cudaEventElapsedTime(&time_v1_ms, start, stop);
        time_v1_ms /= n_runs;

        // Benchmark v2
        for (int w = 0; w < n_warmup; w++) {
            map2alm_cuda_v2(nside, l_max, n_maps, d_map, d_alm);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int r = 0; r < n_runs; r++) {
            map2alm_cuda_v2(nside, l_max, n_maps, d_map, d_alm);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_v2_ms;
        cudaEventElapsedTime(&time_v2_ms, start, stop);
        time_v2_ms /= n_runs;

        printf("  nside=%d: v1=%.2f ms, v2=%.2f ms, speedup=%.2fx\n",
               nside, time_v1_ms, time_v2_ms, time_v1_ms / time_v2_ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        spht_free(d_map);
        spht_free(d_alm);
    }
}

int main() {
    printf("FFT Gm Unit Tests\n");
    printf("=================\n");

    int n_passed = 0;
    int n_tests = 0;

    // Test FFT Gm accuracy
    n_tests++;
    if (test_fft_gm_accuracy()) n_passed++;

    // Test map2alm_v2 accuracy
    n_tests++;
    if (test_map2alm_v2_accuracy()) n_passed++;

    // Benchmark
    benchmark_map2alm();

    printf("\n=================\n");
    printf("Tests passed: %d/%d\n", n_passed, n_tests);

    return (n_passed == n_tests) ? 0 : 1;
}

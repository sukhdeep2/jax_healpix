/**
 * map2alm_v2: Optimized Spherical Harmonic Analysis using cuFFT and cuBLAS
 *
 * Algorithm (following OPTIMIZATION_PLAN.md exactly):
 *   1. Compute Gm for ALL rings using FFT (O(n_pixels * log(n_pixels)))
 *   2. For each m value:
 *      a. Compute Ylm column for north rings (2*nside including equator)
 *      b. Apply (-1)^(l+m) symmetry for south rings (zeros equator to prevent double-count)
 *      c. Use cuBLAS DGEMV: alm[:, m] = Ylm @ Gm[:, m]
 *   3. Apply pixel area normalization
 *
 * Key insight from plan:
 *   - Process ring pairs (north, south) together
 *   - North ring i pairs with South ring (4*nside-2-i)
 *   - Ylm_south[l,m] = (-1)^(l+m) * Ylm_north[l,m]
 */

#include <cublas_v2.h>
#include <cufft.h>
#include "../include/spht_types.h"
#include "../include/fft_gm.cuh"
#include "../include/ylm_recurrence.cuh"
#include <stdio.h>

// ============================================================================
// Kernel: Compute Ylm column for fixed m, all rings (with N-S symmetry)
// ============================================================================

/**
 * Compute Ylm column for fixed m across ALL rings
 *
 * Uses north-south symmetry:
 * - Compute Ylm for north rings (0 to 2*nside-1) using compute_ylm_spin0
 * - Apply (-1)^(l+m) for south rings
 * - Zero out equator in south to prevent double-counting
 *
 * Output layout: Ylm_col[r * n_l + (l-m)] in column-major for cuBLAS
 * where n_l = l_max - m + 1
 */
__global__ void compute_ylm_column_with_symmetry_kernel(
    int m,                        // Fixed m value
    int l_max,
    int n_rings,                  // Total rings = 4*nside - 1
    int nside,
    const real_t* ylm_north,      // [lp1, lp1, n_north_rings] from compute_ylm_spin0
    real_t* Ylm_col               // [n_rings, n_l] column-major output
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;  // Ring index (0 to n_rings-1)
    int l_idx = blockIdx.y * blockDim.y + threadIdx.y;  // l - m

    if (r >= n_rings) return;

    int l = m + l_idx;
    if (l > l_max) return;

    int lp1 = l_max + 1;
    int n_l = l_max - m + 1;
    int n_north_rings = 2 * nside;  // Including equator

    real_t ylm_val = 0.0;

    if (r < n_north_rings) {
        // North ring (or equator): use directly from ylm_north
        // ylm_north layout: [l, m, r] at index l * lp1 * n_north_rings + m * n_north_rings + r
        int src_idx = l * lp1 * n_north_rings + m * n_north_rings + r;
        ylm_val = ylm_north[src_idx];
    } else {
        // South ring: apply (-1)^(l+m) symmetry
        // South ring r corresponds to north ring: 4*nside - 2 - r
        int north_r = 4 * nside - 2 - r;

        // Equator check: if north_r == 2*nside - 1, this is the equator
        // We zero it out for south contribution to prevent double-counting
        if (north_r == 2 * nside - 1) {
            ylm_val = 0.0;
        } else if (north_r >= 0 && north_r < n_north_rings) {
            int src_idx = l * lp1 * n_north_rings + m * n_north_rings + north_r;
            real_t ylm_north_val = ylm_north[src_idx];

            // Apply (-1)^(l+m) factor
            int sign = ((l + m) % 2 == 0) ? 1 : -1;
            ylm_val = sign * ylm_north_val;
        }
    }

    // Output in column-major for cuBLAS: Ylm_col[l_idx, r] at index r * n_l + l_idx
    int dst_idx = r * n_l + l_idx;
    Ylm_col[dst_idx] = ylm_val;
}

// ============================================================================
// cuBLAS Accumulation
// ============================================================================

/**
 * Accumulate alm contributions using cuBLAS DGEMV
 *
 * For fixed m:
 *   alm[t, m:l_max+1, m] = Ylm[:, m:l_max+1]^T @ Gm[t, :, m]
 *
 * Matrix: Ylm_col[n_rings, n_l] in column-major => treated as [n_l, n_rings] row-major
 * Vector: Gm_col[n_rings] (extracted for this m)
 * Result: alm_col[n_l]
 *
 * Since Ylm is real and Gm is complex:
 *   alm.real = Ylm @ Gm.real
 *   alm.imag = Ylm @ Gm.imag
 */
void accumulate_alm_for_m(
    cublasHandle_t handle,
    int l_max,
    int n_rings,
    int n_maps,
    int m,
    const real_t* Ylm_col,        // [n_rings, n_l] column-major
    const complex_t* Gm,          // [n_maps, n_rings, l_max+1]
    complex_t* alm,               // [n_maps, l_max+1, l_max+1]
    real_t* Gm_real_col,          // Preallocated [n_rings]
    real_t* Gm_imag_col,          // Preallocated [n_rings]
    real_t* alm_real_tmp,         // Preallocated [l_max+1]
    real_t* alm_imag_tmp          // Preallocated [l_max+1]
) {
    int n_l = l_max - m + 1;
    int lp1 = l_max + 1;
    double alpha = 1.0;
    double beta = 0.0;
    cublasStatus_t status;

    for (int t = 0; t < n_maps; t++) {
        // Extract Gm[:, m] for this map
        // Gm layout: [n_maps, n_rings, l_max+1]
        // Gm[t, r, m] at offset: t * n_rings * lp1 + r * lp1 + m
        const complex_t* Gm_base = Gm + t * n_rings * lp1;

        // Extract Gm column using cuBLAS Dcopy with stride
        // Real part: stride is 2*lp1 (skip lp1 complex values = 2*lp1 doubles between rings)
        // Wait, Gm[r, m] means consecutive r's are lp1 apart
        // So Gm[0, m], Gm[1, m], ... are at offsets m, lp1+m, 2*lp1+m, ...
        status = cublasDcopy(handle, n_rings,
            (const double*)Gm_base + m * 2,      // Start at Gm[0, m].x
            lp1 * 2,                              // Stride in doubles
            Gm_real_col, 1);

        status = cublasDcopy(handle, n_rings,
            (const double*)Gm_base + m * 2 + 1,  // Start at Gm[0, m].y
            lp1 * 2,
            Gm_imag_col, 1);

        // DGEMV: y = alpha * A * x + beta * y
        //
        // Storage: Ylm_col[r * n_l + l_idx] means A[l_idx, r] at l_idx + r * n_l
        // This is column-major with dimensions (n_l rows, n_rings cols), lda = n_l
        //
        // We want: alm[l_idx] = sum_r Ylm[l_idx, r] * Gm[r]
        // This is y = A * x with A being (n_l, n_rings)
        // Use CUBLAS_OP_N, m = n_l, n = n_rings, lda = n_l

        // Real part: alm.real = Ylm @ Gm.real
        status = cublasDgemv(handle, CUBLAS_OP_N,
            n_l, n_rings,           // Matrix dimensions (rows, cols)
            &alpha,
            Ylm_col, n_l,           // Ylm matrix, lda = n_l
            Gm_real_col, 1,
            &beta,
            alm_real_tmp, 1);

        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS DGEMV failed (real): %d\n", status);
        }

        // Imag part: alm.imag = Ylm @ Gm.imag
        status = cublasDgemv(handle, CUBLAS_OP_N,
            n_l, n_rings,
            &alpha,
            Ylm_col, n_l,
            Gm_imag_col, 1,
            &beta,
            alm_imag_tmp, 1);

        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS DGEMV failed (imag): %d\n", status);
        }

        // Copy results to alm array
        // alm layout: [n_maps, l_max+1, l_max+1]
        // alm[t, l, m] at offset: t * lp1 * lp1 + l * lp1 + m
        // For l = m to l_max, we write to alm[t, m, m], alm[t, m+1, m], ..., alm[t, l_max, m]
        // Stride between consecutive l's is lp1 (in complex units) = 2*lp1 (in doubles)
        complex_t* alm_base = alm + t * lp1 * lp1 + m * lp1 + m;

        status = cublasDcopy(handle, n_l,
            alm_real_tmp, 1,
            (double*)alm_base, 2 * lp1);

        status = cublasDcopy(handle, n_l,
            alm_imag_tmp, 1,
            (double*)alm_base + 1, 2 * lp1);
    }
}

// ============================================================================
// Pixel Area Normalization Kernel
// ============================================================================

__global__ void scale_alm_kernel(
    int n_maps,
    int lp1,
    real_t pix_area,
    complex_t* alm  // [n_maps, lp1, lp1]
) {
    int t = blockIdx.x;
    int l = blockIdx.y * blockDim.x + threadIdx.x;
    int m = blockIdx.z * blockDim.y + threadIdx.y;

    if (t >= n_maps || l >= lp1 || m > l) return;

    int idx = t * lp1 * lp1 + l * lp1 + m;
    alm[idx].x *= pix_area;
    alm[idx].y *= pix_area;
}

// ============================================================================
// Full map2alm_v2 Implementation
// ============================================================================

extern "C"
void map2alm_cuda_v2(int nside, int l_max, int n_maps,
                     const real_t* map_in, complex_t* alm_out) {
    int n_rings = 4 * nside - 1;
    int n_north_rings = 2 * nside;  // Including equator
    int lp1 = l_max + 1;

    // 1. Initialize alm to zero
    CUDA_CHECK(cudaMemset(alm_out, 0, (size_t)n_maps * lp1 * lp1 * sizeof(complex_t)));

    // 2. Compute ring geometry
    ring_geometry_t* geom = allocate_ring_geometry(nside);
    precompute_ring_geometry(nside, geom);

    // 3. Allocate Gm buffer for ALL rings
    complex_t* Gm;
    CUDA_CHECK(cudaMalloc(&Gm, (size_t)n_maps * n_rings * lp1 * sizeof(complex_t)));

    // 4. Compute Gm for ALL rings using FFT
    compute_gm_all_rings(nside, l_max, n_maps, map_in, geom, Gm);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Compute Ylm for NORTH rings only (we'll apply symmetry for south)
    //    Using the proven compute_ylm_spin0 function from ylm_recurrence.cu
    real_t* ylm_north;
    CUDA_CHECK(cudaMalloc(&ylm_north, (size_t)lp1 * lp1 * n_north_rings * sizeof(real_t)));
    compute_ylm_spin0(l_max, n_north_rings, geom->log_beta, geom->beta_sign, ylm_north);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Allocate Ylm column buffer for ALL rings (reused for each m)
    real_t* Ylm_col;
    CUDA_CHECK(cudaMalloc(&Ylm_col, (size_t)n_rings * lp1 * sizeof(real_t)));

    // 7. Allocate working buffers for cuBLAS (reused for each m)
    real_t* Gm_real_col;
    real_t* Gm_imag_col;
    real_t* alm_real_tmp;
    real_t* alm_imag_tmp;
    CUDA_CHECK(cudaMalloc(&Gm_real_col, n_rings * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&Gm_imag_col, n_rings * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&alm_real_tmp, lp1 * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&alm_imag_tmp, lp1 * sizeof(real_t)));

    // 8. cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS initialization failed: %d\n", status);
        cudaFree(Gm);
        cudaFree(ylm_north);
        cudaFree(Ylm_col);
        cudaFree(Gm_real_col);
        cudaFree(Gm_imag_col);
        cudaFree(alm_real_tmp);
        cudaFree(alm_imag_tmp);
        free_ring_geometry(geom);
        return;
    }

    // 9. Process each m value
    for (int m = 0; m <= l_max; m++) {
        int n_l = l_max - m + 1;

        // 9a. Compute Ylm column for ALL rings with N-S symmetry
        dim3 block(16, 16);
        dim3 grid(CEILDIV(n_rings, 16), CEILDIV(n_l, 16));

        compute_ylm_column_with_symmetry_kernel<<<grid, block>>>(
            m, l_max, n_rings, nside, ylm_north, Ylm_col
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 9b. Accumulate to alm using cuBLAS GEMV
        accumulate_alm_for_m(handle, l_max, n_rings, n_maps, m,
                              Ylm_col, Gm, alm_out,
                              Gm_real_col, Gm_imag_col,
                              alm_real_tmp, alm_imag_tmp);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 10. Apply pixel area normalization
    real_t pix_area = 4.0 * PI / (12.0 * nside * nside);

    dim3 block_scale(16, 16);
    dim3 grid_scale(n_maps, CEILDIV(lp1, 16), CEILDIV(lp1, 16));
    scale_alm_kernel<<<grid_scale, block_scale>>>(n_maps, lp1, pix_area, alm_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 11. Cleanup
    cublasDestroy(handle);
    cudaFree(Gm);
    cudaFree(ylm_north);
    cudaFree(Ylm_col);
    cudaFree(Gm_real_col);
    cudaFree(Gm_imag_col);
    cudaFree(alm_real_tmp);
    cudaFree(alm_imag_tmp);
    free_ring_geometry(geom);
}

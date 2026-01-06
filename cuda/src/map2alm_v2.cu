/**
 * map2alm_v2: Optimized Spherical Harmonic Analysis using cuFFT and cuBLAS
 *
 * Algorithm:
 *   1. Compute Gm for ALL rings using FFT (O(n_pixels * log(n_pixels)))
 *   2. Process rings in batches to limit memory:
 *      a. Compute Ylm for batch of north rings
 *      b. Apply (-1)^(l+m) symmetry for corresponding south rings
 *      c. Accumulate to alm using cuBLAS DGEMM (batch all m values together)
 *   3. Apply pixel area normalization
 *
 * Memory optimization:
 *   - Ylm computed in batches of RING_BATCH_SIZE rings
 *   - Uses DGEMM instead of per-m DGEMV for better GPU utilization
 */

#include <cublas_v2.h>
#include <cufft.h>
#include "../include/spht_types.h"
#include "../include/fft_gm.cuh"
#include "../include/ylm_recurrence.cuh"
#include <stdio.h>

// Batch size for ring processing - larger reduces sync overhead
// Memory per batch: ~2 * lp1^2 * batch_size * 8 bytes
// For nside=256, lp1=769: ~10 MB per ring in batch
// For nside=1024, lp1=3073: ~150 MB per ring in batch
// Use 64 rings: ~640 MB for nside=256, ~9.6 GB for nside=1024
#define V2_RING_BATCH_SIZE 64

// ============================================================================
// Kernel: Extract Ylm for a batch of rings and apply N-S symmetry
// ============================================================================

/**
 * Build Ylm matrix for a batch of ring pairs (north + south)
 *
 * For each ring pair:
 *   - North ring at batch_ring_start + local_r
 *   - South ring at 4*nside - 2 - (batch_ring_start + local_r)
 *
 * Output: Ylm_batch[l, m, 2*batch_size] where:
 *   - First batch_size entries are north rings
 *   - Second batch_size entries are south rings (with (-1)^(l+m) applied)
 *
 * For equator ring, south contribution is zeroed.
 */
__global__ void build_ylm_batch_kernel(
    int l_max,
    int nside,
    int batch_ring_start,      // Starting ring index for this batch
    int actual_batch_size,     // Actual number of rings in this batch
    const real_t* ylm_batch_north,  // [lp1, lp1, actual_batch_size] Ylm for north batch
    real_t* Ylm_ns                  // [lp1, lp1, 2*actual_batch_size] output (north then south)
) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int local_r = blockIdx.z;

    if (l > l_max || m > l || local_r >= actual_batch_size) return;

    int lp1 = l_max + 1;
    int n_out_rings = 2 * actual_batch_size;

    // Source index in ylm_batch_north: [l, m, local_r]
    int src_idx = l * lp1 * actual_batch_size + m * actual_batch_size + local_r;
    real_t ylm_north_val = ylm_batch_north[src_idx];

    // North ring output: first half of output
    int dst_idx_north = l * lp1 * n_out_rings + m * n_out_rings + local_r;
    Ylm_ns[dst_idx_north] = ylm_north_val;

    // South ring output: second half of output
    int north_ring_idx = batch_ring_start + local_r;
    int dst_idx_south = l * lp1 * n_out_rings + m * n_out_rings + actual_batch_size + local_r;

    // Check if this is the equator ring
    if (north_ring_idx == 2 * nside - 1) {
        // Equator: zero out south contribution to prevent double-counting
        Ylm_ns[dst_idx_south] = 0.0;
    } else {
        // Apply (-1)^(l+m) symmetry
        int sign = ((l + m) % 2 == 0) ? 1 : -1;
        Ylm_ns[dst_idx_south] = sign * ylm_north_val;
    }
}

// ============================================================================
// Kernel: Extract Gm for a batch of ring pairs
// ============================================================================

/**
 * Extract Gm columns for a batch of ring pairs
 *
 * Input: Gm[n_maps, n_rings_total, lp1]
 * Output: Gm_batch[n_maps, 2*batch_size, lp1] (north then south rings)
 */
__global__ void extract_gm_batch_kernel(
    int n_maps,
    int n_rings_total,
    int l_max,
    int nside,
    int batch_ring_start,
    int actual_batch_size,
    const complex_t* Gm,           // [n_maps, n_rings_total, lp1]
    complex_t* Gm_batch            // [n_maps, 2*batch_size, lp1]
) {
    int t = blockIdx.x;
    int local_r = blockIdx.y;
    int m = blockIdx.z * blockDim.x + threadIdx.x;

    if (t >= n_maps || local_r >= actual_batch_size || m > l_max) return;

    int lp1 = l_max + 1;
    int n_batch_rings = 2 * actual_batch_size;

    int north_ring_idx = batch_ring_start + local_r;
    int south_ring_idx = 4 * nside - 2 - north_ring_idx;

    // Source indices in Gm
    int src_north = t * n_rings_total * lp1 + north_ring_idx * lp1 + m;
    int src_south = t * n_rings_total * lp1 + south_ring_idx * lp1 + m;

    // Destination indices in Gm_batch
    int dst_north = t * n_batch_rings * lp1 + local_r * lp1 + m;
    int dst_south = t * n_batch_rings * lp1 + (actual_batch_size + local_r) * lp1 + m;

    Gm_batch[dst_north] = Gm[src_north];
    Gm_batch[dst_south] = Gm[src_south];
}

// ============================================================================
// Accumulate batch contribution to alm using cuBLAS DGEMM
// ============================================================================

/**
 * Accumulate alm contributions from a batch of rings using DGEMM
 *
 * For each m value:
 *   alm[t, m:lmax+1, m] += Ylm[m:lmax+1, m, :] @ Gm[t, :, m]
 *
 * We process all m values together using batched approach:
 *   - Ylm_ns: [lp1, lp1, n_batch_rings] real
 *   - Gm_batch: [n_maps, n_batch_rings, lp1] complex
 *
 * For efficiency, we do a single large GEMM:
 *   Result[l, m] = sum_r Ylm[l, m, r] * Gm[r, m]
 *
 * This is equivalent to: alm += Ylm @ Gm (with appropriate reshaping)
 */
void accumulate_batch_gemm(
    cublasHandle_t handle,
    int l_max,
    int n_maps,
    int n_batch_rings,
    const real_t* Ylm_ns,          // [lp1, lp1, n_batch_rings]
    const complex_t* Gm_batch,     // [n_maps, n_batch_rings, lp1]
    complex_t* alm,                // [n_maps, lp1, lp1]
    real_t* work_real,             // [lp1 * lp1] workspace
    real_t* work_imag              // [lp1 * lp1] workspace
) {
    int lp1 = l_max + 1;
    double alpha = 1.0;
    double beta_acc = 1.0;   // Accumulate
    double beta_zero = 0.0;

    // For each map
    for (int t = 0; t < n_maps; t++) {
        // We need to compute: alm[l, m] += sum_r Ylm[l, m, r] * Gm[r, m]
        //
        // Reshape as matrix multiplication:
        // - Ylm viewed as [lp1*lp1, n_batch_rings] (flatten l,m dimensions)
        // - Gm viewed as [n_batch_rings, lp1]
        // - Result: [lp1*lp1, lp1] but we only need diagonal blocks
        //
        // Actually, for each m, we have a GEMV. Let's use that approach
        // but with cuBLAS streams for parallelism.
        //
        // Simpler approach: Loop over m, use GEMV
        // This is what we had before but we can remove per-m sync.

        const complex_t* Gm_t = Gm_batch + t * n_batch_rings * lp1;
        complex_t* alm_t = alm + t * lp1 * lp1;

        for (int m = 0; m <= l_max; m++) {
            int n_l = l_max - m + 1;

            // Ylm for this m: Ylm[m:lmax+1, m, :]
            // Layout: Ylm[l, m, r] at l * lp1 * n_batch_rings + m * n_batch_rings + r
            // For fixed m, varying l from m to lmax, this is:
            //   Ylm[m, m, :], Ylm[m+1, m, :], ..., Ylm[lmax, m, :]
            // Stride between consecutive l's is lp1 * n_batch_rings
            // Within each l, r varies from 0 to n_batch_rings-1 contiguously
            //
            // We need matrix A[l-m, r] = Ylm[l, m, r]
            // A is [n_l, n_batch_rings] in row-major
            // In column-major (cuBLAS): A[r, l-m] which is [n_batch_rings, n_l]
            //
            // GEMV: y = A^T x where A is [n_batch_rings, n_l], x is [n_batch_rings], y is [n_l]
            // Use CUBLAS_OP_T

            const real_t* Ylm_m = Ylm_ns + m * lp1 * n_batch_rings + m * n_batch_rings;
            // This points to Ylm[m, m, 0]. Stride to next l is lp1 * n_batch_rings.

            // Gm for this m: Gm[t, :, m]
            // Layout: Gm[t, r, m] at t * n_batch_rings * lp1 + r * lp1 + m
            // For fixed t and m, varying r: stride is lp1

            // We need to extract Gm[:, m] into contiguous arrays
            // Real parts
            cublasDcopy(handle, n_batch_rings,
                (const double*)Gm_t + m * 2,  // Start at Gm[0, m].x
                lp1 * 2,                       // Stride
                work_real, 1);

            // Imag parts
            cublasDcopy(handle, n_batch_rings,
                (const double*)Gm_t + m * 2 + 1,  // Start at Gm[0, m].y
                lp1 * 2,
                work_imag, 1);

            // GEMV for real part
            // A is [n_batch_rings rows, n_l cols] when viewed with lda = lp1 * n_batch_rings
            // But we have Ylm[l, m, r] with l varying faster...
            //
            // Let's think again:
            // Ylm_m points to Ylm[m, m, 0]
            // Ylm[l, m, r] = Ylm_m[(l-m) * lp1 * n_batch_rings + r]
            //
            // For GEMV y = A @ x:
            // y[l-m] = sum_r A[l-m, r] * x[r]
            //        = sum_r Ylm[l, m, r] * Gm[r, m]
            //
            // In cuBLAS column-major, A[i,j] is at A + i + j*lda
            // We want A[l-m, r] at position (l-m) + r * lda
            // With our layout: Ylm_m[(l-m) * lp1 * n_batch_rings + r]
            // This is (l-m) * stride_l + r where stride_l = lp1 * n_batch_rings
            //
            // For column-major with lda = stride_l: A[l-m, r] at (l-m) + r * lda
            // But we have it at (l-m) * stride_l + r
            // These don't match unless we transpose.
            //
            // Actually, we have A[i, j] = Ylm_m[i * stride + j]
            // This is row-major with row stride = lp1 * n_batch_rings
            // In column-major terms, this is A^T stored with lda = 1 (columns contiguous)
            //
            // For GEMV with row-major A[n_l, n_batch_rings]:
            // y = A @ x can be done as y = A^T^T @ x using CUBLAS_OP_T on the transpose
            //
            // Or: treat as column-major B[n_batch_rings, n_l] where B = A^T
            // Then y = B^T @ x = A @ x
            // B[r, l-m] = A[l-m, r] = Ylm_m[(l-m) * stride + r]
            // In col-major: B at position r + (l-m) * lda_B
            // Our storage: (l-m) * stride + r
            // For these to match: r + (l-m) * lda_B = (l-m) * stride + r
            // => lda_B = stride = lp1 * n_batch_rings
            //
            // So B is [n_batch_rings, n_l] with lda = lp1 * n_batch_rings
            // y = B^T @ x => use CUBLAS_OP_T, m=n_l, n=n_batch_rings, lda=lp1*n_batch_rings

            // Real part: alm.real += Ylm @ Gm.real
            cublasDgemv(handle, CUBLAS_OP_T,
                n_batch_rings, n_l,        // Dimensions of B (before transpose)
                &alpha,
                Ylm_m, lp1 * n_batch_rings,  // B and lda
                work_real, 1,              // x
                &beta_acc,
                (double*)(alm_t + m * lp1 + m), 2 * lp1);  // y with stride

            // Imag part: alm.imag += Ylm @ Gm.imag
            cublasDgemv(handle, CUBLAS_OP_T,
                n_batch_rings, n_l,
                &alpha,
                Ylm_m, lp1 * n_batch_rings,
                work_imag, 1,
                &beta_acc,
                (double*)(alm_t + m * lp1 + m) + 1, 2 * lp1);
        }
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
// Full map2alm_v2 Implementation (Batched)
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

    // 5. Allocate batched buffers
    int batch_size = V2_RING_BATCH_SIZE;
    int n_batches = CEILDIV(n_north_rings, batch_size);

    // Ylm for one batch of north rings: [lp1, lp1, batch_size]
    real_t* ylm_batch_north;
    CUDA_CHECK(cudaMalloc(&ylm_batch_north, (size_t)lp1 * lp1 * batch_size * sizeof(real_t)));

    // Ylm with N-S symmetry applied: [lp1, lp1, 2*batch_size]
    real_t* Ylm_ns;
    CUDA_CHECK(cudaMalloc(&Ylm_ns, (size_t)lp1 * lp1 * 2 * batch_size * sizeof(real_t)));

    // Gm for batch of ring pairs: [n_maps, 2*batch_size, lp1]
    complex_t* Gm_batch;
    CUDA_CHECK(cudaMalloc(&Gm_batch, (size_t)n_maps * 2 * batch_size * lp1 * sizeof(complex_t)));

    // Work buffers for GEMV
    real_t* work_real;
    real_t* work_imag;
    CUDA_CHECK(cudaMalloc(&work_real, 2 * batch_size * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&work_imag, 2 * batch_size * sizeof(real_t)));

    // 6. cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS initialization failed: %d\n", status);
        goto cleanup;
    }

    // 7. Process rings in batches
    for (int batch = 0; batch < n_batches; batch++) {
        int batch_start = batch * batch_size;
        int actual_size = min(batch_size, n_north_rings - batch_start);

        if (actual_size <= 0) break;

        // 7a. Compute Ylm for this batch of north rings
        compute_ylm_spin0(l_max, actual_size,
                          geom->log_beta + batch_start,
                          geom->beta_sign + batch_start,
                          ylm_batch_north);

        // 7b. Build Ylm with N-S symmetry
        dim3 block_ylm(16, 16, 1);
        dim3 grid_ylm(CEILDIV(lp1, 16), CEILDIV(lp1, 16), actual_size);
        build_ylm_batch_kernel<<<grid_ylm, block_ylm>>>(
            l_max, nside, batch_start, actual_size,
            ylm_batch_north, Ylm_ns
        );

        // 7c. Extract Gm for this batch of ring pairs
        dim3 block_gm(256);
        dim3 grid_gm(n_maps, actual_size, CEILDIV(lp1, 256));
        extract_gm_batch_kernel<<<grid_gm, block_gm>>>(
            n_maps, n_rings, l_max, nside,
            batch_start, actual_size,
            Gm, Gm_batch
        );

        // 7d. Synchronize before cuBLAS
        CUDA_CHECK(cudaDeviceSynchronize());

        // 7e. Accumulate to alm using cuBLAS
        accumulate_batch_gemm(handle, l_max, n_maps, 2 * actual_size,
                              Ylm_ns, Gm_batch, alm_out,
                              work_real, work_imag);
    }

    // 8. Synchronize after all batches
    CUDA_CHECK(cudaDeviceSynchronize());

    // 9. Apply pixel area normalization
    {
        real_t pix_area = 4.0 * PI / (12.0 * nside * nside);
        dim3 block_scale(16, 16);
        dim3 grid_scale(n_maps, CEILDIV(lp1, 16), CEILDIV(lp1, 16));
        scale_alm_kernel<<<grid_scale, block_scale>>>(n_maps, lp1, pix_area, alm_out);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cublasDestroy(handle);

cleanup:
    cudaFree(Gm);
    cudaFree(ylm_batch_north);
    cudaFree(Ylm_ns);
    cudaFree(Gm_batch);
    cudaFree(work_real);
    cudaFree(work_imag);
    free_ring_geometry(geom);
}

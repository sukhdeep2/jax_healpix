/**
 * FFT-based Gm Computation for map2alm
 *
 * Replaces naive DFT with cuFFT for O(n_pixels * log(n_pixels)) complexity.
 *
 * Mathematical basis:
 *   Gm[m] = sum_{j=0}^{N-1} map[j] * exp(-i * m * phi_j)
 *
 * where phi_j = phi_0 + 2*pi*j/N (uniform spacing within ring).
 *
 * This factors as:
 *   Gm[m] = exp(-i * m * phi_0) * sum_{j=0}^{N-1} map[j] * exp(-2*pi*i * m * j / N)
 *         = exp(-i * m * phi_0) * FFT(map)[m]
 */

#include "../include/fft_gm.cuh"
#include "../include/ylm_recurrence.cuh"
#include <stdio.h>

// ============================================================================
// FFT Plan Management
// ============================================================================

fft_plans_t* create_fft_plans(int nside, int l_max) {
    fft_plans_t* plans;
    CUDA_CHECK(cudaMallocManaged(&plans, sizeof(fft_plans_t)));

    plans->nside = nside;
    plans->l_max = l_max;

    // Equatorial rings: 2*nside + 1 rings, each with 4*nside pixels
    plans->n_equatorial_rings = 2 * nside + 1;
    plans->equatorial_fft_size = 4 * nside;

    // Create batched R2C plan for equatorial rings
    int n = plans->equatorial_fft_size;
    int batch = plans->n_equatorial_rings;

    cufftResult result = cufftPlanMany(&plans->equatorial_plan,
        1,          // 1D FFT
        &n,         // FFT size array
        NULL,       // inembed (NULL = default)
        1,          // istride
        n,          // idist (distance between batches in input)
        NULL,       // onembed
        1,          // ostride
        n / 2 + 1,  // odist (distance between batches in output)
        CUFFT_D2Z,  // Double precision Real-to-Complex
        batch       // Number of FFTs
    );

    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT plan creation failed for equatorial rings: %d\n", result);
        cudaFree(plans);
        return NULL;
    }

    // Polar rings: nside - 1 unique sizes
    plans->n_polar_sizes = nside - 1;
    CUDA_CHECK(cudaMalloc(&plans->polar_plans, plans->n_polar_sizes * sizeof(cufftHandle)));

    // Create plans for each polar ring size
    cufftHandle* h_polar_plans = new cufftHandle[plans->n_polar_sizes];

    for (int i = 0; i < plans->n_polar_sizes; i++) {
        int ring_size = 4 * (i + 1);  // Ring sizes: 4, 8, 12, ..., 4*(nside-1)
        int polar_batch = 2;  // North + South ring pair

        result = cufftPlanMany(&h_polar_plans[i],
            1,                   // 1D FFT
            &ring_size,          // FFT size
            NULL,                // inembed
            1,                   // istride
            ring_size,           // idist
            NULL,                // onembed
            1,                   // ostride
            ring_size / 2 + 1,   // odist
            CUFFT_D2Z,           // Double precision R2C
            polar_batch          // Batch of 2 (north + south)
        );

        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT plan creation failed for polar ring size %d: %d\n", ring_size, result);
            // Cleanup previously created plans
            for (int j = 0; j < i; j++) {
                cufftDestroy(h_polar_plans[j]);
            }
            cufftDestroy(plans->equatorial_plan);
            delete[] h_polar_plans;
            cudaFree(plans->polar_plans);
            cudaFree(plans);
            return NULL;
        }
    }

    // Copy plans to device
    CUDA_CHECK(cudaMemcpy(plans->polar_plans, h_polar_plans,
                          plans->n_polar_sizes * sizeof(cufftHandle),
                          cudaMemcpyHostToDevice));
    delete[] h_polar_plans;

    return plans;
}

void destroy_fft_plans(fft_plans_t* plans) {
    if (plans) {
        cufftDestroy(plans->equatorial_plan);

        // Destroy polar plans
        cufftHandle* h_polar_plans = new cufftHandle[plans->n_polar_sizes];
        CUDA_CHECK(cudaMemcpy(h_polar_plans, plans->polar_plans,
                              plans->n_polar_sizes * sizeof(cufftHandle),
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < plans->n_polar_sizes; i++) {
            cufftDestroy(h_polar_plans[i]);
        }

        delete[] h_polar_plans;
        cudaFree(plans->polar_plans);
        cudaFree(plans);
    }
}

// ============================================================================
// Phase Correction Kernel
// ============================================================================

__global__ void apply_phase_correction_kernel(
    int n_rings,
    int fft_input_size,  // Input size N (number of pixels)
    int fft_out_size,    // Output size from R2C FFT (N/2+1)
    int l_max,
    const cufftDoubleComplex* fft_out,  // [n_rings, fft_out_size]
    const real_t* phi_0,                 // [n_rings]
    complex_t* Gm_out                    // [n_rings, l_max+1]
) {
    int r = blockIdx.x;
    int m = blockIdx.y * blockDim.x + threadIdx.x;

    if (r >= n_rings || m > l_max) return;

    // Get FFT output with periodicity and conjugate symmetry handling
    // DFT is periodic: FFT[m] = FFT[m % N]
    // For R2C FFT: output[k] = DFT[k] for k=0..N/2
    // For k > N/2: DFT[k] = conj(DFT[N-k]) (conjugate symmetry for real input)
    int m_mod = m % fft_input_size;  // Handle periodicity
    cufftDoubleComplex fft_val;
    if (m_mod < fft_out_size) {
        fft_val = fft_out[r * fft_out_size + m_mod];
    } else {
        // Use conjugate symmetry: FFT[m_mod] = conj(FFT[N-m_mod])
        int mirror_m = fft_input_size - m_mod;
        fft_val = fft_out[r * fft_out_size + mirror_m];
        fft_val.y = -fft_val.y;  // Conjugate
    }

    // Phase correction: multiply by exp(-i * m * phi_0[r])
    real_t angle = -m * phi_0[r];
    real_t cos_a = cos(angle);
    real_t sin_a = sin(angle);

    // Complex multiply: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    int out_idx = r * (l_max + 1) + m;
    Gm_out[out_idx].x = fft_val.x * cos_a - fft_val.y * sin_a;
    Gm_out[out_idx].y = fft_val.x * sin_a + fft_val.y * cos_a;
}

// ============================================================================
// Phase Correction Kernel with Map Index Support
// ============================================================================

__global__ void apply_phase_correction_kernel_batched(
    int n_maps,
    int n_rings,
    int fft_input_size,  // Input size N (number of pixels)
    int fft_out_size,    // Output size from R2C FFT (N/2+1)
    int l_max,
    const cufftDoubleComplex* fft_out,  // [n_maps, n_rings, fft_out_size]
    const real_t* phi_0,                 // [n_rings]
    complex_t* Gm_out                    // [n_maps, n_rings, l_max+1]
) {
    int t = blockIdx.z;  // Map index
    int r = blockIdx.x;  // Ring index
    int m = blockIdx.y * blockDim.x + threadIdx.x;  // m value

    if (t >= n_maps || r >= n_rings || m > l_max) return;

    int lp1 = l_max + 1;

    // Get FFT output with periodicity and conjugate symmetry handling
    // DFT is periodic: FFT[m] = FFT[m % N]
    int m_mod = m % fft_input_size;  // Handle periodicity
    cufftDoubleComplex fft_val;
    if (m_mod < fft_out_size) {
        fft_val = fft_out[t * n_rings * fft_out_size + r * fft_out_size + m_mod];
    } else {
        // Use conjugate symmetry: FFT[m_mod] = conj(FFT[N-m_mod])
        int mirror_m = fft_input_size - m_mod;
        fft_val = fft_out[t * n_rings * fft_out_size + r * fft_out_size + mirror_m];
        fft_val.y = -fft_val.y;  // Conjugate
    }

    // Phase correction: multiply by exp(-i * m * phi_0[r])
    real_t angle = -m * phi_0[r];
    real_t cos_a = cos(angle);
    real_t sin_a = sin(angle);

    // Complex multiply: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    int out_idx = t * n_rings * lp1 + r * lp1 + m;
    Gm_out[out_idx].x = fft_val.x * cos_a - fft_val.y * sin_a;
    Gm_out[out_idx].y = fft_val.x * sin_a + fft_val.y * cos_a;
}

// ============================================================================
// Equatorial FFT
// ============================================================================

void compute_gm_equatorial_fft(
    const fft_plans_t* plans,
    int n_maps,
    const real_t* map_equatorial,  // [n_maps, n_equatorial, 4*nside]
    const real_t* phi_0,           // [n_equatorial]
    complex_t* Gm_out              // [n_maps, n_equatorial, l_max+1]
) {
    int n_equatorial = plans->n_equatorial_rings;
    int fft_size = plans->equatorial_fft_size;
    int fft_out_size = fft_size / 2 + 1;
    int l_max = plans->l_max;
    int lp1 = l_max + 1;

    // Allocate FFT output buffer
    cufftDoubleComplex* fft_out;
    CUDA_CHECK(cudaMalloc(&fft_out, (size_t)n_maps * n_equatorial * fft_out_size * sizeof(cufftDoubleComplex)));

    // Process each map
    for (int t = 0; t < n_maps; t++) {
        const real_t* map_ptr = map_equatorial + t * n_equatorial * fft_size;
        cufftDoubleComplex* fft_ptr = fft_out + t * n_equatorial * fft_out_size;

        // Execute batched FFT
        cufftResult result = cufftExecD2Z(plans->equatorial_plan, (cufftDoubleReal*)map_ptr, fft_ptr);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT execution failed: %d\n", result);
            cudaFree(fft_out);
            return;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply phase correction
    dim3 block(256);
    dim3 grid(n_equatorial, CEILDIV(lp1, 256), n_maps);

    apply_phase_correction_kernel_batched<<<grid, block>>>(
        n_maps, n_equatorial, fft_size, fft_out_size, l_max,
        fft_out, phi_0, Gm_out
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(fft_out);
}

// ============================================================================
// Polar Ring FFT (Individual or Paired)
// ============================================================================

/**
 * Kernel to copy ring data with proper padding for polar rings
 */
__global__ void copy_polar_ring_data_kernel(
    int n_maps,
    int n_rings_total,
    int ring_size,
    int fft_pitch,     // 4*nside (padded size)
    int north_ring_idx,
    int south_ring_idx,
    const real_t* map_in,       // [n_maps, n_rings_total, 4*nside]
    real_t* ring_buffer         // [n_maps, 2, ring_size]
) {
    int t = blockIdx.x;
    int j = threadIdx.x;

    if (t >= n_maps || j >= ring_size) return;

    int map_pitch = n_rings_total * fft_pitch;

    // Copy north ring
    ring_buffer[t * 2 * ring_size + j] = map_in[t * map_pitch + north_ring_idx * fft_pitch + j];

    // Copy south ring
    ring_buffer[t * 2 * ring_size + ring_size + j] = map_in[t * map_pitch + south_ring_idx * fft_pitch + j];
}

/**
 * Kernel to apply phase correction for polar rings
 */
__global__ void apply_phase_polar_kernel(
    int n_maps,
    int ring_size,       // Input FFT size N (number of pixels)
    int fft_out_size,    // Output size from R2C FFT (N/2+1)
    int l_max,
    int north_ring_idx,
    int south_ring_idx,
    int n_rings_total,
    real_t phi_0_north,
    real_t phi_0_south,
    const cufftDoubleComplex* fft_out,  // [n_maps, 2, fft_out_size]
    complex_t* Gm_out                    // [n_maps, n_rings_total, l_max+1]
) {
    int t = blockIdx.x;
    int m = blockIdx.y * blockDim.x + threadIdx.x;

    if (t >= n_maps || m > l_max) return;

    int lp1 = l_max + 1;

    // DFT is periodic: FFT[m] = FFT[m % N]
    // For R2C FFT with size N, we store bins 0 to N/2, and use conjugate symmetry for the rest
    // FFT[m] = FFT[m % N], and for m_mod in (N/2, N): FFT[m_mod] = conj(FFT[N - m_mod])

    // North ring phase correction with aliasing and conjugate symmetry handling
    int m_mod_n = m % ring_size;  // Handle periodicity
    cufftDoubleComplex fft_val_n;
    if (m_mod_n < fft_out_size) {
        fft_val_n = fft_out[t * 2 * fft_out_size + m_mod_n];
    } else {
        // Use conjugate symmetry: FFT[m_mod] = conj(FFT[N - m_mod])
        int mirror_m = ring_size - m_mod_n;
        fft_val_n = fft_out[t * 2 * fft_out_size + mirror_m];
        fft_val_n.y = -fft_val_n.y;  // Conjugate
    }

    real_t angle_n = -m * phi_0_north;
    real_t cos_n = cos(angle_n);
    real_t sin_n = sin(angle_n);

    int out_idx_n = t * n_rings_total * lp1 + north_ring_idx * lp1 + m;
    Gm_out[out_idx_n].x = fft_val_n.x * cos_n - fft_val_n.y * sin_n;
    Gm_out[out_idx_n].y = fft_val_n.x * sin_n + fft_val_n.y * cos_n;

    // South ring phase correction with aliasing and conjugate symmetry handling
    int m_mod_s = m % ring_size;  // Handle periodicity (same ring_size as north)
    cufftDoubleComplex fft_val_s;
    if (m_mod_s < fft_out_size) {
        fft_val_s = fft_out[t * 2 * fft_out_size + fft_out_size + m_mod_s];
    } else {
        // Use conjugate symmetry: FFT[m_mod] = conj(FFT[N - m_mod])
        int mirror_m = ring_size - m_mod_s;
        fft_val_s = fft_out[t * 2 * fft_out_size + fft_out_size + mirror_m];
        fft_val_s.y = -fft_val_s.y;  // Conjugate
    }

    real_t angle_s = -m * phi_0_south;
    real_t cos_s = cos(angle_s);
    real_t sin_s = sin(angle_s);

    int out_idx_s = t * n_rings_total * lp1 + south_ring_idx * lp1 + m;
    Gm_out[out_idx_s].x = fft_val_s.x * cos_s - fft_val_s.y * sin_s;
    Gm_out[out_idx_s].y = fft_val_s.x * sin_s + fft_val_s.y * cos_s;
}

void compute_gm_polar_pairs_fft(
    int nside,
    int l_max,
    int n_maps,
    const real_t* map_in,          // [n_maps, n_rings, 4*nside]
    const ring_geometry_t* geom,
    complex_t* Gm_out              // [n_maps, n_rings, l_max+1]
) {
    int n_rings_total = 4 * nside - 1;
    int lp1 = l_max + 1;

    // Copy phi_0 and n_pixels to host for indexing
    real_t* h_phi_0 = new real_t[n_rings_total];
    int* h_n_pixels = new int[n_rings_total];
    CUDA_CHECK(cudaMemcpy(h_phi_0, geom->phi_0, n_rings_total * sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_n_pixels, geom->n_pixels, n_rings_total * sizeof(int), cudaMemcpyDeviceToHost));

    // Process polar ring pairs
    // North polar: rings 0 to nside-2 (indices 0 to nside-2)
    // South polar: rings 3*nside to 4*nside-2 (indices 3*nside-1 to 4*nside-3)
    for (int i = 0; i < nside - 1; i++) {
        int north_ring_idx = i;
        int south_ring_idx = 4 * nside - 2 - i;
        int ring_size = h_n_pixels[north_ring_idx];
        int fft_out_size = ring_size / 2 + 1;

        // Create FFT plan for this ring size
        cufftHandle plan;
        int batch = 2 * n_maps;  // North + South for all maps

        cufftResult result = cufftPlanMany(&plan,
            1,                  // 1D FFT
            &ring_size,         // FFT size
            NULL,               // inembed
            1,                  // istride
            ring_size,          // idist
            NULL,               // onembed
            1,                  // ostride
            fft_out_size,       // odist
            CUFFT_D2Z,          // Double precision R2C
            batch               // Batch size
        );

        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT plan creation failed for polar ring size %d: %d\n", ring_size, result);
            delete[] h_phi_0;
            delete[] h_n_pixels;
            return;
        }

        // Allocate ring buffer and FFT output
        real_t* ring_buffer;
        cufftDoubleComplex* fft_out;
        CUDA_CHECK(cudaMalloc(&ring_buffer, (size_t)n_maps * 2 * ring_size * sizeof(real_t)));
        CUDA_CHECK(cudaMalloc(&fft_out, (size_t)n_maps * 2 * fft_out_size * sizeof(cufftDoubleComplex)));

        // Copy ring data
        dim3 block_copy(ring_size);
        dim3 grid_copy(n_maps);
        copy_polar_ring_data_kernel<<<grid_copy, block_copy>>>(
            n_maps, n_rings_total, ring_size, 4 * nside,
            north_ring_idx, south_ring_idx,
            map_in, ring_buffer
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Execute FFT
        result = cufftExecD2Z(plan, (cufftDoubleReal*)ring_buffer, fft_out);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT execution failed for polar ring: %d\n", result);
            cufftDestroy(plan);
            cudaFree(ring_buffer);
            cudaFree(fft_out);
            delete[] h_phi_0;
            delete[] h_n_pixels;
            return;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Apply phase correction
        dim3 block_phase(256);
        dim3 grid_phase(n_maps, CEILDIV(lp1, 256));
        apply_phase_polar_kernel<<<grid_phase, block_phase>>>(
            n_maps, ring_size, fft_out_size, l_max,
            north_ring_idx, south_ring_idx, n_rings_total,
            h_phi_0[north_ring_idx], h_phi_0[south_ring_idx],
            fft_out, Gm_out
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Cleanup
        cufftDestroy(plan);
        cudaFree(ring_buffer);
        cudaFree(fft_out);
    }

    delete[] h_phi_0;
    delete[] h_n_pixels;
}

// ============================================================================
// Main Entry Point: Compute Gm for All Rings
// ============================================================================

void compute_gm_all_rings(
    int nside,
    int l_max,
    int n_maps,
    const real_t* map_in,          // [n_maps, n_rings, 4*nside]
    const ring_geometry_t* geom,
    complex_t* Gm_out              // [n_maps, n_rings, l_max+1]
) {
    int n_rings = 4 * nside - 1;
    int lp1 = l_max + 1;
    int fft_size = 4 * nside;
    int fft_out_size = fft_size / 2 + 1;

    // Initialize Gm to zero (important for m >= npix case)
    CUDA_CHECK(cudaMemset(Gm_out, 0, (size_t)n_maps * n_rings * lp1 * sizeof(complex_t)));

    // Equatorial rings: indices nside-1 to 3*nside-1 (inclusive)
    int eq_start = nside - 1;
    int eq_end = 3 * nside - 1;
    int n_equatorial = eq_end - eq_start + 1;  // 2*nside + 1

    // Create equatorial FFT plan
    cufftHandle eq_plan;
    int eq_batch = n_equatorial;

    cufftResult result = cufftPlanMany(&eq_plan,
        1,              // 1D FFT
        &fft_size,      // FFT size
        NULL,           // inembed
        1,              // istride
        fft_size,       // idist
        NULL,           // onembed
        1,              // ostride
        fft_out_size,   // odist
        CUFFT_D2Z,      // Double precision R2C
        eq_batch        // Batch size
    );

    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT plan creation failed for equatorial rings: %d\n", result);
        return;
    }

    // Allocate FFT output buffer for equatorial rings
    cufftDoubleComplex* eq_fft_out;
    CUDA_CHECK(cudaMalloc(&eq_fft_out, (size_t)n_maps * n_equatorial * fft_out_size * sizeof(cufftDoubleComplex)));

    // Process equatorial rings for each map
    for (int t = 0; t < n_maps; t++) {
        const real_t* map_eq_ptr = map_in + t * n_rings * fft_size + eq_start * fft_size;
        cufftDoubleComplex* fft_ptr = eq_fft_out + t * n_equatorial * fft_out_size;

        result = cufftExecD2Z(eq_plan, (cufftDoubleReal*)map_eq_ptr, fft_ptr);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT execution failed for equatorial rings: %d\n", result);
            cufftDestroy(eq_plan);
            cudaFree(eq_fft_out);
            return;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply phase correction to equatorial rings
    dim3 block_eq(256);
    dim3 grid_eq(n_equatorial, CEILDIV(lp1, 256), n_maps);

    // Copy phi_0 for equatorial rings
    real_t* eq_phi_0;
    CUDA_CHECK(cudaMalloc(&eq_phi_0, n_equatorial * sizeof(real_t)));
    CUDA_CHECK(cudaMemcpy(eq_phi_0, geom->phi_0 + eq_start, n_equatorial * sizeof(real_t), cudaMemcpyDeviceToDevice));

    // Need to write equatorial Gm to correct locations
    // Create a kernel that writes to the correct offset
    for (int t = 0; t < n_maps; t++) {
        apply_phase_correction_kernel<<<dim3(n_equatorial, CEILDIV(lp1, 256)), 256>>>(
            n_equatorial, fft_size, fft_out_size, l_max,
            eq_fft_out + t * n_equatorial * fft_out_size,
            eq_phi_0,
            Gm_out + t * n_rings * lp1 + eq_start * lp1
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup equatorial
    cufftDestroy(eq_plan);
    cudaFree(eq_fft_out);
    cudaFree(eq_phi_0);

    // Process polar ring pairs
    compute_gm_polar_pairs_fft(nside, l_max, n_maps, map_in, geom, Gm_out);
}

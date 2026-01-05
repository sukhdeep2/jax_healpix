/**
 * YLM Recurrence Kernels for SPHT
 *
 * Computes spin-weighted spherical harmonics using recurrence relations
 * from:
 *   - arXiv:1010.2084 (spin-0)
 *   - arXiv:astro-ph/0502469 (spin-2)
 */

#include "../include/ylm_recurrence.cuh"
#include "../include/log_arithmetic.cuh"
#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ============================================================================
// Ring Geometry Precomputation
// ============================================================================

/**
 * Kernel to precompute ring geometry
 * Reference: jax_healpix/SPHT_jax.py ring_log_beta()
 */
__global__ void precompute_ring_geometry_kernel(int nside, ring_geometry_t* geom) {
    int ring_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_rings = 4 * nside - 1;

    if (ring_idx >= n_rings) return;

    int ring_i = ring_idx + 1;  // 1-indexed ring number

    // Polar caps: rings 1 to nside-1 (north) and 3*nside+1 to 4*nside-1 (south)
    if (ring_i < nside) {
        // North polar cap: beta = 1 - i^2 / (3 * nside^2)
        // In log space: log(1 - i^2/(3*nside^2))
        real_t log_i2_over_3n2 = 2.0 * log((real_t)ring_i) - 2.0 * log((real_t)nside) - log(3.0);

        // logsumexp(0, log_i2_over_3n2, +1, -1) computes log(1 - i^2/(3*nside^2))
        real_t log_result;
        int8_t sign_result;
        logsumexp(0.0, log_i2_over_3n2, 1, -1, &log_result, &sign_result);

        geom->log_beta[ring_idx] = log_result;
        geom->beta_sign[ring_idx] = 1;  // North cap has positive cos(theta)
        geom->phi_0[ring_idx] = PI / (2.0 * ring_i) * 0.5;
        geom->n_pixels[ring_idx] = 4 * ring_i;

    } else if (ring_i > 3 * nside) {
        // South polar cap: mirror of north
        int mirror_i = 4 * nside - ring_i;
        real_t log_i2_over_3n2 = 2.0 * log((real_t)mirror_i) - 2.0 * log((real_t)nside) - log(3.0);

        real_t log_result;
        int8_t sign_result;
        logsumexp(0.0, log_i2_over_3n2, 1, -1, &log_result, &sign_result);

        geom->log_beta[ring_idx] = log_result;
        geom->beta_sign[ring_idx] = -1;  // South cap has negative cos(theta)
        geom->phi_0[ring_idx] = PI / (2.0 * mirror_i) * 0.5;
        geom->n_pixels[ring_idx] = 4 * mirror_i;

    } else {
        // Equatorial belt: beta = 4/3 - 2i/(3*nside)
        real_t beta_value = 4.0 / 3.0 - 2.0 * ring_i / (3.0 * nside);
        // Clamp to avoid -inf which causes NaN in recurrence
        // At equator cos(theta)=0, use very large negative log instead of -inf
        real_t abs_beta = fabs(beta_value);
        geom->log_beta[ring_idx] = (abs_beta < 1e-300) ? -700.0 : log(abs_beta);
        geom->beta_sign[ring_idx] = (beta_value >= 0) ? 1 : -1;

        int s = (ring_i % 2 == 0) ? 1 : 2;
        geom->phi_0[ring_idx] = PI / (2.0 * nside) * (1.0 - s / 2.0);
        geom->n_pixels[ring_idx] = 4 * nside;
    }

    // Compute log(sin(theta)) = 0.5 * log(1 - cos^2(theta))
    // sin^2 = 1 - beta^2, so log(sin) = 0.5 * log(1 - exp(2*log_beta))
    real_t log_beta_sq = 2.0 * geom->log_beta[ring_idx];
    real_t log_sin_sq;
    int8_t sign_sin_sq;
    logsumexp(0.0, log_beta_sq, 1, -1, &log_sin_sq, &sign_sin_sq);
    geom->log_sin_theta[ring_idx] = 0.5 * log_sin_sq;
}

ring_geometry_t* allocate_ring_geometry(int nside) {
    int n_rings = 4 * nside - 1;

    ring_geometry_t* geom;
    CUDA_CHECK(cudaMallocManaged(&geom, sizeof(ring_geometry_t)));

    CUDA_CHECK(cudaMalloc(&geom->log_beta, n_rings * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&geom->beta_sign, n_rings * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&geom->log_sin_theta, n_rings * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&geom->phi_0, n_rings * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&geom->n_pixels, n_rings * sizeof(int)));

    geom->n_rings = n_rings;
    geom->nside = nside;

    return geom;
}

void free_ring_geometry(ring_geometry_t* geom) {
    if (geom) {
        cudaFree(geom->log_beta);
        cudaFree(geom->beta_sign);
        cudaFree(geom->log_sin_theta);
        cudaFree(geom->phi_0);
        cudaFree(geom->n_pixels);
        cudaFree(geom);
    }
}

void precompute_ring_geometry(int nside, ring_geometry_t* geom) {
    int n_rings = 4 * nside - 1;
    int block_size = 256;
    int grid_size = CEILDIV(n_rings, block_size);

    precompute_ring_geometry_kernel<<<grid_size, block_size>>>(nside, geom);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// YLM Allocation
// ============================================================================

ylm_log_t* allocate_ylm_log(int l_max, int n_rings) {
    ylm_log_t* ylm;
    CUDA_CHECK(cudaMallocManaged(&ylm, sizeof(ylm_log_t)));

    size_t size = (size_t)(l_max + 1) * (l_max + 1) * n_rings;
    CUDA_CHECK(cudaMalloc(&ylm->log_ylm, size * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&ylm->sign_ylm, size * sizeof(int8_t)));

    ylm->l_max = l_max;
    ylm->n_rings = n_rings;

    // Initialize to -inf (log of 0) and sign=1
    CUDA_CHECK(cudaMemset(ylm->sign_ylm, 1, size * sizeof(int8_t)));

    return ylm;
}

void free_ylm_log(ylm_log_t* ylm) {
    if (ylm) {
        cudaFree(ylm->log_ylm);
        cudaFree(ylm->sign_ylm);
        cudaFree(ylm);
    }
}

// ============================================================================
// Diagonal Elements: sYLM_ll0
// ============================================================================

/**
 * Device function to compute log(A_lm)
 * Eq. 14 of ref 1: A_{l,m} = sqrt((4l^2 - 1) / (l^2 - m^2))
 */
__device__ __forceinline__
real_t log_A_lm(int l, int m) {
    // log(A_{l,m}) = 0.5 * [log(4l^2 - 1) - log(l^2 - m^2)]
    real_t log_num = logdiffexp(log(4.0) + 2.0 * log((real_t)l), 0.0);  // log(4l^2 - 1)
    real_t log_denom = logdiffexp(2.0 * log((real_t)l), 2.0 * log((real_t)m));  // log(l^2 - m^2)
    return 0.5 * (log_num - log_denom);
}

/**
 * Kernel for computing diagonal elements Y_{l,l}
 * Reference: jax_healpix/YLM_jax_log.py sYLM_ll0_log()
 *
 * SPLIT into two kernels to avoid race condition:
 * - This kernel computes Y_{l,l} for all l
 * - sYLM_ll_minus1_kernel computes Y_{l,l-1} AFTER this kernel completes
 */
__global__ void sYLM_ll_diagonal_kernel(int l_max, int n_rings,
                                         const real_t* log_beta_s,
                                         real_t* log_ylm,
                                         int8_t* sign_ylm) {
    int l = blockIdx.x * blockDim.x + threadIdx.x + 1;  // l from 1 to l_max
    int ring = blockIdx.y * blockDim.y + threadIdx.y;

    if (l > l_max || ring >= n_rings) return;

    int lp1 = l_max + 1;  // Dimension size

    // Compute cumulative prefactor: sum_{k=1}^{l} log((2k+1)/(2k))
    // This is 0.5 * sum_{k=1}^{l} [log(2k+1) - log(2k)]
    real_t log_prefact = 0.0;
    for (int k = 1; k <= l; k++) {
        log_prefact += log((real_t)(2 * k + 1)) - log((real_t)(2 * k));
    }
    log_prefact *= 0.5;

    // Y_{l,l} = (-1)^l * sin(theta)^l * prefactor
    // In log space: log|Y_{l,l}| = l * log(sin_theta) + log(prefact)
    int idx = INDEX_3D(l, l, ring, lp1, n_rings);
    log_ylm[idx] = l * log_beta_s[ring] + log_prefact;
    sign_ylm[idx] = (l % 2 == 0) ? 1 : -1;  // (-1)^l
}

/**
 * Kernel for computing sub-diagonal elements Y_{l,l-1}
 * MUST be called AFTER sYLM_ll_diagonal_kernel completes
 *
 * Y_{l,l-1} = Y_{l-1,l-1} * cos(theta) * sqrt(2l+1)
 */
__global__ void sYLM_ll_minus1_kernel(int l_max, int n_rings,
                                       const real_t* log_beta,
                                       const int8_t* beta_sign,
                                       real_t* log_ylm,
                                       int8_t* sign_ylm) {
    int l = blockIdx.x * blockDim.x + threadIdx.x + 1;  // l from 1 to l_max
    int ring = blockIdx.y * blockDim.y + threadIdx.y;

    if (l > l_max || ring >= n_rings) return;

    int lp1 = l_max + 1;

    // Y_{l,l-1} = Y_{l-1,l-1} * cos(theta) * sqrt(2(l-1)+3) = Y_{l-1,l-1} * cos(theta) * sqrt(2l+1)
    // For l=1: Y_{1,0} = Y_{0,0} * cos(theta) * sqrt(3)
    int idx_prev = INDEX_3D(l - 1, l - 1, ring, lp1, n_rings);
    int idx_lm1 = INDEX_3D(l, l - 1, ring, lp1, n_rings);

    log_ylm[idx_lm1] = log_ylm[idx_prev] + log_beta[ring] + 0.5 * log((real_t)(2 * (l - 1) + 3));
    sign_ylm[idx_lm1] = sign_ylm[idx_prev] * beta_sign[ring];
}

/**
 * Initialize Y_{0,0} = 1 (in log space: log(1) = 0, sign = +1)
 */
__global__ void init_Y00_kernel(int l_max, int n_rings,
                                 real_t* log_ylm,
                                 int8_t* sign_ylm) {
    int ring = blockIdx.x * blockDim.x + threadIdx.x;
    if (ring >= n_rings) return;

    int lp1 = l_max + 1;

    // Y_{0,0} = 1 -> log = 0, sign = +1
    int idx00 = INDEX_3D(0, 0, ring, lp1, n_rings);
    log_ylm[idx00] = 0.0;
    sign_ylm[idx00] = 1;

    // Set invalid entries to -inf
    for (int m = 1; m <= l_max; m++) {
        int idx0m = INDEX_3D(0, m, ring, lp1, n_rings);
        log_ylm[idx0m] = -INFINITY;
        sign_ylm[idx0m] = 1;
    }

    // Y_{1,m} for m > 1 is invalid
    for (int m = 2; m <= l_max; m++) {
        int idx1m = INDEX_3D(1, m, ring, lp1, n_rings);
        log_ylm[idx1m] = -INFINITY;
        sign_ylm[idx1m] = 1;
    }
}

void sYLM_ll0_log(int l_max, int n_rings,
                  const real_t* log_beta_s,
                  const real_t* log_beta,
                  const int8_t* beta_sign,
                  ylm_log_t* ylm) {
    // Initialize Y_{0,0}
    int block_size = 256;
    int grid_size = CEILDIV(n_rings, block_size);
    init_Y00_kernel<<<grid_size, block_size>>>(l_max, n_rings, ylm->log_ylm, ylm->sign_ylm);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute diagonal elements Y_{l,l}
    dim3 block(32, 8);  // 256 threads
    dim3 grid(CEILDIV(l_max, 32), CEILDIV(n_rings, 8));
    sYLM_ll_diagonal_kernel<<<grid, block>>>(l_max, n_rings, log_beta_s,
                                              ylm->log_ylm, ylm->sign_ylm);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());  // CRITICAL: Y_{l,l} must be done before Y_{l,l-1}

    // Compute sub-diagonal elements Y_{l,l-1}
    sYLM_ll_minus1_kernel<<<grid, block>>>(l_max, n_rings, log_beta, beta_sign,
                                            ylm->log_ylm, ylm->sign_ylm);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Off-Diagonal Recurrence: sYLM_l0
// ============================================================================

/**
 * Kernel for a single l value in the recurrence
 * Y_{l,m} = A_{l,m} * cos(theta) * Y_{l-1,m} - B_{l,m} * Y_{l-2,m}
 */
__global__ void sYLM_l0_single_l_kernel(int l, int l_max, int n_rings,
                                         const real_t* log_beta,
                                         const int8_t* beta_sign,
                                         real_t* log_ylm,
                                         int8_t* sign_ylm) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int ring = blockIdx.y * blockDim.y + threadIdx.y;

    if (ring >= n_rings) return;
    if (m >= l || m > l - 2) return;  // Only valid for m < l-1 (diagonal and sub-diagonal already computed)

    int lp1 = l_max + 1;

    // Compute recurrence coefficients in log space
    real_t log_Alm = log_A_lm(l, m);
    real_t log_Alm_prev = log_A_lm(l - 1, m);
    real_t log_Blm = log_Alm - log_Alm_prev;

    // Get indices
    int idx_lm1 = INDEX_3D(l - 1, m, ring, lp1, n_rings);
    int idx_lm2 = INDEX_3D(l - 2, m, ring, lp1, n_rings);
    int idx = INDEX_3D(l, m, ring, lp1, n_rings);

    // R1 = A_{l,m} * cos(theta) * Y_{l-1,m}
    real_t log_R1 = log_Alm + log_beta[ring] + log_ylm[idx_lm1];
    int8_t sign_R1 = beta_sign[ring] * sign_ylm[idx_lm1];

    // R2 = -B_{l,m} * Y_{l-2,m}  (note the minus sign)
    real_t log_R2 = log_Blm + log_ylm[idx_lm2];
    int8_t sign_R2 = -1 * sign_ylm[idx_lm2];

    // Y_{l,m} = R1 + R2 using logsumexp
    real_t log_result;
    int8_t sign_result;
    logsumexp(log_R1, log_R2, sign_R1, sign_R2, &log_result, &sign_result);

    // Store result
    log_ylm[idx] = log_result;
    sign_ylm[idx] = sign_result;
}

void sYLM_l0_all(int l_max, int n_rings,
                 const real_t* log_beta,
                 const int8_t* beta_sign,
                 ylm_log_t* ylm) {
    dim3 block(32, 8);

    // Sequential launch over l values due to dependency
    // CRITICAL: Must synchronize after each l because Y_{l,m} depends on Y_{l-1,m} and Y_{l-2,m}
    for (int l = 2; l <= l_max; l++) {
        dim3 grid(CEILDIV(l_max + 1, 32), CEILDIV(n_rings, 8));
        sYLM_l0_single_l_kernel<<<grid, block>>>(l, l_max, n_rings,
                                                   log_beta, beta_sign,
                                                   ylm->log_ylm, ylm->sign_ylm);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());  // Wait for this l before computing l+1
    }
}

// ============================================================================
// Spin-2 Harmonics: sYLM_l2
// ============================================================================

/**
 * Device function to compute log(alpha_lm)
 * Eq. A8 in ref 2: alpha_{l,m} = sqrt((2l+1)(l^2-m^2)/(2l-1))
 */
__device__ __forceinline__
real_t log_alpha_lm(int l, int m) {
    return 0.5 * (log((real_t)(2 * l + 1)) + log((real_t)(l * l - m * m)) - log((real_t)(2 * l - 1)));
}

/**
 * Kernel for computing spin-2 harmonics
 * Reference: jax_healpix/YLM_jax_log.py sYLM_l2()
 */
__global__ void sYLM_l2_kernel(int l_max, int n_rings,
                                const real_t* log_beta,
                                const real_t* log_beta_s2,
                                const int8_t* beta_sign,
                                const real_t* log_ylm_s0,
                                const int8_t* sign_ylm_s0,
                                real_t* log_ylm_p2,
                                int8_t* sign_ylm_p2,
                                real_t* log_ylm_m2,
                                int8_t* sign_ylm_m2) {
    int l = blockIdx.x * blockDim.x + threadIdx.x + 2;  // l from 2 to l_max
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int ring = blockIdx.z;

    if (l > l_max || m > l || ring >= n_rings) return;

    int lp1 = l_max + 1;

    // Normalization factor: sqrt((l-2)!/(l+2)!) = exp(0.5*(lgamma(l-1) - lgamma(l+3)))
    real_t log_factorial_norm = 0.5 * (lgamma((real_t)(l - 1)) - lgamma((real_t)(l + 3)));

    real_t log_alm = log_alpha_lm(l, m);
    real_t log_m = log((real_t)m + 1e-300);  // Avoid log(0)
    real_t log_l = log((real_t)l);

    // Get spin-0 values
    int idx_l = INDEX_3D(l, m, ring, lp1, n_rings);
    int idx_lm1 = INDEX_3D(l - 1, m, ring, lp1, n_rings);

    real_t log_Ylm = log_ylm_s0[idx_l];
    int8_t sign_Ylm = sign_ylm_s0[idx_l];
    real_t log_Ylm1 = log_ylm_s0[idx_lm1];
    int8_t sign_Ylm1 = sign_ylm_s0[idx_lm1];

    // ============ SPIN +2 ============
    // m2l = m^2 - l
    real_t log_m2l;
    int8_t sign_m2l;
    logsumexp(2.0 * log_m, log_l, 1, -1, &log_m2l, &sign_m2l);

    // First part of term 1: 2/sin^2 * m2l
    real_t log_term1 = log(2.0) - log_beta_s2[ring] + log_m2l;
    int8_t sign_term1 = sign_m2l;

    // Subtract l(l-1)
    real_t log_ll1 = log_l + log((real_t)(l - 1));
    logsumexp(log_term1, log_ll1, sign_term1, -1, &log_term1, &sign_term1);

    // Multiply by Y_{l,m}
    log_term1 += log_Ylm;
    sign_term1 *= sign_Ylm;

    // Term 2: 2 * alpha * cos/sin^2 * Y_{l-1,m}
    real_t log_term2 = log(2.0) + log_beta[ring] - log_beta_s2[ring] + log_alm + log_Ylm1;
    int8_t sign_term2 = beta_sign[ring] * sign_Ylm1;

    // Sum terms
    real_t log_result_p2;
    int8_t sign_result_p2;
    logsumexp(log_term1, log_term2, sign_term1, sign_term2, &log_result_p2, &sign_result_p2);

    // Apply normalization
    log_result_p2 += log_factorial_norm;

    // Store +2 spin
    log_ylm_p2[idx_l] = log_result_p2;
    sign_ylm_p2[idx_l] = sign_result_p2;

    // ============ SPIN -2 ============
    // Inner bracket: alpha * Y_{l-1} - (l-1)*cos*Y_l
    real_t log_inner1 = log_alm + log_Ylm1;
    int8_t sign_inner1 = sign_Ylm1;

    real_t log_inner2 = log((real_t)(l - 1)) + log_beta[ring] + log_Ylm;
    int8_t sign_inner2 = -1 * beta_sign[ring] * sign_Ylm;

    real_t log_inner;
    int8_t sign_inner;
    logsumexp(log_inner1, log_inner2, sign_inner1, sign_inner2, &log_inner, &sign_inner);

    // Multiply by 2m/sin^2
    real_t log_result_m2 = log(2.0) + log_m - log_beta_s2[ring] + log_inner;
    int8_t sign_result_m2 = sign_inner;
    log_result_m2 += log_factorial_norm;

    // Store -2 spin
    log_ylm_m2[idx_l] = log_result_m2;
    sign_ylm_m2[idx_l] = sign_result_m2;
}

void sYLM_l2_compute(int l_max, int n_rings,
                     const real_t* log_beta,
                     const real_t* log_beta_s2,
                     const int8_t* beta_sign,
                     const ylm_log_t* ylm_spin0,
                     real_t* ylm_spin2_p,
                     int8_t* sign_spin2_p,
                     real_t* ylm_spin2_m,
                     int8_t* sign_spin2_m) {
    dim3 block(16, 16, 1);
    dim3 grid(CEILDIV(l_max - 1, 16), CEILDIV(l_max + 1, 16), n_rings);

    sYLM_l2_kernel<<<grid, block>>>(l_max, n_rings,
                                     log_beta, log_beta_s2, beta_sign,
                                     ylm_spin0->log_ylm, ylm_spin0->sign_ylm,
                                     ylm_spin2_p, sign_spin2_p,
                                     ylm_spin2_m, sign_spin2_m);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Normalization and Conversion
// ============================================================================

/**
 * Kernel to normalize YLM values and convert from log to linear
 *
 * Applies 1/sqrt(4*pi) normalization to ALL elements.
 * Reference: jax_healpix/YLM_jax_log.py line 227:
 *   ylm[0] -= 0.5 * jnp.log(4 * jnp.pi)  # Applied to entire array
 */
__global__ void ylm_normalize_kernel(int l_max, int n_rings,
                                      const real_t* log_ylm,
                                      const int8_t* sign_ylm,
                                      real_t* ylm_out) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int ring = blockIdx.z;

    if (l > l_max || m > l || ring >= n_rings) return;

    int lp1 = l_max + 1;
    int idx = INDEX_3D(l, m, ring, lp1, n_rings);

    // Subtract log(sqrt(4*pi)) from ALL elements (JAX line 227)
    real_t log_val = log_ylm[idx] - 0.5 * LOG_4PI;

    // Clamp to prevent overflow
    log_val = clamp_log(log_val);

    // Convert to linear: ylm = sign * exp(log_val)
    ylm_out[idx] = sign_ylm[idx] * exp(log_val);
}

void ylm_normalize(int l_max, int n_rings,
                   const ylm_log_t* ylm_log,
                   real_t* ylm_out) {
    dim3 block(16, 16, 1);
    dim3 grid(CEILDIV(l_max + 1, 16), CEILDIV(l_max + 1, 16), n_rings);

    ylm_normalize_kernel<<<grid, block>>>(l_max, n_rings,
                                           ylm_log->log_ylm, ylm_log->sign_ylm,
                                           ylm_out);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Complete YLM Computation
// ============================================================================

void compute_ylm_spin0(int l_max, int n_rings,
                       const real_t* log_beta,
                       const int8_t* beta_sign,
                       real_t* ylm_out) {
    // Allocate log-space storage
    ylm_log_t* ylm_log = allocate_ylm_log(l_max, n_rings);

    // Compute sin(theta) in log space: log(sin) = 0.5 * log(1 - cos^2)
    real_t* log_beta_s;
    CUDA_CHECK(cudaMalloc(&log_beta_s, n_rings * sizeof(real_t)));

    // Kernel to compute log_beta_s
    // For simplicity, do this on host (can be optimized)
    real_t* h_log_beta = new real_t[n_rings];
    int8_t* h_beta_sign = new int8_t[n_rings];
    real_t* h_log_beta_s = new real_t[n_rings];

    CUDA_CHECK(cudaMemcpy(h_log_beta, log_beta, n_rings * sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_beta_sign, beta_sign, n_rings * sizeof(int8_t), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_rings; i++) {
        // sin^2 = 1 - cos^2
        real_t log_cos2 = 2.0 * h_log_beta[i];
        real_t log_sin2;
        int8_t sign_dummy;
        // Use host version of logsumexp
        real_t max_log = fmax(0.0, log_cos2);
        real_t min_log = fmin(0.0, log_cos2);
        log_sin2 = max_log + log1p(-1.0 * exp(min_log - max_log));
        h_log_beta_s[i] = 0.5 * log_sin2;
        // Clamp to prevent NaN
        if (h_log_beta_s[i] < -100.0) h_log_beta_s[i] = -100.0;
    }

    CUDA_CHECK(cudaMemcpy(log_beta_s, h_log_beta_s, n_rings * sizeof(real_t), cudaMemcpyHostToDevice));

    // Step 1: Compute diagonal elements
    sYLM_ll0_log(l_max, n_rings, log_beta_s, log_beta, beta_sign, ylm_log);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Compute off-diagonal elements via recurrence
    sYLM_l0_all(l_max, n_rings, log_beta, beta_sign, ylm_log);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: Normalize and convert to linear
    ylm_normalize(l_max, n_rings, ylm_log, ylm_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    delete[] h_log_beta;
    delete[] h_beta_sign;
    delete[] h_log_beta_s;
    CUDA_CHECK(cudaFree(log_beta_s));
    free_ylm_log(ylm_log);
}

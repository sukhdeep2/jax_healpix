/**
 * alm2map_v5: Multi-precision spherical harmonic synthesis (alm -> map)
 *
 * Algorithm (following JAX implementation):
 *   1. Scale alm[:,:,m>0] by 2 to account for missing m<0
 *   2. For each ring pair (north + south):
 *      a. Compute Ylm using proper recurrence (same as map2alm_v5)
 *      b. Compute Fmy[m] = sum_l(alm[l,m] * Ylm[l,m]) for each m
 *      c. Synthesize map: map[j] = Real(sum_m(Fmy[m] * e^(im*phi_j)))
 *      d. Apply north-south symmetry: Ylm(-beta) = (-1)^(l+m) * Ylm(beta)
 *
 * Reference: jax_healpix/SPHT_jax.py alm2map() and alm2ring_ns_dot()
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/spht_types.h"
#include <stdio.h>
#include <type_traits>

// Maximum m values per thread (same as map2alm_v5)
#define MAX_M_PER_THREAD 64

// ============================================================================
// Type traits for precision handling (same as map2alm_v5)
// ============================================================================

template<typename T>
struct SynthTraits;

template<>
struct SynthTraits<double> {
    using real_type = double;
    using accum_type = double;
    static constexpr double PI_VAL = M_PI;
    static __device__ __forceinline__ double sqrt_impl(double x) { return sqrt(x); }
    static __device__ __forceinline__ void sincos_impl(double x, double* s, double* c) { sincos(x, s, c); }
};

template<>
struct SynthTraits<float> {
    using real_type = float;
    using accum_type = float;
    static constexpr float PI_VAL = 3.14159265f;
    static __device__ __forceinline__ float sqrt_impl(float x) { return sqrtf(x); }
    static __device__ __forceinline__ void sincos_impl(float x, float* s, float* c) { sincosf(x, s, c); }
};

// ============================================================================
// Device helper: compute ring geometry (same as map2alm_v5)
// ============================================================================

template<typename T>
__device__ __forceinline__
void compute_ring_geometry_synth(int ring_idx, int nside,
                                  T* cos_theta, T* sin_theta,
                                  double* phi_0, int* n_pixels) {
    using Traits = SynthTraits<T>;
    int ring_i = ring_idx + 1;

    if (ring_i < nside) {
        // North polar cap
        T i2_over_3n2 = T(ring_i * ring_i) / T(3.0 * nside * nside);
        *cos_theta = T(1.0) - i2_over_3n2;
        *sin_theta = Traits::sqrt_impl(T(1.0) - (*cos_theta) * (*cos_theta));
        *phi_0 = M_PI / (2.0 * ring_i) * 0.5;
        *n_pixels = 4 * ring_i;
    } else if (ring_i > 3 * nside) {
        // South polar cap
        int mirror_i = 4 * nside - ring_i;
        T i2_over_3n2 = T(mirror_i * mirror_i) / T(3.0 * nside * nside);
        *cos_theta = -(T(1.0) - i2_over_3n2);
        *sin_theta = Traits::sqrt_impl(T(1.0) - (*cos_theta) * (*cos_theta));
        *phi_0 = M_PI / (2.0 * mirror_i) * 0.5;
        *n_pixels = 4 * mirror_i;
    } else {
        // Equatorial belt
        *cos_theta = T(4.0 / 3.0) - T(2.0 * ring_i) / T(3.0 * nside);
        *sin_theta = Traits::sqrt_impl(T(1.0) - (*cos_theta) * (*cos_theta));
        int s = (ring_i % 2 == 0) ? 1 : 2;
        *phi_0 = M_PI / (2.0 * nside) * (1.0 - s / 2.0);
        *n_pixels = 4 * nside;
    }
}

// ============================================================================
// Device helper: Ylm recurrence for north ring
// Computes Fmy[m] = sum_l(alm[l,m] * Ylm[l,m])
// T = storage type, R = recurrence precision type
// ============================================================================

template<typename T, typename R>
__device__ void ylm_synthesis_column_north(
    int m, int l_max,
    R cos_theta, R sin_theta,
    const T* alm_real, const T* alm_imag,  // [lp1, lp1] - pre-scaled alm
    int lp1,
    R* fmy_re, R* fmy_im  // Output: Fmy for this m
) {
    using Traits = SynthTraits<R>;

    // Initialize accumulator
    *fmy_re = R(0);
    *fmy_im = R(0);

    // Y[m,m] = (-1)^m * sin^m(theta) * sqrt((2m+1)!!/(2m)!!) / sqrt(4*pi)
    R Ymm = R(1.0) / Traits::sqrt_impl(R(4.0 * Traits::PI_VAL));

    for (int k = 1; k <= m; k++) {
        Ymm *= -sin_theta * Traits::sqrt_impl(R(2.0 * k + 1.0) / R(2.0 * k));
    }

    // Accumulate Y[m,m] contribution
    {
        int l = m;
        R alm_re = R(alm_real[l * lp1 + m]);
        R alm_im = R(alm_imag[l * lp1 + m]);
        *fmy_re += Ymm * alm_re;
        *fmy_im += Ymm * alm_im;
    }

    if (m == l_max) return;

    // Y[m+1,m] = cos(theta) * sqrt(2m+3) * Y[m,m]
    R Ym1m = cos_theta * Traits::sqrt_impl(R(2.0 * m + 3.0)) * Ymm;

    {
        int l = m + 1;
        R alm_re = R(alm_real[l * lp1 + m]);
        R alm_im = R(alm_imag[l * lp1 + m]);
        *fmy_re += Ym1m * alm_re;
        *fmy_im += Ym1m * alm_im;
    }

    if (m + 1 == l_max) return;

    // Recurrence: Y[l,m] = A[l,m] * cos(theta) * Y[l-1,m] - B[l,m] * Y[l-2,m]
    R Ylm2 = Ymm;
    R Ylm1 = Ym1m;

    for (int l = m + 2; l <= l_max; l++) {
        R l2 = R(l * l);
        R m2 = R(m * m);
        R lm1_2 = R((l - 1) * (l - 1));

        R A_lm = Traits::sqrt_impl((R(4.0) * l2 - R(1.0)) / (l2 - m2));
        R B_lm = Traits::sqrt_impl((R(2.0 * l + 1.0)) / (R(2.0 * l - 3.0)) * (lm1_2 - m2) / (l2 - m2));

        R Ylm = A_lm * cos_theta * Ylm1 - B_lm * Ylm2;

        R alm_re = R(alm_real[l * lp1 + m]);
        R alm_im = R(alm_imag[l * lp1 + m]);
        *fmy_re += Ylm * alm_re;
        *fmy_im += Ylm * alm_im;

        Ylm2 = Ylm1;
        Ylm1 = Ylm;
    }
}

// ============================================================================
// Device helper: Ylm recurrence for south ring with (-1)^(l+m) symmetry
// Computes Fmy[m] = sum_l(alm[l,m] * (-1)^(l+m) * Ylm[l,m])
// Using the symmetry: Ylm(-cos_theta) = (-1)^(l+m) * Ylm(cos_theta)
// ============================================================================

template<typename T, typename R>
__device__ void ylm_synthesis_column_south(
    int m, int l_max,
    R cos_theta, R sin_theta,  // Note: using north theta; symmetry handles south
    const T* alm_real, const T* alm_imag,
    int lp1,
    R* fmy_re, R* fmy_im
) {
    using Traits = SynthTraits<R>;

    *fmy_re = R(0);
    *fmy_im = R(0);

    // Y[m,m]
    R Ymm = R(1.0) / Traits::sqrt_impl(R(4.0 * Traits::PI_VAL));
    for (int k = 1; k <= m; k++) {
        Ymm *= -sin_theta * Traits::sqrt_impl(R(2.0 * k + 1.0) / R(2.0 * k));
    }

    // For south: Ylm(-beta) = (-1)^(l+m) * Ylm(beta)
    {
        int l = m;
        int sign_ns = ((l + m) % 2 == 0) ? 1 : -1;
        R alm_re = R(alm_real[l * lp1 + m]);
        R alm_im = R(alm_imag[l * lp1 + m]);
        *fmy_re += R(sign_ns) * Ymm * alm_re;
        *fmy_im += R(sign_ns) * Ymm * alm_im;
    }

    if (m == l_max) return;

    // Y[m+1,m]
    R Ym1m = cos_theta * Traits::sqrt_impl(R(2.0 * m + 3.0)) * Ymm;

    {
        int l = m + 1;
        int sign_ns = ((l + m) % 2 == 0) ? 1 : -1;
        R alm_re = R(alm_real[l * lp1 + m]);
        R alm_im = R(alm_imag[l * lp1 + m]);
        *fmy_re += R(sign_ns) * Ym1m * alm_re;
        *fmy_im += R(sign_ns) * Ym1m * alm_im;
    }

    if (m + 1 == l_max) return;

    // Recurrence
    R Ylm2 = Ymm;
    R Ylm1 = Ym1m;

    for (int l = m + 2; l <= l_max; l++) {
        R l2 = R(l * l);
        R m2 = R(m * m);
        R lm1_2 = R((l - 1) * (l - 1));

        R A_lm = Traits::sqrt_impl((R(4.0) * l2 - R(1.0)) / (l2 - m2));
        R B_lm = Traits::sqrt_impl((R(2.0 * l + 1.0)) / (R(2.0 * l - 3.0)) * (lm1_2 - m2) / (l2 - m2));

        R Ylm = A_lm * cos_theta * Ylm1 - B_lm * Ylm2;

        int sign_ns = ((l + m) % 2 == 0) ? 1 : -1;
        R alm_re = R(alm_real[l * lp1 + m]);
        R alm_im = R(alm_imag[l * lp1 + m]);
        *fmy_re += R(sign_ns) * Ylm * alm_re;
        *fmy_im += R(sign_ns) * Ylm * alm_im;

        Ylm2 = Ylm1;
        Ylm1 = Ylm;
    }
}

// ============================================================================
// Kernel to scale alm: multiply m>0 by 2
// ============================================================================

template<typename T>
__global__ void scale_alm_m_kernel(
    int n_maps, int lp1,
    const T* __restrict__ alm_in_real,
    const T* __restrict__ alm_in_imag,
    T* __restrict__ alm_out_real,
    T* __restrict__ alm_out_imag
) {
    int t = blockIdx.x;
    int l = blockIdx.y * blockDim.x + threadIdx.x;
    int m = blockIdx.z * blockDim.y + threadIdx.y;

    if (t >= n_maps || l >= lp1 || m > l) return;

    size_t idx = (size_t)t * lp1 * lp1 + l * lp1 + m;

    T scale = (m > 0) ? T(2.0) : T(1.0);
    alm_out_real[idx] = alm_in_real[idx] * scale;
    alm_out_imag[idx] = alm_in_imag[idx] * scale;
}

// ============================================================================
// Main synthesis kernel
// T = storage type, R = recurrence precision type
// One block per ring pair (north + corresponding south)
// ============================================================================

template<typename T, typename R>
__global__ void alm2map_fused_kernel_v5(
    int nside, int l_max, int n_maps, int n_rings,
    const T* __restrict__ alm_real,   // [n_maps, lp1, lp1] - already scaled by 2 for m>0
    const T* __restrict__ alm_imag,   // [n_maps, lp1, lp1]
    T* __restrict__ map_out           // [n_maps, n_rings, 4*nside]
) {
    using Traits = SynthTraits<T>;
    using accum_t = typename Traits::accum_type;

    int north_ring_idx = blockIdx.x;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;
    int max_ring_pixels = 4 * nside;

    if (north_ring_idx >= n_north_rings) return;

    int south_ring_idx = n_rings - 1 - north_ring_idx;
    bool is_equator = (north_ring_idx == 2 * nside - 1);

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Compute ring geometry for north ring
    T cos_theta_n_t, sin_theta_n_t;
    R cos_theta_n, sin_theta_n;
    double phi_0_n;
    int n_pixels_n;
    compute_ring_geometry_synth<T>(north_ring_idx, nside,
                                    &cos_theta_n_t, &sin_theta_n_t,
                                    &phi_0_n, &n_pixels_n);
    cos_theta_n = R(cos_theta_n_t);
    sin_theta_n = R(sin_theta_n_t);

    // South ring geometry
    T cos_theta_s_t, sin_theta_s_t;
    double phi_0_s = 0.0;
    int n_pixels_s = 0;
    if (!is_equator) {
        compute_ring_geometry_synth<T>(south_ring_idx, nside,
                                        &cos_theta_s_t, &sin_theta_s_t,
                                        &phi_0_s, &n_pixels_s);
    }

    // Shared memory for Fmy values (used for pixel synthesis)
    extern __shared__ char shared_mem[];
    accum_t* fmy_north_re = (accum_t*)shared_mem;
    accum_t* fmy_north_im = fmy_north_re + lp1;
    accum_t* fmy_south_re = fmy_north_im + lp1;
    accum_t* fmy_south_im = fmy_south_re + lp1;

    // Process each map
    for (int t = 0; t < n_maps; t++) {
        const T* alm_real_t = alm_real + (size_t)t * lp1 * lp1;
        const T* alm_imag_t = alm_imag + (size_t)t * lp1 * lp1;
        T* map_t = map_out + (size_t)t * n_rings * max_ring_pixels;

        // Phase 1: Compute Fmy[m] = sum_l(alm[l,m] * Ylm) for each m
        for (int m = tid; m <= l_max; m += block_size) {
            // North ring
            R fmy_re_n, fmy_im_n;
            ylm_synthesis_column_north<T, R>(m, l_max, cos_theta_n, sin_theta_n,
                                              alm_real_t, alm_imag_t, lp1,
                                              &fmy_re_n, &fmy_im_n);

            // Apply phase factor: Fmy *= e^(i * m * phi_0)
            // For alm2map (synthesis), phase is +1 (vs -1 for map2alm)
            double phase_n = m * phi_0_n;
            double cos_pn = cos(phase_n), sin_pn = sin(phase_n);
            fmy_north_re[m] = accum_t(fmy_re_n * cos_pn - fmy_im_n * sin_pn);
            fmy_north_im[m] = accum_t(fmy_re_n * sin_pn + fmy_im_n * cos_pn);

            // South ring (using symmetry: Ylm(-beta) = (-1)^(l+m) * Ylm(beta))
            if (!is_equator) {
                R fmy_re_s, fmy_im_s;
                ylm_synthesis_column_south<T, R>(m, l_max, cos_theta_n, sin_theta_n,
                                                  alm_real_t, alm_imag_t, lp1,
                                                  &fmy_re_s, &fmy_im_s);

                double phase_s = m * phi_0_s;
                double cos_ps = cos(phase_s), sin_ps = sin(phase_s);
                fmy_south_re[m] = accum_t(fmy_re_s * cos_ps - fmy_im_s * sin_ps);
                fmy_south_im[m] = accum_t(fmy_re_s * sin_ps + fmy_im_s * cos_ps);
            }
        }

        __syncthreads();

        // Phase 2: Synthesize map pixels
        // map[j] = Real(sum_m(Fmy[m] * e^(im * phi_j)))
        // where phi_j = 2 * pi * j / n_pixels

        // North ring pixels
        for (int j = tid; j < n_pixels_n; j += block_size) {
            accum_t sum_re = accum_t(0);

            for (int m = 0; m <= l_max; m++) {
                // Phase for this pixel: e^(im * 2*pi*j/n_pixels)
                double phi_j = 2.0 * Traits::PI_VAL * m * j / n_pixels_n;
                double cos_phi = cos(phi_j);
                double sin_phi = sin(phi_j);

                // Re(Fmy * e^(im*phi)) = Fmy_re * cos(m*phi) - Fmy_im * sin(m*phi)
                sum_re += fmy_north_re[m] * cos_phi - fmy_north_im[m] * sin_phi;
            }

            map_t[north_ring_idx * max_ring_pixels + j] = T(sum_re);
        }

        // South ring pixels
        if (!is_equator) {
            for (int j = tid; j < n_pixels_s; j += block_size) {
                accum_t sum_re = accum_t(0);

                for (int m = 0; m <= l_max; m++) {
                    double phi_j = 2.0 * Traits::PI_VAL * m * j / n_pixels_s;
                    double cos_phi = cos(phi_j);
                    double sin_phi = sin(phi_j);

                    sum_re += fmy_south_re[m] * cos_phi - fmy_south_im[m] * sin_phi;
                }

                map_t[south_ring_idx * max_ring_pixels + j] = T(sum_re);
            }
        }

        __syncthreads();
    }
}

// ============================================================================
// Host wrapper implementation
// T = storage type, R = recurrence precision
// ============================================================================

template<typename T, typename R>
void alm2map_cuda_v5_impl(int nside, int l_max, int n_maps,
                           const T* alm_in_real, const T* alm_in_imag,
                           T* map_out) {
    int n_rings = 4 * nside - 1;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;
    int max_ring_pixels = 4 * nside;

    // Allocate scaled alm (m>0 multiplied by 2)
    T *alm_scaled_real, *alm_scaled_imag;
    size_t alm_size = (size_t)n_maps * lp1 * lp1 * sizeof(T);
    CUDA_CHECK(cudaMalloc(&alm_scaled_real, alm_size));
    CUDA_CHECK(cudaMalloc(&alm_scaled_imag, alm_size));

    // Scale alm: m>0 by 2
    dim3 block_scale(16, 16);
    dim3 grid_scale(n_maps, CEILDIV(lp1, 16), CEILDIV(lp1, 16));
    scale_alm_m_kernel<T><<<grid_scale, block_scale>>>(
        n_maps, lp1, alm_in_real, alm_in_imag, alm_scaled_real, alm_scaled_imag
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize output map to zero
    CUDA_CHECK(cudaMemset(map_out, 0, (size_t)n_maps * n_rings * max_ring_pixels * sizeof(T)));

    // Launch synthesis kernel
    int block_size = 256;
    int m_per_thread = (lp1 + block_size - 1) / block_size;
    if (m_per_thread > MAX_M_PER_THREAD) {
        fprintf(stderr, "Error: l_max=%d requires %d m values per thread, max is %d\n",
                l_max, m_per_thread, MAX_M_PER_THREAD);
        cudaFree(alm_scaled_real);
        cudaFree(alm_scaled_imag);
        return;
    }

    // Shared memory: 4 arrays of lp1 accum_t values
    using accum_t = typename SynthTraits<T>::accum_type;
    size_t shared_size = 4 * lp1 * sizeof(accum_t);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (shared_size > prop.sharedMemPerBlock) {
        fprintf(stderr, "Error: Required shared memory %zu exceeds limit %zu\n",
                shared_size, prop.sharedMemPerBlock);
        cudaFree(alm_scaled_real);
        cudaFree(alm_scaled_imag);
        return;
    }

    int grid_size = n_north_rings;
    alm2map_fused_kernel_v5<T, R><<<grid_size, block_size, shared_size>>>(
        nside, l_max, n_maps, n_rings,
        alm_scaled_real, alm_scaled_imag, map_out
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    cudaFree(alm_scaled_real);
    cudaFree(alm_scaled_imag);
}

// ============================================================================
// C API entry points
// Naming: alm2map_cuda_v5_{storage}_{recurrence}
// ============================================================================

extern "C" {

// Float64 storage, Float64 recurrence (highest accuracy, default)
void alm2map_cuda_v5_f64_f64(int nside, int l_max, int n_maps,
                              const double* alm_in_real, const double* alm_in_imag,
                              double* map_out) {
    alm2map_cuda_v5_impl<double, double>(nside, l_max, n_maps,
                                          alm_in_real, alm_in_imag, map_out);
}

// Float64 storage, Float32 recurrence
void alm2map_cuda_v5_f64_f32(int nside, int l_max, int n_maps,
                              const double* alm_in_real, const double* alm_in_imag,
                              double* map_out) {
    alm2map_cuda_v5_impl<double, float>(nside, l_max, n_maps,
                                         alm_in_real, alm_in_imag, map_out);
}

// Float32 storage, Float64 recurrence
void alm2map_cuda_v5_f32_f64(int nside, int l_max, int n_maps,
                              const float* alm_in_real, const float* alm_in_imag,
                              float* map_out) {
    alm2map_cuda_v5_impl<float, double>(nside, l_max, n_maps,
                                         alm_in_real, alm_in_imag, map_out);
}

// Float32 storage, Float32 recurrence (fastest)
void alm2map_cuda_v5_f32_f32(int nside, int l_max, int n_maps,
                              const float* alm_in_real, const float* alm_in_imag,
                              float* map_out) {
    alm2map_cuda_v5_impl<float, float>(nside, l_max, n_maps,
                                        alm_in_real, alm_in_imag, map_out);
}

// Backwards-compatible aliases
void alm2map_cuda_v5_f64(int nside, int l_max, int n_maps,
                          const double* alm_in_real, const double* alm_in_imag,
                          double* map_out) {
    alm2map_cuda_v5_f64_f64(nside, l_max, n_maps, alm_in_real, alm_in_imag, map_out);
}

void alm2map_cuda_v5_f32(int nside, int l_max, int n_maps,
                          const float* alm_in_real, const float* alm_in_imag,
                          float* map_out) {
    alm2map_cuda_v5_f32_f32(nside, l_max, n_maps, alm_in_real, alm_in_imag, map_out);
}

// Wrapper that takes complex_t input for compatibility with existing API
void alm2map_cuda_v5(int nside, int l_max, int n_maps,
                      const complex_t* alm_in, real_t* map_out) {
    int lp1 = l_max + 1;
    size_t alm_count = (size_t)n_maps * lp1 * lp1;

    // Separate complex alm into real and imag parts
    double *alm_real, *alm_imag;
    CUDA_CHECK(cudaMalloc(&alm_real, alm_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&alm_imag, alm_count * sizeof(double)));

    // Copy from interleaved complex format
    CUDA_CHECK(cudaMemcpy2D(alm_real, sizeof(double),
                            (const double*)alm_in, 2 * sizeof(double),
                            sizeof(double), alm_count,
                            cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(alm_imag, sizeof(double),
                            (const double*)alm_in + 1, 2 * sizeof(double),
                            sizeof(double), alm_count,
                            cudaMemcpyDeviceToDevice));

    alm2map_cuda_v5_impl<double, double>(nside, l_max, n_maps,
                                          alm_real, alm_imag, map_out);

    cudaFree(alm_real);
    cudaFree(alm_imag);
}

} // extern "C"

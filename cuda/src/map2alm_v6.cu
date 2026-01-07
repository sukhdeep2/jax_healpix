/**
 * map2alm_v6: Optimal warp-per-m algorithm with NO atomics
 *
 * Key optimizations over v5:
 *   1. Two-phase approach: Gm computation + warp reduction
 *   2. NO atomic adds - uses warp shuffle reduction
 *   3. Gm cached in shared memory (loaded once per m)
 *   4. Ylm recurrence state cached per lane
 *   5. Only 1 warp-level sync for shared memory load
 *
 * Supports: float64, float32, bfloat16
 *
 * Algorithm:
 *   Phase 1: For each (ring, m) compute Gm via direct DFT
 *   Phase 2: For each m, reduce across rings using warp shuffles
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cufft.h>
#include "../include/spht_types.h"
#include "../include/spht_transform_config.cuh"
#include "../include/log_arithmetic.cuh"
#include "../include/bluestein_fft.h"
#include <stdio.h>
#include <type_traits>
#include <map>
#include <mutex>

// ============================================================================
// Global Phase1 Method (definition - declared in spht_transform_config.cuh)
// ============================================================================

Phase1Method g_phase1_method = Phase1Method::DFT;

// cuFFT plan cache is defined in bluestein_core.cu, use the shared function
// get_cached_cufft_plan() declared in spht_transform_config.cuh
static inline cufftHandle get_cached_plan(int fft_size, int batch, cufftType type) {
    return get_cached_cufft_plan(fft_size, batch, type);
}

// ============================================================================
// Workspace Cache - Eliminates malloc/free overhead on repeated calls
// ============================================================================

template<typename T, typename R>
struct Map2almWorkspace {
    // Buffer pointers
    R* Gm_north_re = nullptr;
    R* Gm_north_im = nullptr;
    R* Gm_south_re = nullptr;
    R* Gm_south_im = nullptr;
    R* cos_theta = nullptr;
    R* sin_theta = nullptr;
    int* ring_sizes = nullptr;
    void* chirped_data = nullptr;
    void* conj_chirp_fft = nullptr;
    R* bs_cos_theta = nullptr;
    R* bs_sin_theta = nullptr;
    R* eq_phi_0 = nullptr;
    void* fft_out = nullptr;

    // Cached sizes for reallocation check
    size_t gm_size = 0;
    size_t geom_size = 0;
    size_t ring_sizes_size = 0;
    size_t chirped_data_size = 0;
    size_t conj_chirp_size = 0;
    size_t eq_phi0_size = 0;
    size_t fft_out_size = 0;

    void ensure_size(size_t new_gm, size_t new_geom, size_t new_ring_sizes) {
        if (new_gm > gm_size) {
            if (Gm_north_re) cudaFree(Gm_north_re);
            if (Gm_north_im) cudaFree(Gm_north_im);
            if (Gm_south_re) cudaFree(Gm_south_re);
            if (Gm_south_im) cudaFree(Gm_south_im);
            cudaMalloc(&Gm_north_re, new_gm);
            cudaMalloc(&Gm_north_im, new_gm);
            cudaMalloc(&Gm_south_re, new_gm);
            cudaMalloc(&Gm_south_im, new_gm);
            gm_size = new_gm;
        }
        if (new_geom > geom_size) {
            if (cos_theta) cudaFree(cos_theta);
            if (sin_theta) cudaFree(sin_theta);
            if (bs_cos_theta) cudaFree(bs_cos_theta);
            if (bs_sin_theta) cudaFree(bs_sin_theta);
            cudaMalloc(&cos_theta, new_geom);
            cudaMalloc(&sin_theta, new_geom);
            cudaMalloc(&bs_cos_theta, new_geom);
            cudaMalloc(&bs_sin_theta, new_geom);
            geom_size = new_geom;
        }
        if (new_ring_sizes > ring_sizes_size) {
            if (ring_sizes) cudaFree(ring_sizes);
            cudaMalloc(&ring_sizes, new_ring_sizes);
            ring_sizes_size = new_ring_sizes;
        }
    }

    void ensure_bluestein(size_t new_chirped, size_t new_conj) {
        if (new_chirped > chirped_data_size) {
            if (chirped_data) cudaFree(chirped_data);
            cudaMalloc(&chirped_data, new_chirped);
            chirped_data_size = new_chirped;
        }
        if (new_conj > conj_chirp_size) {
            if (conj_chirp_fft) cudaFree(conj_chirp_fft);
            cudaMalloc(&conj_chirp_fft, new_conj);
            conj_chirp_size = new_conj;
        }
    }

    void ensure_fft(size_t new_eq_phi0, size_t new_fft_out) {
        if (new_eq_phi0 > eq_phi0_size) {
            if (eq_phi_0) cudaFree(eq_phi_0);
            cudaMalloc(&eq_phi_0, new_eq_phi0);
            eq_phi0_size = new_eq_phi0;
        }
        if (new_fft_out > fft_out_size) {
            if (fft_out) cudaFree(fft_out);
            cudaMalloc(&fft_out, new_fft_out);
            fft_out_size = new_fft_out;
        }
    }

    ~Map2almWorkspace() {
        if (Gm_north_re) cudaFree(Gm_north_re);
        if (Gm_north_im) cudaFree(Gm_north_im);
        if (Gm_south_re) cudaFree(Gm_south_re);
        if (Gm_south_im) cudaFree(Gm_south_im);
        if (cos_theta) cudaFree(cos_theta);
        if (sin_theta) cudaFree(sin_theta);
        if (ring_sizes) cudaFree(ring_sizes);
        if (chirped_data) cudaFree(chirped_data);
        if (conj_chirp_fft) cudaFree(conj_chirp_fft);
        if (bs_cos_theta) cudaFree(bs_cos_theta);
        if (bs_sin_theta) cudaFree(bs_sin_theta);
        if (eq_phi_0) cudaFree(eq_phi_0);
        if (fft_out) cudaFree(fft_out);
    }
};

// Global workspace instances (one per precision combination)
static Map2almWorkspace<double, double> g_m2a_ws_f64_f64;
static Map2almWorkspace<double, float>  g_m2a_ws_f64_f32;
static Map2almWorkspace<float, double>  g_m2a_ws_f32_f64;
static Map2almWorkspace<float, float>   g_m2a_ws_f32_f32;
static Map2almWorkspace<__nv_bfloat16, float> g_m2a_ws_bf16_f32;

template<typename T, typename R>
Map2almWorkspace<T, R>& get_map2alm_workspace();

template<> Map2almWorkspace<double, double>& get_map2alm_workspace<double, double>() { return g_m2a_ws_f64_f64; }
template<> Map2almWorkspace<double, float>&  get_map2alm_workspace<double, float>()  { return g_m2a_ws_f64_f32; }
template<> Map2almWorkspace<float, double>&  get_map2alm_workspace<float, double>()  { return g_m2a_ws_f32_f64; }
template<> Map2almWorkspace<float, float>&   get_map2alm_workspace<float, float>()   { return g_m2a_ws_f32_f32; }
template<> Map2almWorkspace<__nv_bfloat16, float>& get_map2alm_workspace<__nv_bfloat16, float>() { return g_m2a_ws_bf16_f32; }

// ============================================================================
// Type traits for multi-precision support
// ============================================================================

template<typename T>
struct V6Traits;

template<>
struct V6Traits<double> {
    using storage_t = double;
    using compute_t = double;
    using complex_storage_t = double2;
    static constexpr double PI_VAL = 3.14159265358979323846;
    static constexpr double LOG_4PI_VAL = 2.5310242469692907;

    static __device__ __forceinline__ double sqrt_d(double x) { return sqrt(x); }
    static __device__ __forceinline__ double exp_d(double x) { return exp(x); }
    static __device__ __forceinline__ double log_d(double x) { return log(x); }
    static __device__ __forceinline__ void sincos_d(double x, double* s, double* c) { sincos(x, s, c); }
    static __device__ __forceinline__ double load(const double* p) { return __ldg(p); }
    static __device__ __forceinline__ double2 load2(const double2* p) { return *p; }
};

template<>
struct V6Traits<float> {
    using storage_t = float;
    using compute_t = float;
    using complex_storage_t = float2;
    static constexpr float PI_VAL = 3.14159265f;
    static constexpr float LOG_4PI_VAL = 2.53102425f;

    static __device__ __forceinline__ float sqrt_d(float x) { return sqrtf(x); }
    static __device__ __forceinline__ float exp_d(float x) { return expf(x); }
    static __device__ __forceinline__ float log_d(float x) { return logf(x); }
    static __device__ __forceinline__ void sincos_d(float x, float* s, float* c) { sincosf(x, s, c); }
    static __device__ __forceinline__ float load(const float* p) { return __ldg(p); }
    static __device__ __forceinline__ float2 load2(const float2* p) { return *p; }
};

template<>
struct V6Traits<__nv_bfloat16> {
    using storage_t = __nv_bfloat16;
    using compute_t = float;  // Compute in float for stability
    using complex_storage_t = __nv_bfloat162;
    static constexpr float PI_VAL = 3.14159265f;
    static constexpr float LOG_4PI_VAL = 2.53102425f;

    static __device__ __forceinline__ float sqrt_d(float x) { return sqrtf(x); }
    static __device__ __forceinline__ float exp_d(float x) { return expf(x); }
    static __device__ __forceinline__ float log_d(float x) { return logf(x); }
    static __device__ __forceinline__ void sincos_d(float x, float* s, float* c) { sincosf(x, s, c); }
    static __device__ __forceinline__ float load(const __nv_bfloat16* p) { return __bfloat162float(__ldg(p)); }
    static __device__ __forceinline__ float2 load2(const __nv_bfloat162* p) {
        __nv_bfloat162 v = *p;
        return make_float2(__bfloat162float(v.x), __bfloat162float(v.y));
    }
};

// ============================================================================
// Device helper: Compute ring geometry
// ============================================================================

template<typename T>
__device__ __forceinline__ void compute_ring_geom_v6(
    int ring_idx, int nside,
    T* cos_theta, T* sin_theta, T* phi_0, int* n_pixels
) {
    using Traits = V6Traits<T>;
    using C = typename Traits::compute_t;

    int ring_i = ring_idx + 1;  // 1-indexed
    C cos_th, sin_th, phi0;
    int npix;

    if (ring_i < nside) {
        // North polar cap
        C i2_3n2 = C(ring_i * ring_i) / C(3.0 * nside * nside);
        cos_th = C(1.0) - i2_3n2;
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        phi0 = C(Traits::PI_VAL) / C(2.0 * ring_i) * C(0.5);
        npix = 4 * ring_i;
    } else if (ring_i > 3 * nside) {
        // South polar cap
        int mirror_i = 4 * nside - ring_i;
        C i2_3n2 = C(mirror_i * mirror_i) / C(3.0 * nside * nside);
        cos_th = -(C(1.0) - i2_3n2);
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        phi0 = C(Traits::PI_VAL) / C(2.0 * mirror_i) * C(0.5);
        npix = 4 * mirror_i;
    } else {
        // Equatorial belt
        cos_th = C(4.0 / 3.0) - C(2.0 * ring_i) / C(3.0 * nside);
        sin_th = Traits::sqrt_d(C(1.0) - cos_th * cos_th);
        int s = (ring_i % 2 == 0) ? 1 : 2;
        phi0 = C(Traits::PI_VAL) / C(2.0 * nside) * C(1.0 - s / 2.0);
        npix = 4 * nside;
    }

    *cos_theta = T(cos_th);
    *sin_theta = T(sin_th);
    *phi_0 = T(phi0);
    *n_pixels = npix;
}

// ============================================================================
// Phase 1: Compute Gm_north and Gm_south for all (ring_pair, m)
// Each block handles one north ring pair, threads handle m values
// North/South separation allows Phase 2 to load half the data at a time
// ============================================================================

template<typename T, typename R>
__global__ void compute_gm_kernel_v6(
    int nside, int l_max, int n_maps, int n_rings,
    const T* __restrict__ map_in,
    R* __restrict__ Gm_north_re,   // [n_maps, lp1, n_north_rings] - optimized for coalesced reads
    R* __restrict__ Gm_north_im,
    R* __restrict__ Gm_south_re,
    R* __restrict__ Gm_south_im,
    R* __restrict__ cos_theta_out,  // [n_north_rings]
    R* __restrict__ sin_theta_out
) {
    using Traits = V6Traits<T>;
    using C = typename Traits::compute_t;

    int north_ring = blockIdx.x;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;
    int max_pix = 4 * nside;

    if (north_ring >= n_north_rings) return;

    int south_ring = n_rings - 1 - north_ring;
    bool is_equator = (north_ring == 2 * nside - 1);

    // Compute geometry
    T cos_th_n, sin_th_n, phi0_n;
    int n_pix_n;
    compute_ring_geom_v6<T>(north_ring, nside, &cos_th_n, &sin_th_n, &phi0_n, &n_pix_n);

    T cos_th_s, sin_th_s, phi0_s;
    int n_pix_s = 0;
    if (!is_equator) {
        compute_ring_geom_v6<T>(south_ring, nside, &cos_th_s, &sin_th_s, &phi0_s, &n_pix_s);
    }

    // Store geometry (thread 0 only)
    if (threadIdx.x == 0) {
        cos_theta_out[north_ring] = R(cos_th_n);
        sin_theta_out[north_ring] = R(sin_th_n);
    }

    // Process each map
    for (int t = 0; t < n_maps; t++) {
        const T* map_t = map_in + (size_t)t * n_rings * max_pix;

        // Each thread handles one or more m values
        for (int m = threadIdx.x; m <= l_max; m += blockDim.x) {
            C gn_re = C(0), gn_im = C(0);
            C gs_re = C(0), gs_im = C(0);

            // North ring DFT with proper phase handling
            // G_m = exp(-i*m*phi_0) * sum_j f(phi_j) * exp(-i*m_eff*j*2π/n_pix)
            // where m_eff = m % n_pix for the periodic DFT part
            C dphi_n = C(2.0 * Traits::PI_VAL) / C(n_pix_n);
            int m_eff_n = m % n_pix_n;

            for (int j = 0; j < n_pix_n; j++) {
                C val = C(Traits::load(&map_t[north_ring * max_pix + j]));
                // Phase = -m*phi_0 - m_eff*j*dphi (phi_0 uses m, not m_eff!)
                C angle = -C(m) * C(phi0_n) - C(m_eff_n) * C(j) * dphi_n;
                C c, s;
                Traits::sincos_d(angle, &s, &c);
                gn_re += val * c;
                gn_im += val * s;
            }

            // South ring DFT
            if (!is_equator && n_pix_s > 0) {
                C dphi_s = C(2.0 * Traits::PI_VAL) / C(n_pix_s);
                int m_eff_s = m % n_pix_s;

                for (int j = 0; j < n_pix_s; j++) {
                    C val = C(Traits::load(&map_t[south_ring * max_pix + j]));
                    C angle = -C(m) * C(phi0_s) - C(m_eff_s) * C(j) * dphi_s;
                    C c, s;
                    Traits::sincos_d(angle, &s, &c);
                    gs_re += val * c;
                    gs_im += val * s;
                }
            }

            // Store north and south separately (combination done in Phase 2)
            // Layout: [n_maps, lp1, n_north_rings] for coalesced reads in Phase 2
            size_t idx = (size_t)t * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            Gm_north_re[idx] = R(gn_re);
            Gm_north_im[idx] = R(gn_im);
            Gm_south_re[idx] = R(gs_re);
            Gm_south_im[idx] = R(gs_im);
        }
    }
}

// ============================================================================
// Hybrid FFT/DFT: Polar DFT kernel (for rings 0 to nside-2)
// These rings have varying sizes (4, 8, ..., 4*(nside-1))
// DFT is acceptable here since rings are small
// ============================================================================

template<typename T, typename R>
__global__ void compute_gm_polar_dft_kernel(
    int nside, int l_max, int n_maps, int n_rings,
    const T* __restrict__ map_in,
    R* __restrict__ Gm_north_re,   // [n_maps, lp1, n_north_rings]
    R* __restrict__ Gm_north_im,
    R* __restrict__ Gm_south_re,
    R* __restrict__ Gm_south_im,
    R* __restrict__ cos_theta_out,  // [n_north_rings]
    R* __restrict__ sin_theta_out
) {
    using Traits = V6Traits<T>;
    using C = typename Traits::compute_t;

    // Only process polar rings: north_ring 0 to nside-2
    int north_ring = blockIdx.x;
    int n_polar_rings = nside - 1;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;
    int max_pix = 4 * nside;

    if (north_ring >= n_polar_rings) return;

    int south_ring = n_rings - 1 - north_ring;

    // Compute geometry for north polar ring
    T cos_th_n, sin_th_n, phi0_n;
    int n_pix_n;
    compute_ring_geom_v6<T>(north_ring, nside, &cos_th_n, &sin_th_n, &phi0_n, &n_pix_n);

    // South polar ring geometry
    T cos_th_s, sin_th_s, phi0_s;
    int n_pix_s;
    compute_ring_geom_v6<T>(south_ring, nside, &cos_th_s, &sin_th_s, &phi0_s, &n_pix_s);

    // Store geometry (thread 0 only)
    if (threadIdx.x == 0) {
        cos_theta_out[north_ring] = R(cos_th_n);
        sin_theta_out[north_ring] = R(sin_th_n);
    }

    // Process each map
    for (int t = 0; t < n_maps; t++) {
        const T* map_t = map_in + (size_t)t * n_rings * max_pix;

        // Each thread handles one or more m values
        for (int m = threadIdx.x; m <= l_max; m += blockDim.x) {
            C gn_re = C(0), gn_im = C(0);
            C gs_re = C(0), gs_im = C(0);

            // North ring DFT
            C dphi_n = C(2.0 * Traits::PI_VAL) / C(n_pix_n);
            int m_eff_n = m % n_pix_n;

            for (int j = 0; j < n_pix_n; j++) {
                C val = C(Traits::load(&map_t[north_ring * max_pix + j]));
                C angle = -C(m) * C(phi0_n) - C(m_eff_n) * C(j) * dphi_n;
                C c, s;
                Traits::sincos_d(angle, &s, &c);
                gn_re += val * c;
                gn_im += val * s;
            }

            // South ring DFT
            C dphi_s = C(2.0 * Traits::PI_VAL) / C(n_pix_s);
            int m_eff_s = m % n_pix_s;

            for (int j = 0; j < n_pix_s; j++) {
                C val = C(Traits::load(&map_t[south_ring * max_pix + j]));
                C angle = -C(m) * C(phi0_s) - C(m_eff_s) * C(j) * dphi_s;
                C c, s;
                Traits::sincos_d(angle, &s, &c);
                gs_re += val * c;
                gs_im += val * s;
            }

            // Store north and south separately (combination done in Phase 2)
            // Layout: [n_maps, lp1, n_north_rings]
            size_t idx = (size_t)t * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
            Gm_north_re[idx] = R(gn_re);
            Gm_north_im[idx] = R(gn_im);
            Gm_south_re[idx] = R(gs_re);
            Gm_south_im[idx] = R(gs_im);
        }
    }
}

// ============================================================================
// Hybrid FFT/DFT: Equatorial FFT phase correction and combination kernel
// Applies exp(-i*m*phi_0) phase correction and stores N/S separately
// ============================================================================

// Double precision FFT version of phase correction kernel
// FFT output and phi_0 are double, Gm output is R
template<typename R>
__global__ void equatorial_phase_combine_kernel(
    int nside, int l_max, int n_maps,
    int fft_size,      // 4*nside
    int fft_out_size,  // 2*nside+1
    int n_equatorial,  // 2*nside+1 rings
    const cufftDoubleComplex* __restrict__ fft_out,  // [n_maps, n_equatorial, fft_out_size]
    const double* __restrict__ phi_0,    // [n_equatorial] - double for f64 FFT
    R* __restrict__ Gm_north_re,     // [n_maps, lp1, n_north_rings]
    R* __restrict__ Gm_north_im,
    R* __restrict__ Gm_south_re,
    R* __restrict__ Gm_south_im,
    R* __restrict__ cos_theta_out,  // [n_north_rings]
    R* __restrict__ sin_theta_out
) {
    // Grid: blockIdx.x = m value, blockIdx.y = map index
    // Process equatorial north_rings (nside-1 to 2*nside-1, that's nside+1 rings)
    int m = blockIdx.x;
    int map_idx = blockIdx.y;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;
    int n_eq_north = nside + 1;  // Number of equatorial north_ring values

    if (m > l_max || map_idx >= n_maps) return;

    // Process each equatorial north_ring (one per thread block iteration)
    for (int k = threadIdx.x; k < n_eq_north; k += blockDim.x) {
        int north_ring = nside - 1 + k;  // north_ring index (nside-1 to 2*nside-1)
        bool is_equator = (k == nside);  // k=nside means north_ring=2*nside-1 (equator)

        // FFT buffer indices for north and south equatorial rings
        // Equatorial rings span ring indices nside-1 to 3*nside-1
        // FFT buffer index = ring_index - (nside-1)
        int fft_idx_north = k;                    // 0 to nside
        int fft_idx_south = 2 * nside - k;        // 2*nside down to nside

        // Get FFT output with periodicity and conjugate symmetry
        int m_mod = m % fft_size;
        cufftDoubleComplex fft_val_n, fft_val_s;

        size_t fft_base_n = (size_t)map_idx * n_equatorial * fft_out_size + fft_idx_north * fft_out_size;
        size_t fft_base_s = (size_t)map_idx * n_equatorial * fft_out_size + fft_idx_south * fft_out_size;

        if (m_mod < fft_out_size) {
            fft_val_n = fft_out[fft_base_n + m_mod];
            fft_val_s = fft_out[fft_base_s + m_mod];
        } else {
            // Use conjugate symmetry: FFT[m_mod] = conj(FFT[N-m_mod])
            int mirror_m = fft_size - m_mod;
            fft_val_n = fft_out[fft_base_n + mirror_m];
            fft_val_n.y = -fft_val_n.y;
            fft_val_s = fft_out[fft_base_s + mirror_m];
            fft_val_s.y = -fft_val_s.y;
        }

        // Phase correction: multiply by exp(-i * m * phi_0)
        double phi_0_n = (double)phi_0[fft_idx_north];
        double phi_0_s = (double)phi_0[fft_idx_south];

        double angle_n = -m * phi_0_n;
        double angle_s = -m * phi_0_s;
        double cos_n, sin_n, cos_s, sin_s;
        sincos(angle_n, &sin_n, &cos_n);
        sincos(angle_s, &sin_s, &cos_s);

        // Apply phase correction
        double gn_re = fft_val_n.x * cos_n - fft_val_n.y * sin_n;
        double gn_im = fft_val_n.x * sin_n + fft_val_n.y * cos_n;
        double gs_re = fft_val_s.x * cos_s - fft_val_s.y * sin_s;
        double gs_im = fft_val_s.x * sin_s + fft_val_s.y * cos_s;

        // Store N/S separately (combination done in Phase 2)
        // Equator: south = 0 (no south partner)
        R north_re = R(gn_re);
        R north_im = R(gn_im);
        R south_re = is_equator ? R(0) : R(gs_re);
        R south_im = is_equator ? R(0) : R(gs_im);

        // Output layout: [n_maps, lp1, n_north_rings]
        size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
        Gm_north_re[idx] = north_re;
        Gm_north_im[idx] = north_im;
        Gm_south_re[idx] = south_re;
        Gm_south_im[idx] = south_im;
    }

    // Store geometry (only once per north_ring, thread 0 of m=0, map=0)
    if (m == 0 && map_idx == 0) {
        for (int k = threadIdx.x; k < n_eq_north; k += blockDim.x) {
            int north_ring = nside - 1 + k;
            int ring_idx = north_ring;  // Ring index in full map

            // Compute geometry for equatorial ring
            double cos_th = 4.0/3.0 - 2.0 * (ring_idx + 1) / (3.0 * nside);
            double sin_th = sqrt(1.0 - cos_th * cos_th);

            cos_theta_out[north_ring] = R(cos_th);
            sin_theta_out[north_ring] = R(sin_th);
        }
    }
}

// Float32 version of phase correction kernel
// FFT output and phi_0 are float, but Gm output is R (can be double for f32_f64 mode)
template<typename R>
__global__ void equatorial_phase_combine_kernel_f32(
    int nside, int l_max, int n_maps,
    int fft_size,      // 4*nside
    int fft_out_size,  // 2*nside+1
    int n_equatorial,  // 2*nside+1 rings
    const cufftComplex* __restrict__ fft_out,  // [n_maps, n_equatorial, fft_out_size]
    const float* __restrict__ phi_0,    // [n_equatorial] - float for f32 FFT
    R* __restrict__ Gm_north_re,
    R* __restrict__ Gm_north_im,
    R* __restrict__ Gm_south_re,
    R* __restrict__ Gm_south_im,
    R* __restrict__ cos_theta_out,
    R* __restrict__ sin_theta_out
) {
    int m = blockIdx.x;
    int map_idx = blockIdx.y;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;
    int n_eq_north = nside + 1;

    if (m > l_max || map_idx >= n_maps) return;

    for (int k = threadIdx.x; k < n_eq_north; k += blockDim.x) {
        int north_ring = nside - 1 + k;
        bool is_equator = (k == nside);

        int fft_idx_north = k;
        int fft_idx_south = 2 * nside - k;

        int m_mod = m % fft_size;
        cufftComplex fft_val_n, fft_val_s;

        size_t fft_base_n = (size_t)map_idx * n_equatorial * fft_out_size + fft_idx_north * fft_out_size;
        size_t fft_base_s = (size_t)map_idx * n_equatorial * fft_out_size + fft_idx_south * fft_out_size;

        if (m_mod < fft_out_size) {
            fft_val_n = fft_out[fft_base_n + m_mod];
            fft_val_s = fft_out[fft_base_s + m_mod];
        } else {
            int mirror_m = fft_size - m_mod;
            fft_val_n = fft_out[fft_base_n + mirror_m];
            fft_val_n.y = -fft_val_n.y;
            fft_val_s = fft_out[fft_base_s + mirror_m];
            fft_val_s.y = -fft_val_s.y;
        }

        float phi_0_n = phi_0[fft_idx_north];
        float phi_0_s = phi_0[fft_idx_south];

        float angle_n = -m * phi_0_n;
        float angle_s = -m * phi_0_s;
        float cos_n, sin_n, cos_s, sin_s;
        sincosf(angle_n, &sin_n, &cos_n);
        sincosf(angle_s, &sin_s, &cos_s);

        float gn_re = fft_val_n.x * cos_n - fft_val_n.y * sin_n;
        float gn_im = fft_val_n.x * sin_n + fft_val_n.y * cos_n;
        float gs_re = fft_val_s.x * cos_s - fft_val_s.y * sin_s;
        float gs_im = fft_val_s.x * sin_s + fft_val_s.y * cos_s;

        // Store N/S separately (combination done in Phase 2)
        R north_re = R(gn_re);
        R north_im = R(gn_im);
        R south_re = is_equator ? R(0) : R(gs_re);
        R south_im = is_equator ? R(0) : R(gs_im);

        size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
        Gm_north_re[idx] = north_re;
        Gm_north_im[idx] = north_im;
        Gm_south_re[idx] = south_re;
        Gm_south_im[idx] = south_im;
    }

    if (m == 0 && map_idx == 0) {
        for (int k = threadIdx.x; k < n_eq_north; k += blockDim.x) {
            int north_ring = nside - 1 + k;
            int ring_idx = north_ring;
            float cos_th = 4.0f/3.0f - 2.0f * (ring_idx + 1) / (3.0f * nside);
            float sin_th = sqrtf(1.0f - cos_th * cos_th);
            cos_theta_out[north_ring] = R(cos_th);
            sin_theta_out[north_ring] = R(sin_th);
        }
    }
}

// ============================================================================
// Bluestein FFT: Compute Gm for ALL rings using chirp-z transform
// This allows a single batched FFT for all rings regardless of size
// Algorithm: DFT of size N via convolution in FFT of size M >= 2N-1
// ============================================================================

// Kernel to apply pre-chirp multiplication and zero-pad for Bluestein
// Input: map data for one ring
// Output: chirp-multiplied, zero-padded sequence ready for FFT
template<typename T>
__global__ void bluestein_pre_chirp_kernel(
    int nside, int n_maps, int n_rings,
    int M,  // Padded FFT size (power of 2)
    const T* __restrict__ map_in,      // [n_maps, n_rings, max_pix]
    cufftDoubleComplex* __restrict__ chirped_out,  // [n_maps, n_rings, M]
    double* __restrict__ cos_theta_out,
    double* __restrict__ sin_theta_out,
    int* __restrict__ ring_sizes_out   // [n_rings] - store ring sizes for later
) {
    using Traits = V6Traits<T>;
    using C = typename Traits::compute_t;

    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    // Compute ring geometry
    T cos_th, sin_th, phi0;
    int N;  // Ring size
    compute_ring_geom_v6<T>(ring_idx, nside, &cos_th, &sin_th, &phi0, &N);

    // Store geometry (only once per ring)
    if (map_idx == 0 && threadIdx.x == 0) {
        if (ring_idx < 2 * nside) {  // north ring indices
            cos_theta_out[ring_idx] = double(cos_th);
            sin_theta_out[ring_idx] = double(sin_th);
        }
        ring_sizes_out[ring_idx] = N;
    }

    // Get input/output pointers for this ring and map
    const T* ring_data = map_in + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;
    cufftDoubleComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;

    // Each thread handles multiple elements
    double pi_over_N = M_PI / double(N);

    for (int n = threadIdx.x; n < M; n += blockDim.x) {
        cufftDoubleComplex val;
        if (n < N) {
            // Load input and multiply by pre-chirp: exp(-πi * n² / N)
            double x = double(Traits::load(&ring_data[n]));
            double angle = -pi_over_N * double(n) * double(n);
            double c, s;
            sincos(angle, &s, &c);
            val.x = x * c;
            val.y = x * s;
        } else {
            // Zero padding
            val.x = 0.0;
            val.y = 0.0;
        }
        chirped[n] = val;
    }
}

// Float32 version of pre-chirp kernel
template<typename T>
__global__ void bluestein_pre_chirp_kernel_f32(
    int nside, int n_maps, int n_rings,
    int M,
    const T* __restrict__ map_in,
    cufftComplex* __restrict__ chirped_out,
    float* __restrict__ cos_theta_out,
    float* __restrict__ sin_theta_out,
    int* __restrict__ ring_sizes_out
) {
    using Traits = V6Traits<T>;
    using C = typename Traits::compute_t;

    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    int max_pix = 4 * nside;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    T cos_th, sin_th, phi0;
    int N;
    compute_ring_geom_v6<T>(ring_idx, nside, &cos_th, &sin_th, &phi0, &N);

    if (map_idx == 0 && threadIdx.x == 0) {
        if (ring_idx < 2 * nside) {
            cos_theta_out[ring_idx] = float(cos_th);
            sin_theta_out[ring_idx] = float(sin_th);
        }
        ring_sizes_out[ring_idx] = N;
    }

    const T* ring_data = map_in + (size_t)map_idx * n_rings * max_pix + ring_idx * max_pix;
    cufftComplex* chirped = chirped_out + (size_t)map_idx * n_rings * M + ring_idx * M;

    float pi_over_N = float(M_PI) / float(N);

    for (int n = threadIdx.x; n < M; n += blockDim.x) {
        cufftComplex val;
        if (n < N) {
            float x = float(Traits::load(&ring_data[n]));
            float angle = -pi_over_N * float(n) * float(n);
            float c, s;
            sincosf(angle, &s, &c);
            val.x = x * c;
            val.y = x * s;
        } else {
            val.x = 0.0f;
            val.y = 0.0f;
        }
        chirped[n] = val;
    }
}

// Kernel to compute conjugate chirp FFT for all unique ring sizes
// This is called once and cached
__global__ void bluestein_compute_conj_chirp_fft_kernel(
    int nside, int M,
    cufftDoubleComplex* __restrict__ conj_chirp_fft  // [nside, M] - one per unique ring size
) {
    // Each block handles one ring size
    int size_idx = blockIdx.x;  // 0 = size 4, 1 = size 8, ..., nside-1 = size 4*nside
    int N = 4 * (size_idx + 1);

    if (N > 4 * nside) return;

    cufftDoubleComplex* chirp = conj_chirp_fft + size_idx * M;
    double pi_over_N = M_PI / double(N);

    // Fill conjugate chirp with wrap-around for convolution
    // h'[j] = exp(+πi * j² / N) for j = 0..N-1 and M-N+1..M-1 (wrap-around)
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        cufftDoubleComplex val;
        int j_eff;  // Effective index for chirp computation

        if (j < N) {
            j_eff = j;
        } else if (j >= M - N + 1) {
            // Wrap-around: position M-k corresponds to -k
            j_eff = M - j;  // This gives the absolute value
        } else {
            // Zero in the middle
            val.x = 0.0;
            val.y = 0.0;
            chirp[j] = val;
            continue;
        }

        double angle = pi_over_N * double(j_eff) * double(j_eff);
        double c, s;
        sincos(angle, &s, &c);
        val.x = c;
        val.y = s;
        chirp[j] = val;
    }
}

// Float32 version of conjugate chirp computation
__global__ void bluestein_compute_conj_chirp_fft_kernel_f32(
    int nside, int M,
    cufftComplex* __restrict__ conj_chirp_fft
) {
    int size_idx = blockIdx.x;
    int N = 4 * (size_idx + 1);

    if (N > 4 * nside) return;

    cufftComplex* chirp = conj_chirp_fft + size_idx * M;
    float pi_over_N = float(M_PI) / float(N);

    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        cufftComplex val;
        int j_eff;

        if (j < N) {
            j_eff = j;
        } else if (j >= M - N + 1) {
            j_eff = M - j;
        } else {
            val.x = 0.0f;
            val.y = 0.0f;
            chirp[j] = val;
            continue;
        }

        float angle = pi_over_N * float(j_eff) * float(j_eff);
        float c, s;
        sincosf(angle, &s, &c);
        val.x = c;
        val.y = s;
        chirp[j] = val;
    }
}

// Kernel to apply pointwise multiplication with ring-specific conjugate chirp FFT
__global__ void bluestein_pointwise_mult_kernel(
    int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,  // [n_rings] - actual ring size N
    cufftDoubleComplex* __restrict__ fft_data,  // [n_maps, n_rings, M] - in-place
    const cufftDoubleComplex* __restrict__ conj_chirp_fft  // [nside, M]
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    int N = ring_sizes[ring_idx];
    int size_idx = (N / 4) - 1;  // Index into conj_chirp_fft

    cufftDoubleComplex* data = fft_data + (size_t)map_idx * n_rings * M + ring_idx * M;
    const cufftDoubleComplex* chirp = conj_chirp_fft + size_idx * M;

    for (int k = threadIdx.x; k < M; k += blockDim.x) {
        cufftDoubleComplex d = data[k];
        cufftDoubleComplex h = chirp[k];
        // Complex multiplication
        cufftDoubleComplex result;
        result.x = d.x * h.x - d.y * h.y;
        result.y = d.x * h.y + d.y * h.x;
        data[k] = result;
    }
}

// Float32 version of pointwise multiplication
__global__ void bluestein_pointwise_mult_kernel_f32(
    int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    cufftComplex* __restrict__ fft_data,
    const cufftComplex* __restrict__ conj_chirp_fft
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;

    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    int N = ring_sizes[ring_idx];
    int size_idx = (N / 4) - 1;

    cufftComplex* data = fft_data + (size_t)map_idx * n_rings * M + ring_idx * M;
    const cufftComplex* chirp = conj_chirp_fft + size_idx * M;

    for (int k = threadIdx.x; k < M; k += blockDim.x) {
        cufftComplex d = data[k];
        cufftComplex h = chirp[k];
        cufftComplex result;
        result.x = d.x * h.x - d.y * h.y;
        result.y = d.x * h.y + d.y * h.x;
        data[k] = result;
    }
}

// Kernel to extract Gm values from Bluestein result with post-chirp and phase correction
// Stores N/S separately for Phase 2 sequential loading
template<typename R>
__global__ void bluestein_extract_gm_kernel(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings,
    int M,
    const int* __restrict__ ring_sizes,
    const cufftDoubleComplex* __restrict__ ifft_data,  // [n_maps, n_rings, M]
    R* __restrict__ Gm_north_re,   // [n_maps, lp1, n_north_rings]
    R* __restrict__ Gm_north_im,
    R* __restrict__ Gm_south_re,
    R* __restrict__ Gm_south_im
) {
    // Grid: blockIdx.x = north_ring, blockIdx.y = map
    int north_ring = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;

    if (north_ring >= n_north_rings || map_idx >= n_maps) return;

    int south_ring = n_rings - 1 - north_ring;
    bool is_equator = (north_ring == 2 * nside - 1);

    int N_north = ring_sizes[north_ring];
    int N_south = is_equator ? 0 : ring_sizes[south_ring];

    // IFFT output for this ring (needs normalization by M)
    const cufftDoubleComplex* ifft_north = ifft_data +
        (size_t)map_idx * n_rings * M + north_ring * M;
    const cufftDoubleComplex* ifft_south = is_equator ? nullptr :
        (ifft_data + (size_t)map_idx * n_rings * M + south_ring * M);

    double pi_over_N_north = M_PI / double(N_north);
    double pi_over_N_south = is_equator ? 0.0 : M_PI / double(N_south);
    double inv_M = 1.0 / double(M);

    // Get phi_0 for phase correction
    double phi0_north, phi0_south;
    {
        // Compute phi_0 from ring geometry
        int ring_i = north_ring + 1;
        if (ring_i < nside) {
            phi0_north = M_PI / (2.0 * ring_i) * 0.5;
        } else if (ring_i <= 3 * nside) {
            int s = (ring_i % 2 == 0) ? 1 : 2;
            phi0_north = M_PI / (2.0 * nside) * (1.0 - s / 2.0);
        } else {
            int mirror_i = 4 * nside - ring_i;
            phi0_north = M_PI / (2.0 * mirror_i) * 0.5;
        }
    }
    if (!is_equator) {
        int ring_i = south_ring + 1;
        if (ring_i < nside) {
            phi0_south = M_PI / (2.0 * ring_i) * 0.5;
        } else if (ring_i <= 3 * nside) {
            int s = (ring_i % 2 == 0) ? 1 : 2;
            phi0_south = M_PI / (2.0 * nside) * (1.0 - s / 2.0);
        } else {
            int mirror_i = 4 * nside - ring_i;
            phi0_south = M_PI / (2.0 * mirror_i) * 0.5;
        }
    }

    // Each thread handles multiple m values
    for (int m = threadIdx.x; m <= l_max; m += blockDim.x) {
        // Bluestein gives us DFT output at index k (mod N periodically)
        // Need to apply post-chirp: multiply by exp(-πi * k² / N)
        // And phase correction: multiply by exp(-i * m * phi_0)

        // North ring
        int k_north = m % N_north;
        cufftDoubleComplex z_north = ifft_north[k_north];
        // Normalize by M (IFFT normalization)
        z_north.x *= inv_M;
        z_north.y *= inv_M;
        // Post-chirp
        double post_angle_north = -pi_over_N_north * double(k_north) * double(k_north);
        double post_c_north, post_s_north;
        sincos(post_angle_north, &post_s_north, &post_c_north);
        double gn_re_raw = z_north.x * post_c_north - z_north.y * post_s_north;
        double gn_im_raw = z_north.x * post_s_north + z_north.y * post_c_north;
        // Phase correction: exp(-i * m * phi_0)
        double phase_angle_north = -double(m) * phi0_north;
        double phase_c_north, phase_s_north;
        sincos(phase_angle_north, &phase_s_north, &phase_c_north);
        double gn_re = gn_re_raw * phase_c_north - gn_im_raw * phase_s_north;
        double gn_im = gn_re_raw * phase_s_north + gn_im_raw * phase_c_north;

        // South ring
        double gs_re = 0.0, gs_im = 0.0;
        if (!is_equator && ifft_south != nullptr) {
            int k_south = m % N_south;
            cufftDoubleComplex z_south = ifft_south[k_south];
            z_south.x *= inv_M;
            z_south.y *= inv_M;
            double post_angle_south = -pi_over_N_south * double(k_south) * double(k_south);
            double post_c_south, post_s_south;
            sincos(post_angle_south, &post_s_south, &post_c_south);
            double gs_re_raw = z_south.x * post_c_south - z_south.y * post_s_south;
            double gs_im_raw = z_south.x * post_s_south + z_south.y * post_c_south;
            double phase_angle_south = -double(m) * phi0_south;
            double phase_c_south, phase_s_south;
            sincos(phase_angle_south, &phase_s_south, &phase_c_south);
            gs_re = gs_re_raw * phase_c_south - gs_im_raw * phase_s_south;
            gs_im = gs_re_raw * phase_s_south + gs_im_raw * phase_c_south;
        }

        // Store N/S separately (combination done in Phase 2)
        size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
        Gm_north_re[idx] = R(gn_re);
        Gm_north_im[idx] = R(gn_im);
        Gm_south_re[idx] = R(gs_re);
        Gm_south_im[idx] = R(gs_im);
    }
}

// Float32 version of extract kernel
template<typename R>
__global__ void bluestein_extract_gm_kernel_f32(
    int nside, int l_max, int n_maps, int n_rings, int n_north_rings,
    int M,
    const int* __restrict__ ring_sizes,
    const cufftComplex* __restrict__ ifft_data,
    R* __restrict__ Gm_north_re,
    R* __restrict__ Gm_north_im,
    R* __restrict__ Gm_south_re,
    R* __restrict__ Gm_south_im
) {
    int north_ring = blockIdx.x;
    int map_idx = blockIdx.y;
    int lp1 = l_max + 1;

    if (north_ring >= n_north_rings || map_idx >= n_maps) return;

    int south_ring = n_rings - 1 - north_ring;
    bool is_equator = (north_ring == 2 * nside - 1);

    int N_north = ring_sizes[north_ring];
    int N_south = is_equator ? 0 : ring_sizes[south_ring];

    const cufftComplex* ifft_north = ifft_data +
        (size_t)map_idx * n_rings * M + north_ring * M;
    const cufftComplex* ifft_south = is_equator ? nullptr :
        (ifft_data + (size_t)map_idx * n_rings * M + south_ring * M);

    float pi_over_N_north = float(M_PI) / float(N_north);
    float pi_over_N_south = is_equator ? 0.0f : float(M_PI) / float(N_south);
    float inv_M = 1.0f / float(M);

    float phi0_north, phi0_south;
    {
        int ring_i = north_ring + 1;
        if (ring_i < nside) {
            phi0_north = float(M_PI) / float(2 * ring_i) * 0.5f;
        } else if (ring_i <= 3 * nside) {
            int s = (ring_i % 2 == 0) ? 1 : 2;
            phi0_north = float(M_PI) / float(2 * nside) * (1.0f - s / 2.0f);
        } else {
            int mirror_i = 4 * nside - ring_i;
            phi0_north = float(M_PI) / float(2 * mirror_i) * 0.5f;
        }
    }
    if (!is_equator) {
        int ring_i = south_ring + 1;
        if (ring_i < nside) {
            phi0_south = float(M_PI) / float(2 * ring_i) * 0.5f;
        } else if (ring_i <= 3 * nside) {
            int s = (ring_i % 2 == 0) ? 1 : 2;
            phi0_south = float(M_PI) / float(2 * nside) * (1.0f - s / 2.0f);
        } else {
            int mirror_i = 4 * nside - ring_i;
            phi0_south = float(M_PI) / float(2 * mirror_i) * 0.5f;
        }
    }

    for (int m = threadIdx.x; m <= l_max; m += blockDim.x) {
        int k_north = m % N_north;
        cufftComplex z_north = ifft_north[k_north];
        z_north.x *= inv_M;
        z_north.y *= inv_M;
        float post_angle_north = -pi_over_N_north * float(k_north) * float(k_north);
        float post_c_north, post_s_north;
        sincosf(post_angle_north, &post_s_north, &post_c_north);
        float gn_re_raw = z_north.x * post_c_north - z_north.y * post_s_north;
        float gn_im_raw = z_north.x * post_s_north + z_north.y * post_c_north;
        float phase_angle_north = -float(m) * phi0_north;
        float phase_c_north, phase_s_north;
        sincosf(phase_angle_north, &phase_s_north, &phase_c_north);
        float gn_re = gn_re_raw * phase_c_north - gn_im_raw * phase_s_north;
        float gn_im = gn_re_raw * phase_s_north + gn_im_raw * phase_c_north;

        float gs_re = 0.0f, gs_im = 0.0f;
        if (!is_equator && ifft_south != nullptr) {
            int k_south = m % N_south;
            cufftComplex z_south = ifft_south[k_south];
            z_south.x *= inv_M;
            z_south.y *= inv_M;
            float post_angle_south = -pi_over_N_south * float(k_south) * float(k_south);
            float post_c_south, post_s_south;
            sincosf(post_angle_south, &post_s_south, &post_c_south);
            float gs_re_raw = z_south.x * post_c_south - z_south.y * post_s_south;
            float gs_im_raw = z_south.x * post_s_south + z_south.y * post_c_south;
            float phase_angle_south = -float(m) * phi0_south;
            float phase_c_south, phase_s_south;
            sincosf(phase_angle_south, &phase_s_south, &phase_c_south);
            gs_re = gs_re_raw * phase_c_south - gs_im_raw * phase_s_south;
            gs_im = gs_re_raw * phase_s_south + gs_im_raw * phase_c_south;
        }

        // Store N/S separately (combination done in Phase 2)
        size_t idx = (size_t)map_idx * lp1 * n_north_rings + (size_t)m * n_north_rings + north_ring;
        Gm_north_re[idx] = R(gn_re);
        Gm_north_im[idx] = R(gn_im);
        Gm_south_re[idx] = R(gs_re);
        Gm_south_im[idx] = R(gs_im);
    }
}

// ============================================================================
// Phase 2: Reduce to alm using warp-per-m with ring batching and multi-map
// Uses sequential north/south loading to halve shared memory per map
// Ylm is computed once per l and reused for both north and south passes
// ============================================================================

// Default ring batch size for v6 - can be reduced for multi-map
#undef RING_BATCH_SIZE
#define RING_BATCH_SIZE 256

// Maximum maps that can be processed in parallel (shared memory limited)
// Shared mem layout: geometry (2 arrays, shared) + Gm (2 arrays per map for N or S)
// With sequential N/S loading, we only need 2 Gm arrays per map at a time
// f64: 2*256*8 + N*2*256*8 <= 48KB -> N <= 11
// f32: 2*256*4 + N*2*256*4 <= 48KB -> N <= 23
#define MAX_PARALLEL_MAPS_F64 11
#define MAX_PARALLEL_MAPS_F32 23

template<typename T, typename R, bool USE_PRECOMPUTED_COEFF, int RINGS_PER_LANE>
__global__ void reduce_to_alm_kernel_v6(
    int nside, int l_max, int n_maps, int n_north_rings,
    int ring_batch_size, int n_maps_parallel,  // configurable parameters
    int ring_pass, int total_ring_passes,      // multi-pass ring processing
    const R* __restrict__ Gm_north_re,
    const R* __restrict__ Gm_north_im,
    const R* __restrict__ Gm_south_re,
    const R* __restrict__ Gm_south_im,
    const R* __restrict__ cos_theta,
    const R* __restrict__ sin_theta,
    R pix_area,
    T* __restrict__ alm_out_re,
    T* __restrict__ alm_out_im
) {
    using Traits = V6Traits<R>;
    using C = typename Traits::compute_t;

    // One block per m, use only first warp (32 threads)
    int m = blockIdx.x;
    int lane = threadIdx.x;
    int lp1 = l_max + 1;

    if (m > l_max || lane >= 32) return;

    // Shared memory layout (sequential N/S loading):
    // - Geometry (shared across maps): cos_theta, sin_theta [2 * ring_batch_size]
    // - Optional coefficients: log_A, log_B [2 * lp1] (if USE_PRECOMPUTED_COEFF)
    // - Gm per map: only 2 arrays at a time (north OR south) [2 * ring_batch_size each]
    extern __shared__ char smem[];
    R* sh_cos_th = (R*)smem;
    R* sh_sin_th = sh_cos_th + ring_batch_size;
    // Precomputed recurrence coefficients (only if enabled)
    C* sh_log_A = USE_PRECOMPUTED_COEFF ? (C*)(sh_sin_th + ring_batch_size) : nullptr;
    C* sh_log_B = USE_PRECOMPUTED_COEFF ? (sh_log_A + lp1) : nullptr;
    // Gm arrays for parallel maps
    R* sh_Gm_base = USE_PRECOMPUTED_COEFF ? (R*)(sh_log_B + lp1) : (R*)(sh_sin_th + ring_batch_size);

    // Per-lane Ylm recurrence state in LOG-SPACE (local memory, L1 cached)
    // We store log(|Ylm|) and sign separately for numerical stability
    // Array size is compile-time constant from template parameter
    C log_Ylm_prev1[RINGS_PER_LANE];
    C log_Ylm_prev2[RINGS_PER_LANE];
    int8_t sign_prev1[RINGS_PER_LANE];
    int8_t sign_prev2[RINGS_PER_LANE];
    // Save initial Ymm for restoring between north/south passes
    C log_Ymm_saved[RINGS_PER_LANE];
    int8_t sign_Ymm_saved[RINGS_PER_LANE];
    // Precomputed log(|cos_th|) and sign for each ring (avoid recomputing in l-loop)
    C log_cos_th_cached[RINGS_PER_LANE];
    int8_t sign_cos_th_cached[RINGS_PER_LANE];

    // Per-lane accumulators for each parallel map
    C sum_re[MAX_PARALLEL_MAPS_F32];
    C sum_im[MAX_PARALLEL_MAPS_F32];

    // Calculate ring range for this pass (multi-pass support)
    // Each lane handles rings: lane, lane+32, lane+64, ...
    // ring_pass determines which subset of rings this kernel invocation processes
    int rings_per_lane_total = (n_north_rings + 31) / 32;
    int ring_lane_start = ring_pass * RINGS_PER_LANE;
    int ring_lane_end = min(ring_lane_start + RINGS_PER_LANE, rings_per_lane_total);

    // Precompute m-dependent constants ONCE (not per batch)
    using LogTraits = LogArithmeticTraits<C>;
    const C log_prefact = compute_log_prefact_ymm<C>(m);
    const C log_norm = -C(0.5) * LogTraits::log_d(C(4.0) * LogTraits::PI_VAL);
    const C log_recur_C_m1 = C(0.5) * LogTraits::log_d(C(2*m + 3));  // For l = m+1

    // Precompute recurrence coefficients for this m (only if enabled)
    if (USE_PRECOMPUTED_COEFF) {
        for (int l = m + 2 + lane; l <= l_max; l += 32) {
            sh_log_A[l] = log_A_lm<C>(l, m);
            sh_log_B[l] = log_B_lm<C>(l, m);
        }
        __syncwarp();
    }

    // Process maps in batches of n_maps_parallel
    for (int map_batch_start = 0; map_batch_start < n_maps; map_batch_start += n_maps_parallel) {
        int map_batch_end = min(map_batch_start + n_maps_parallel, n_maps);
        int n_maps_in_batch = map_batch_end - map_batch_start;

        // Process ring batches
        for (int batch_start = 0; batch_start < n_north_rings; batch_start += ring_batch_size) {
            int batch_end = min(batch_start + ring_batch_size, n_north_rings);
            int batch_size = batch_end - batch_start;

            // Cooperative load of geometry (shared across all maps, stays for both N/S passes)
            for (int r = lane; r < batch_size; r += 32) {
                int global_r = batch_start + r;
                sh_cos_th[r] = cos_theta[global_r];
                sh_sin_th[r] = sin_theta[global_r];
            }
            __syncwarp();

            // Determine which rings this lane handles in this batch (accounting for ring_pass)
            // Global k values this lane could handle in this batch
            int k_batch_start = (batch_start > lane) ? (batch_start - lane + 31) / 32 : 0;
            int k_batch_end = (batch_end > lane) ? (batch_end - 1 - lane) / 32 + 1 : 0;
            int n_my_rings_total = (n_north_rings + 31 - lane) / 32;
            k_batch_end = min(k_batch_end, n_my_rings_total);

            // Intersect batch range with ring_pass range, then shift to 0-based for local arrays
            // ring_lane_start and ring_lane_end define the global k range for this ring_pass
            int k_start = max(k_batch_start, ring_lane_start) - ring_lane_start;
            int k_end = min(k_batch_end, ring_lane_end) - ring_lane_start;
            k_start = max(k_start, 0);
            k_end = max(k_end, 0);

            // Compute and save initial Y[m,m] in LOG-SPACE for rings in this batch
            // Reference: jax_healpix/YLM_jax_log.py sYLM_ll0_log() lines 29-62
            for (int k = k_start; k < k_end; k++) {
                int global_r = lane + 32 * (k + ring_lane_start);
                int local_r = global_r - batch_start;
                C sin_th = C(sh_sin_th[local_r]);
                C cos_th = C(sh_cos_th[local_r]);

                // Precompute log_cos_th once per ring (reused across all l iterations)
                log_cos_th_cached[k] = safe_log_typed<C>(cos_th);
                sign_cos_th_cached[k] = (cos_th >= C(0)) ? int8_t(1) : int8_t(-1);

                // Y[m,m] = (-1)^m * sin(th)^m * prefact / sqrt(4*pi)
                // log|Y[m,m]| = m * log|sin(th)| + log_prefact + log_norm
                C log_sin_th = safe_log_typed<C>(sin_th);
                C log_Ymm = C(m) * log_sin_th + log_prefact + log_norm;
                int8_t sign_Ymm = ((m & 1) == 0) ? int8_t(1) : int8_t(-1);  // (-1)^m

                log_Ymm_saved[k] = log_Ymm;  // Save for south pass
                sign_Ymm_saved[k] = sign_Ymm;
                log_Ylm_prev1[k] = log_Ymm;
                sign_prev1[k] = sign_Ymm;
                log_Ylm_prev2[k] = LogTraits::LOG_MIN;  // Zero in log-space
                sign_prev2[k] = 0;
            }

            // ================================================================
            // NORTH PASS: Load Gm_north, accumulate
            // ================================================================
            for (int t = 0; t < n_maps_in_batch; t++) {
                int global_t = map_batch_start + t;
                size_t base_idx = (size_t)global_t * lp1 * n_north_rings + (size_t)m * n_north_rings;
                R* sh_Gm_t = sh_Gm_base + t * 2 * ring_batch_size;  // 2 arrays per map

                for (int r = lane; r < batch_size; r += 32) {
                    size_t idx = base_idx + batch_start + r;
                    sh_Gm_t[0 * ring_batch_size + r] = Gm_north_re[idx];
                    sh_Gm_t[1 * ring_batch_size + r] = Gm_north_im[idx];
                }
            }
            __syncwarp();

            // Process l = m to l_max for north pass (LOG-SPACE recurrence)
            for (int l = m; l <= l_max; l++) {
                for (int t = 0; t < n_maps_in_batch; t++) {
                    sum_re[t] = C(0);
                    sum_im[t] = C(0);
                }

                for (int k = k_start; k < k_end; k++) {
                    int global_r = lane + 32 * (k + ring_lane_start);
                    int local_r = global_r - batch_start;
                    // Use precomputed log_cos_th (computed once per ring, not per l)
                    C log_cos_th = log_cos_th_cached[k];
                    int8_t sign_cos_th = sign_cos_th_cached[k];

                    C log_Ylm;
                    int8_t sign_Ylm;

                    if (l == m) {
                        // Use stored Y[m,m]
                        log_Ylm = log_Ylm_prev1[k];
                        sign_Ylm = sign_prev1[k];
                    } else if (l == m + 1) {
                        // Y[m+1,m] = cos_th * sqrt(2m+3) * Y[m,m]
                        // log|Y[m+1,m]| = log|cos_th| + log_recur_C_m1 + log|Y[m,m]|
                        log_Ylm = log_cos_th + log_recur_C_m1 + log_Ylm_prev1[k];
                        sign_Ylm = sign_cos_th * sign_prev1[k];

                        // Update recurrence state
                        log_Ylm_prev2[k] = log_Ylm_prev1[k];
                        sign_prev2[k] = sign_prev1[k];
                        log_Ylm_prev1[k] = log_Ylm;
                        sign_prev1[k] = sign_Ylm;
                    } else {
                        // Y[l,m] = A*cos_th*Y[l-1,m] - B*Y[l-2,m]
                        C log_A = USE_PRECOMPUTED_COEFF ? sh_log_A[l] : log_A_lm<C>(l, m);
                        C log_B = USE_PRECOMPUTED_COEFF ? sh_log_B[l] : log_B_lm<C>(l, m);
                        C R1 = log_A + log_cos_th + log_Ylm_prev1[k];
                        int8_t S1 = sign_cos_th * sign_prev1[k];
                        C R2 = log_B + log_Ylm_prev2[k];
                        int8_t S2 = -sign_prev2[k];  // Negative due to subtraction

                        logsumexp_fast<C>(R1, R2, S1, S2, &log_Ylm, &sign_Ylm);

                        // Update recurrence state
                        log_Ylm_prev2[k] = log_Ylm_prev1[k];
                        sign_prev2[k] = sign_prev1[k];
                        log_Ylm_prev1[k] = log_Ylm;
                        sign_prev1[k] = sign_Ylm;
                    }

                    // Convert to LINEAR only here for Gm multiplication
                    C Ylm = sign_Ylm * LogTraits::exp_d(log_Ylm);

                    for (int t = 0; t < n_maps_in_batch; t++) {
                        R* sh_Gm_t = sh_Gm_base + t * 2 * ring_batch_size;
                        C gm_re = C(sh_Gm_t[0 * ring_batch_size + local_r]);
                        C gm_im = C(sh_Gm_t[1 * ring_batch_size + local_r]);
                        sum_re[t] += Ylm * gm_re;
                        sum_im[t] += Ylm * gm_im;
                    }
                }

                // Warp reduce and output north contribution
                for (int t = 0; t < n_maps_in_batch; t++) {
                    C sr = sum_re[t];
                    C si = sum_im[t];

                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        sr += __shfl_down_sync(0xffffffff, sr, offset);
                        si += __shfl_down_sync(0xffffffff, si, offset);
                    }

                    if (lane == 0) {
                        int global_t = map_batch_start + t;
                        T* alm_re_t = alm_out_re + (size_t)global_t * lp1 * lp1;
                        T* alm_im_t = alm_out_im + (size_t)global_t * lp1 * lp1;

                        // Initialize only on first batch of first ring pass
                        if (batch_start == 0 && ring_pass == 0) {
                            alm_re_t[l * lp1 + m] = T(sr * C(pix_area));
                            alm_im_t[l * lp1 + m] = T(si * C(pix_area));
                        } else {
                            alm_re_t[l * lp1 + m] = T(C(alm_re_t[l * lp1 + m]) + sr * C(pix_area));
                            alm_im_t[l * lp1 + m] = T(C(alm_im_t[l * lp1 + m]) + si * C(pix_area));
                        }
                    }
                }
            }

            // ================================================================
            // SOUTH PASS: Restore Ylm, load Gm_south, accumulate with sign
            // ================================================================
            // Restore Ylm log-space state
            for (int k = k_start; k < k_end; k++) {
                log_Ylm_prev1[k] = log_Ymm_saved[k];
                sign_prev1[k] = sign_Ymm_saved[k];
                log_Ylm_prev2[k] = LogTraits::LOG_MIN;  // Zero in log-space
                sign_prev2[k] = 0;
            }

            // Load Gm_south (reusing same shared memory)
            for (int t = 0; t < n_maps_in_batch; t++) {
                int global_t = map_batch_start + t;
                size_t base_idx = (size_t)global_t * lp1 * n_north_rings + (size_t)m * n_north_rings;
                R* sh_Gm_t = sh_Gm_base + t * 2 * ring_batch_size;

                for (int r = lane; r < batch_size; r += 32) {
                    size_t idx = base_idx + batch_start + r;
                    sh_Gm_t[0 * ring_batch_size + r] = Gm_south_re[idx];
                    sh_Gm_t[1 * ring_batch_size + r] = Gm_south_im[idx];
                }
            }
            __syncwarp();

            // Process l = m to l_max for south pass (LOG-SPACE recurrence)
            for (int l = m; l <= l_max; l++) {
                for (int t = 0; t < n_maps_in_batch; t++) {
                    sum_re[t] = C(0);
                    sum_im[t] = C(0);
                }

                // Sign for south: +1 if (l+m) even, -1 if odd
                C parity_sign = ((l + m) & 1) ? C(-1) : C(1);

                for (int k = k_start; k < k_end; k++) {
                    int global_r = lane + 32 * (k + ring_lane_start);
                    int local_r = global_r - batch_start;
                    // Use precomputed log_cos_th (computed once per ring, not per l)
                    C log_cos_th = log_cos_th_cached[k];
                    int8_t sign_cos_th = sign_cos_th_cached[k];

                    C log_Ylm;
                    int8_t sign_Ylm;

                    if (l == m) {
                        log_Ylm = log_Ylm_prev1[k];
                        sign_Ylm = sign_prev1[k];
                    } else if (l == m + 1) {
                        log_Ylm = log_cos_th + log_recur_C_m1 + log_Ylm_prev1[k];
                        sign_Ylm = sign_cos_th * sign_prev1[k];

                        log_Ylm_prev2[k] = log_Ylm_prev1[k];
                        sign_prev2[k] = sign_prev1[k];
                        log_Ylm_prev1[k] = log_Ylm;
                        sign_prev1[k] = sign_Ylm;
                    } else {
                        C log_A = USE_PRECOMPUTED_COEFF ? sh_log_A[l] : log_A_lm<C>(l, m);
                        C log_B = USE_PRECOMPUTED_COEFF ? sh_log_B[l] : log_B_lm<C>(l, m);
                        C R1 = log_A + log_cos_th + log_Ylm_prev1[k];
                        int8_t S1 = sign_cos_th * sign_prev1[k];
                        C R2 = log_B + log_Ylm_prev2[k];
                        int8_t S2 = -sign_prev2[k];

                        logsumexp_fast<C>(R1, R2, S1, S2, &log_Ylm, &sign_Ylm);

                        log_Ylm_prev2[k] = log_Ylm_prev1[k];
                        sign_prev2[k] = sign_prev1[k];
                        log_Ylm_prev1[k] = log_Ylm;
                        sign_prev1[k] = sign_Ylm;
                    }

                    // Convert to LINEAR with parity sign for south hemisphere
                    C Ylm = parity_sign * sign_Ylm * LogTraits::exp_d(log_Ylm);

                    for (int t = 0; t < n_maps_in_batch; t++) {
                        R* sh_Gm_t = sh_Gm_base + t * 2 * ring_batch_size;
                        C gm_re = C(sh_Gm_t[0 * ring_batch_size + local_r]);
                        C gm_im = C(sh_Gm_t[1 * ring_batch_size + local_r]);
                        sum_re[t] += Ylm * gm_re;
                        sum_im[t] += Ylm * gm_im;
                    }
                }

                // Warp reduce and ADD to alm (south contribution)
                for (int t = 0; t < n_maps_in_batch; t++) {
                    C sr = sum_re[t];
                    C si = sum_im[t];

                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        sr += __shfl_down_sync(0xffffffff, sr, offset);
                        si += __shfl_down_sync(0xffffffff, si, offset);
                    }

                    if (lane == 0) {
                        int global_t = map_batch_start + t;
                        T* alm_re_t = alm_out_re + (size_t)global_t * lp1 * lp1;
                        T* alm_im_t = alm_out_im + (size_t)global_t * lp1 * lp1;
                        // Always add (north pass already wrote initial value)
                        alm_re_t[l * lp1 + m] = T(C(alm_re_t[l * lp1 + m]) + sr * C(pix_area));
                        alm_im_t[l * lp1 + m] = T(C(alm_im_t[l * lp1 + m]) + si * C(pix_area));
                    }
                }
            }
        }
    }
}

// ============================================================================
// Explicit template instantiations for reduce_to_alm_kernel_v6
// ============================================================================

// Float32 storage, Float32 recurrence - all ring sizes
template __global__ void reduce_to_alm_kernel_v6<float, float, true, 16>(int, int, int, int, int, int, int, int, const float*, const float*, const float*, const float*, const float*, const float*, float, float*, float*);
template __global__ void reduce_to_alm_kernel_v6<float, float, true, 32>(int, int, int, int, int, int, int, int, const float*, const float*, const float*, const float*, const float*, const float*, float, float*, float*);
template __global__ void reduce_to_alm_kernel_v6<float, float, true, 64>(int, int, int, int, int, int, int, int, const float*, const float*, const float*, const float*, const float*, const float*, float, float*, float*);
template __global__ void reduce_to_alm_kernel_v6<float, float, true, 128>(int, int, int, int, int, int, int, int, const float*, const float*, const float*, const float*, const float*, const float*, float, float*, float*);
template __global__ void reduce_to_alm_kernel_v6<float, float, false, 16>(int, int, int, int, int, int, int, int, const float*, const float*, const float*, const float*, const float*, const float*, float, float*, float*);
template __global__ void reduce_to_alm_kernel_v6<float, float, false, 32>(int, int, int, int, int, int, int, int, const float*, const float*, const float*, const float*, const float*, const float*, float, float*, float*);
template __global__ void reduce_to_alm_kernel_v6<float, float, false, 64>(int, int, int, int, int, int, int, int, const float*, const float*, const float*, const float*, const float*, const float*, float, float*, float*);
template __global__ void reduce_to_alm_kernel_v6<float, float, false, 128>(int, int, int, int, int, int, int, int, const float*, const float*, const float*, const float*, const float*, const float*, float, float*, float*);

// Float64 storage, Float64 recurrence - all ring sizes
template __global__ void reduce_to_alm_kernel_v6<double, double, true, 16>(int, int, int, int, int, int, int, int, const double*, const double*, const double*, const double*, const double*, const double*, double, double*, double*);
template __global__ void reduce_to_alm_kernel_v6<double, double, true, 32>(int, int, int, int, int, int, int, int, const double*, const double*, const double*, const double*, const double*, const double*, double, double*, double*);
template __global__ void reduce_to_alm_kernel_v6<double, double, true, 64>(int, int, int, int, int, int, int, int, const double*, const double*, const double*, const double*, const double*, const double*, double, double*, double*);
template __global__ void reduce_to_alm_kernel_v6<double, double, true, 128>(int, int, int, int, int, int, int, int, const double*, const double*, const double*, const double*, const double*, const double*, double, double*, double*);
template __global__ void reduce_to_alm_kernel_v6<double, double, false, 16>(int, int, int, int, int, int, int, int, const double*, const double*, const double*, const double*, const double*, const double*, double, double*, double*);
template __global__ void reduce_to_alm_kernel_v6<double, double, false, 32>(int, int, int, int, int, int, int, int, const double*, const double*, const double*, const double*, const double*, const double*, double, double*, double*);
template __global__ void reduce_to_alm_kernel_v6<double, double, false, 64>(int, int, int, int, int, int, int, int, const double*, const double*, const double*, const double*, const double*, const double*, double, double*, double*);
template __global__ void reduce_to_alm_kernel_v6<double, double, false, 128>(int, int, int, int, int, int, int, int, const double*, const double*, const double*, const double*, const double*, const double*, double, double*, double*);

// ============================================================================
// Host wrapper implementation
// ============================================================================

// Helper function to compute optimal ring batch size and parallel maps
template<typename R>
void compute_v6_params(int n_maps, int* ring_batch_size, int* n_maps_parallel) {
    // Available shared memory (48KB)
    const size_t MAX_SMEM = 48 * 1024;
    const size_t elem_size = sizeof(R);

    // Default batch size
    int batch = RING_BATCH_SIZE;  // 256

    // Calculate max parallel maps for default batch size
    // With sequential N/S loading, we only need 2 Gm arrays per map at a time
    // Shared mem: geometry (2 arrays) + Gm (2 arrays per map)
    // smem = 2 * batch * elem + n_par * 2 * batch * elem
    // n_par = (MAX_SMEM / elem - 2 * batch) / (2 * batch)
    int max_parallel = (MAX_SMEM / elem_size - 2 * batch) / (2 * batch);

    // Cap at compile-time maximum
    if (std::is_same<R, float>::value) {
        max_parallel = min(max_parallel, MAX_PARALLEL_MAPS_F32);
    } else {
        max_parallel = min(max_parallel, MAX_PARALLEL_MAPS_F64);
    }
    max_parallel = max(1, max_parallel);

    // If n_maps <= max_parallel, use default batch and process all maps together
    if (n_maps <= max_parallel) {
        *ring_batch_size = batch;
        *n_maps_parallel = n_maps;
        return;
    }

    // Otherwise, try to balance batch size vs parallel maps
    // Option 1: Use default batch, process maps in groups of max_parallel
    // Option 2: Reduce batch size to fit more maps in parallel

    // For now, use default batch and let the kernel handle map batching
    *ring_batch_size = batch;
    *n_maps_parallel = max_parallel;
}

// Helper to compute equatorial phi_0 values on GPU
__global__ void compute_equatorial_phi0_kernel(int nside, int n_equatorial, double* phi_0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_equatorial) return;

    // Equatorial ring indices: nside-1 to 3*nside-1
    int ring_idx = nside - 1 + i;  // 0-indexed ring in full map
    int ring_i = ring_idx + 1;      // 1-indexed for formula

    // phi_0 for equatorial rings: pi/(2*nside) * (1 - s/2) where s = 1 if even ring, 2 if odd
    int s = ((ring_i - nside) % 2 == 0) ? 1 : 2;
    phi_0[i] = M_PI / (2.0 * nside) * (1.0 - s / 2.0);
}

__global__ void compute_equatorial_phi0_kernel_f32(int nside, int n_equatorial, float* phi_0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_equatorial) return;

    int ring_idx = nside - 1 + i;
    int ring_i = ring_idx + 1;
    int s = ((ring_i - nside) % 2 == 0) ? 1 : 2;
    phi_0[i] = (float)(M_PI / (2.0 * nside) * (1.0 - s / 2.0));
}

template<typename T, typename R>
void map2alm_cuda_v6_impl(
    int nside, int l_max, int n_maps,
    const T* map_in,
    T* alm_out_re, T* alm_out_im
) {
    int n_rings = 4 * nside - 1;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;

    // Calculate optimal kernel configuration (template size, multi-pass, etc.)
    KernelConfig config = calculate_kernel_config<R>(nside, n_maps);

    // Get cached workspace (avoids malloc/free overhead on repeated calls)
    Map2almWorkspace<T, R>& ws = get_map2alm_workspace<T, R>();

    // Compute required sizes
    size_t gm_size = (size_t)n_maps * n_north_rings * lp1 * sizeof(R);
    size_t geom_size = n_north_rings * sizeof(R);
    size_t ring_sizes_size = n_rings * sizeof(int);

    // Ensure workspace has sufficient size (only reallocates if needed)
    ws.ensure_size(gm_size, geom_size, ring_sizes_size);

    // Use cached pointers
    R* Gm_north_re = ws.Gm_north_re;
    R* Gm_north_im = ws.Gm_north_im;
    R* Gm_south_re = ws.Gm_south_re;
    R* Gm_south_im = ws.Gm_south_im;
    R* cos_theta = ws.cos_theta;
    R* sin_theta = ws.sin_theta;

    // ================================================================
    // Phase 1: Compute Gm - method selected at runtime
    // ================================================================

    // Timing events (enabled via environment variable)
    static bool timing_enabled = (getenv("SPHT_TIMING") != nullptr);
    cudaEvent_t start_p1, end_p1, start_p2, end_p2;
    if (timing_enabled) {
        cudaEventCreate(&start_p1);
        cudaEventCreate(&end_p1);
        cudaEventCreate(&start_p2);
        cudaEventCreate(&end_p2);
        cudaEventRecord(start_p1);
    }

    bool use_double_precision = std::is_same<T, double>::value;

    if (g_phase1_method == Phase1Method::BLUESTEIN) {
        // ============================================================
        // BLUESTEIN FFT: All rings processed via chirp-z transform
        // ============================================================

        // Compute M = next power of 2 >= 2*max_ring_size - 1
        int max_ring_size = 4 * nside;
        int M = next_power_of_2(2 * max_ring_size - 1);

        // Use cached ring_sizes buffer
        int* ring_sizes = ws.ring_sizes;

        if (use_double_precision) {
            // Double precision Bluestein
            size_t chirp_data_size = (size_t)n_maps * n_rings * M * sizeof(cufftDoubleComplex);
            size_t conj_chirp_size = (size_t)nside * M * sizeof(cufftDoubleComplex);

            // Use cached Bluestein buffers
            ws.ensure_bluestein(chirp_data_size, conj_chirp_size);
            cufftDoubleComplex* chirped_data = (cufftDoubleComplex*)ws.chirped_data;
            cufftDoubleComplex* conj_chirp_fft = (cufftDoubleComplex*)ws.conj_chirp_fft;
            double* bs_cos_theta = (double*)ws.bs_cos_theta;
            double* bs_sin_theta = (double*)ws.bs_sin_theta;

            // Step 1: Pre-chirp multiplication and padding
            dim3 grid_pre(n_rings, n_maps);
            bluestein_pre_chirp_kernel<T><<<grid_pre, 256>>>(
                nside, n_maps, n_rings, M, map_in,
                chirped_data, bs_cos_theta, bs_sin_theta, ring_sizes
            );
            CUDA_CHECK(cudaGetLastError());

            // Step 2: Compute conjugate chirp for all unique ring sizes and FFT them
            bluestein_compute_conj_chirp_fft_kernel<<<nside, 256>>>(nside, M, conj_chirp_fft);
            CUDA_CHECK(cudaGetLastError());

            // FFT the conjugate chirps (use cached plan)
            cufftHandle chirp_fft_plan = get_cached_plan(M, nside, CUFFT_Z2Z);
            cufftExecZ2Z(chirp_fft_plan, conj_chirp_fft, conj_chirp_fft, CUFFT_FORWARD);

            // Step 3: FFT all chirped input data (batched, use cached plan)
            cufftHandle data_fft_plan = get_cached_plan(M, n_maps * n_rings, CUFFT_Z2Z);
            cufftExecZ2Z(data_fft_plan, chirped_data, chirped_data, CUFFT_FORWARD);
            CUDA_CHECK(cudaGetLastError());

            // Step 4: Pointwise multiply with ring-specific conjugate chirp FFT
            bluestein_pointwise_mult_kernel<<<grid_pre, 256>>>(
                n_maps, n_rings, M, ring_sizes, chirped_data, conj_chirp_fft
            );
            CUDA_CHECK(cudaGetLastError());

            // Step 5: IFFT all data (reuse cached plan)
            cufftExecZ2Z(data_fft_plan, chirped_data, chirped_data, CUFFT_INVERSE);
            CUDA_CHECK(cudaGetLastError());

            // Step 6: Extract Gm with post-chirp and phase correction
            dim3 grid_extract(n_north_rings, n_maps);
            bluestein_extract_gm_kernel<R><<<grid_extract, 256>>>(
                nside, l_max, n_maps, n_rings, n_north_rings, M,
                ring_sizes, chirped_data,
                Gm_north_re, Gm_north_im, Gm_south_re, Gm_south_im
            );
            CUDA_CHECK(cudaGetLastError());

            // Copy geometry to output buffers (cast from double to R)
            if (std::is_same<R, double>::value) {
                CUDA_CHECK(cudaMemcpy(cos_theta, bs_cos_theta, geom_size, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(sin_theta, bs_sin_theta, geom_size, cudaMemcpyDeviceToDevice));
            } else {
                // Need to convert double -> float
                // For simplicity, recompute in Phase 2 kernel or use a conversion kernel
                // Actually the reduce kernel computes from Gm data, so geometry is needed
                // Let's use DFT kernel just for geometry if needed, or add a simple conversion
                int block_geom = 256;
                int grid_geom = (n_north_rings + block_geom - 1) / block_geom;
                // Temporary: just use the DFT kernel to compute geometry
                compute_gm_kernel_v6<T, R><<<n_north_rings, min(256, lp1)>>>(
                    nside, l_max, 0, n_rings, map_in,  // n_maps=0 to skip Gm computation
                    Gm_north_re, Gm_north_im, Gm_south_re, Gm_south_im,
                    cos_theta, sin_theta
                );
            }

            // No cudaFree - buffers are cached in workspace
        } else {
            // Float32 precision Bluestein
            size_t chirp_data_size = (size_t)n_maps * n_rings * M * sizeof(cufftComplex);
            size_t conj_chirp_size = (size_t)nside * M * sizeof(cufftComplex);

            // Use cached Bluestein buffers
            ws.ensure_bluestein(chirp_data_size, conj_chirp_size);
            cufftComplex* chirped_data = (cufftComplex*)ws.chirped_data;
            cufftComplex* conj_chirp_fft = (cufftComplex*)ws.conj_chirp_fft;
            float* bs_cos_theta = (float*)ws.bs_cos_theta;
            float* bs_sin_theta = (float*)ws.bs_sin_theta;

            dim3 grid_pre(n_rings, n_maps);
            bluestein_pre_chirp_kernel_f32<T><<<grid_pre, 256>>>(
                nside, n_maps, n_rings, M, map_in,
                chirped_data, bs_cos_theta, bs_sin_theta, ring_sizes
            );
            CUDA_CHECK(cudaGetLastError());

            bluestein_compute_conj_chirp_fft_kernel_f32<<<nside, 256>>>(nside, M, conj_chirp_fft);
            CUDA_CHECK(cudaGetLastError());

            // Use cached plans for float32
            cufftHandle chirp_fft_plan = get_cached_plan(M, nside, CUFFT_C2C);
            cufftExecC2C(chirp_fft_plan, conj_chirp_fft, conj_chirp_fft, CUFFT_FORWARD);

            cufftHandle data_fft_plan = get_cached_plan(M, n_maps * n_rings, CUFFT_C2C);
            cufftExecC2C(data_fft_plan, chirped_data, chirped_data, CUFFT_FORWARD);
            CUDA_CHECK(cudaGetLastError());

            bluestein_pointwise_mult_kernel_f32<<<grid_pre, 256>>>(
                n_maps, n_rings, M, ring_sizes, chirped_data, conj_chirp_fft
            );
            CUDA_CHECK(cudaGetLastError());

            cufftExecC2C(data_fft_plan, chirped_data, chirped_data, CUFFT_INVERSE);
            CUDA_CHECK(cudaGetLastError());

            dim3 grid_extract(n_north_rings, n_maps);
            bluestein_extract_gm_kernel_f32<R><<<grid_extract, 256>>>(
                nside, l_max, n_maps, n_rings, n_north_rings, M,
                ring_sizes, chirped_data,
                Gm_north_re, Gm_north_im, Gm_south_re, Gm_south_im
            );
            CUDA_CHECK(cudaGetLastError());

            // Copy geometry
            if (std::is_same<R, float>::value) {
                CUDA_CHECK(cudaMemcpy(cos_theta, bs_cos_theta, geom_size, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(sin_theta, bs_sin_theta, geom_size, cudaMemcpyDeviceToDevice));
            } else {
                // R is double but T is float - need conversion (unusual case)
                compute_gm_kernel_v6<T, R><<<n_north_rings, min(256, lp1)>>>(
                    nside, l_max, 0, n_rings, map_in,
                    Gm_north_re, Gm_north_im, Gm_south_re, Gm_south_im,
                    cos_theta, sin_theta
                );
            }

            // No cudaFree - buffers are cached in workspace
        }

        // No cudaFree for ring_sizes - cached in workspace

    } else if (g_phase1_method == Phase1Method::FFT_EQUATORIAL) {
        // ============================================================
        // HYBRID FFT/DFT: FFT for equatorial, DFT for polar rings
        // ============================================================

        // Phase 1a: Polar rings via DFT (nside-1 ring pairs)
        int n_polar_rings = nside - 1;
        if (n_polar_rings > 0) {
            int block_size_polar = min(256, lp1);
            compute_gm_polar_dft_kernel<T, R><<<n_polar_rings, block_size_polar>>>(
                nside, l_max, n_maps, n_rings, map_in,
                Gm_north_re, Gm_north_im, Gm_south_re, Gm_south_im,
                cos_theta, sin_theta
            );
            CUDA_CHECK(cudaGetLastError());
        }

        // Phase 1b: Equatorial rings via FFT (2*nside+1 rings)
        int n_equatorial = 2 * nside + 1;
        int fft_size = 4 * nside;
        int fft_out_size = fft_size / 2 + 1;
        int eq_start_ring = nside - 1;

        void* eq_phi_0;
        if (use_double_precision) {
            CUDA_CHECK(cudaMalloc(&eq_phi_0, n_equatorial * sizeof(double)));
            compute_equatorial_phi0_kernel<<<(n_equatorial + 255) / 256, 256>>>(
                nside, n_equatorial, (double*)eq_phi_0);
        } else {
            CUDA_CHECK(cudaMalloc(&eq_phi_0, n_equatorial * sizeof(float)));
            compute_equatorial_phi0_kernel_f32<<<(n_equatorial + 255) / 256, 256>>>(
                nside, n_equatorial, (float*)eq_phi_0);
        }
        CUDA_CHECK(cudaGetLastError());

        cufftHandle fft_plan;
        cufftResult fft_result;

        if (use_double_precision) {
            fft_result = cufftPlanMany(&fft_plan, 1, &fft_size,
                NULL, 1, fft_size, NULL, 1, fft_out_size,
                CUFFT_D2Z, n_equatorial);
        } else {
            fft_result = cufftPlanMany(&fft_plan, 1, &fft_size,
                NULL, 1, fft_size, NULL, 1, fft_out_size,
                CUFFT_R2C, n_equatorial);
        }

        if (fft_result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT plan creation failed: %d, falling back to DFT\n", fft_result);
            cudaFree(eq_phi_0);
            goto use_dft_fallback;
        }

        {
            size_t fft_out_elem_size = use_double_precision ?
                                       sizeof(cufftDoubleComplex) : sizeof(cufftComplex);
            void* fft_out;
            CUDA_CHECK(cudaMalloc(&fft_out, (size_t)n_maps * n_equatorial * fft_out_size * fft_out_elem_size));

            for (int t = 0; t < n_maps; t++) {
                const T* eq_map_ptr = map_in + (size_t)t * n_rings * fft_size + eq_start_ring * fft_size;

                if (use_double_precision) {
                    cufftDoubleComplex* fft_out_ptr = (cufftDoubleComplex*)fft_out +
                                                      (size_t)t * n_equatorial * fft_out_size;
                    fft_result = cufftExecD2Z(fft_plan, (cufftDoubleReal*)eq_map_ptr, fft_out_ptr);
                } else {
                    cufftComplex* fft_out_ptr = (cufftComplex*)fft_out +
                                                (size_t)t * n_equatorial * fft_out_size;
                    fft_result = cufftExecR2C(fft_plan, (cufftReal*)eq_map_ptr, fft_out_ptr);
                }

                if (fft_result != CUFFT_SUCCESS) {
                    fprintf(stderr, "cuFFT execution failed: %d\n", fft_result);
                    cudaFree(fft_out);
                    cufftDestroy(fft_plan);
                    cudaFree(eq_phi_0);
                    goto use_dft_fallback;
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            dim3 block_phase(128);
            dim3 grid_phase(lp1, n_maps);

            if (use_double_precision) {
                equatorial_phase_combine_kernel<R><<<grid_phase, block_phase>>>(
                    nside, l_max, n_maps, fft_size, fft_out_size, n_equatorial,
                    (cufftDoubleComplex*)fft_out, (double*)eq_phi_0,
                    Gm_north_re, Gm_north_im, Gm_south_re, Gm_south_im,
                    cos_theta, sin_theta
                );
            } else {
                equatorial_phase_combine_kernel_f32<R><<<grid_phase, block_phase>>>(
                    nside, l_max, n_maps, fft_size, fft_out_size, n_equatorial,
                    (cufftComplex*)fft_out, (float*)eq_phi_0,
                    Gm_north_re, Gm_north_im, Gm_south_re, Gm_south_im,
                    cos_theta, sin_theta
                );
            }
            CUDA_CHECK(cudaGetLastError());

            cudaFree(fft_out);
            cufftDestroy(fft_plan);
            cudaFree(eq_phi_0);
        }
        goto skip_dft;

    use_dft_fallback:
        // Fall through to DFT
        ;
    }

    if (g_phase1_method == Phase1Method::DFT) {
        // ============================================================
        // DFT: Direct DFT for all rings (default, simple)
        // ============================================================
        int block_size_p1 = min(256, lp1);
        compute_gm_kernel_v6<T, R><<<n_north_rings, block_size_p1>>>(
            nside, l_max, n_maps, n_rings, map_in,
            Gm_north_re, Gm_north_im, Gm_south_re, Gm_south_im,
            cos_theta, sin_theta
        );
        CUDA_CHECK(cudaGetLastError());
    }

skip_dft:

    // End Phase 1 timing, start Phase 2 timing
    if (timing_enabled) {
        cudaEventRecord(end_p1);
        cudaEventRecord(start_p2);
    }

    // Compute optimal ring batch size and parallel maps
    int ring_batch_size, n_maps_parallel;
    compute_v6_params<R>(n_maps, &ring_batch_size, &n_maps_parallel);

    // Phase 2: Reduce to alm with ring batching and multi-map
    // Check if coefficients fit in shared memory (48KB limit)
    const size_t MAX_SMEM = 48 * 1024;
    size_t coeff_smem = 2 * lp1 * sizeof(R);
    size_t base_smem = (2 + 4 * n_maps_parallel) * ring_batch_size * sizeof(R);
    bool use_precomputed = (base_smem + coeff_smem) <= MAX_SMEM;

    size_t smem_size = use_precomputed ? (base_smem + coeff_smem) : base_smem;

    R pix_area = R(4.0 * M_PI / (12.0 * nside * nside));

    // Dispatch macro for template size selection
    #define DISPATCH_KERNEL(RINGS_PER_LANE_VAL, USE_PRECOMP) \
        reduce_to_alm_kernel_v6<T, R, USE_PRECOMP, RINGS_PER_LANE_VAL><<<lp1, 32, smem_size>>>( \
            nside, l_max, n_maps, n_north_rings, \
            ring_batch_size, n_maps_parallel, \
            ring_pass, config.n_ring_passes, \
            Gm_north_re, Gm_north_im, Gm_south_re, Gm_south_im, \
            cos_theta, sin_theta, pix_area, \
            alm_out_re, alm_out_im \
        )

    // Multi-pass loop over ring passes
    for (int ring_pass = 0; ring_pass < config.n_ring_passes; ring_pass++) {
        // Dispatch based on template size (config.rings_per_lane)
        if (use_precomputed) {
            switch (config.rings_per_lane) {
                case 16:  DISPATCH_KERNEL(16, true);  break;
                case 32:  DISPATCH_KERNEL(32, true);  break;
                case 64:  DISPATCH_KERNEL(64, true);  break;
                case 128: DISPATCH_KERNEL(128, true); break;
                default:
                    fprintf(stderr, "Error: Invalid rings_per_lane=%d\n", config.rings_per_lane);
                    return;
            }
        } else {
            switch (config.rings_per_lane) {
                case 16:  DISPATCH_KERNEL(16, false);  break;
                case 32:  DISPATCH_KERNEL(32, false);  break;
                case 64:  DISPATCH_KERNEL(64, false);  break;
                case 128: DISPATCH_KERNEL(128, false); break;
                default:
                    fprintf(stderr, "Error: Invalid rings_per_lane=%d\n", config.rings_per_lane);
                    return;
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }
    #undef DISPATCH_KERNEL

    CUDA_CHECK(cudaDeviceSynchronize());

    // End Phase 2 timing and report
    if (timing_enabled) {
        cudaEventRecord(end_p2);
        cudaEventSynchronize(end_p2);

        float p1_ms, p2_ms;
        cudaEventElapsedTime(&p1_ms, start_p1, end_p1);
        cudaEventElapsedTime(&p2_ms, start_p2, end_p2);

        fprintf(stderr, "[SPHT_TIMING] nside=%d l_max=%d n_maps=%d Phase1=%.2fms Phase2=%.2fms Total=%.2fms\n",
                nside, l_max, n_maps, p1_ms, p2_ms, p1_ms + p2_ms);

        cudaEventDestroy(start_p1);
        cudaEventDestroy(end_p1);
        cudaEventDestroy(start_p2);
        cudaEventDestroy(end_p2);
    }

    // No cleanup needed - all buffers are cached in workspace for reuse
}

// ============================================================================
// C API entry points
// ============================================================================

extern "C" {

// Float64 storage, Float64 recurrence
void map2alm_cuda_v6_f64_f64(int nside, int l_max, int n_maps,
                              const double* map_in,
                              double* alm_out_re, double* alm_out_im) {
    map2alm_cuda_v6_impl<double, double>(nside, l_max, n_maps, map_in, alm_out_re, alm_out_im);
}

// Float64 storage, Float32 recurrence
void map2alm_cuda_v6_f64_f32(int nside, int l_max, int n_maps,
                              const double* map_in,
                              double* alm_out_re, double* alm_out_im) {
    map2alm_cuda_v6_impl<double, float>(nside, l_max, n_maps, map_in, alm_out_re, alm_out_im);
}

// Float32 storage, Float64 recurrence
void map2alm_cuda_v6_f32_f64(int nside, int l_max, int n_maps,
                              const float* map_in,
                              float* alm_out_re, float* alm_out_im) {
    map2alm_cuda_v6_impl<float, double>(nside, l_max, n_maps, map_in, alm_out_re, alm_out_im);
}

// Float32 storage, Float32 recurrence
void map2alm_cuda_v6_f32_f32(int nside, int l_max, int n_maps,
                              const float* map_in,
                              float* alm_out_re, float* alm_out_im) {
    map2alm_cuda_v6_impl<float, float>(nside, l_max, n_maps, map_in, alm_out_re, alm_out_im);
}

// BFloat16 storage, Float32 recurrence
void map2alm_cuda_v6_bf16_f32(int nside, int l_max, int n_maps,
                               const __nv_bfloat16* map_in,
                               __nv_bfloat16* alm_out_re, __nv_bfloat16* alm_out_im) {
    map2alm_cuda_v6_impl<__nv_bfloat16, float>(nside, l_max, n_maps, map_in, alm_out_re, alm_out_im);
}

// Convenience aliases
void map2alm_cuda_v6_f64(int nside, int l_max, int n_maps,
                          const double* map_in,
                          double* alm_out_re, double* alm_out_im) {
    map2alm_cuda_v6_f64_f64(nside, l_max, n_maps, map_in, alm_out_re, alm_out_im);
}

void map2alm_cuda_v6_f32(int nside, int l_max, int n_maps,
                          const float* map_in,
                          float* alm_out_re, float* alm_out_im) {
    map2alm_cuda_v6_f32_f32(nside, l_max, n_maps, map_in, alm_out_re, alm_out_im);
}

// Legacy complex_t output wrapper
void map2alm_cuda_v6(int nside, int l_max, int n_maps,
                      const real_t* map_in, complex_t* alm_out) {
    int lp1 = l_max + 1;
    size_t alm_size = (size_t)n_maps * lp1 * lp1;

    // Allocate temporary separate real/imag buffers
    double *alm_re, *alm_im;
    CUDA_CHECK(cudaMalloc(&alm_re, alm_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&alm_im, alm_size * sizeof(double)));

    map2alm_cuda_v6_f64_f64(nside, l_max, n_maps, map_in, alm_re, alm_im);

    // Interleave to complex output
    double* alm_out_ptr = (double*)alm_out;
    CUDA_CHECK(cudaMemcpy2D(alm_out_ptr, 2 * sizeof(double),
                            alm_re, sizeof(double),
                            sizeof(double), alm_size,
                            cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(alm_out_ptr + 1, 2 * sizeof(double),
                            alm_im, sizeof(double),
                            sizeof(double), alm_size,
                            cudaMemcpyDeviceToDevice));

    cudaFree(alm_re);
    cudaFree(alm_im);
}

// ============================================================================
// Runtime configuration API
// ============================================================================

// Set Phase 1 method: 0=DFT (default), 1=FFT_EQUATORIAL, 2=BLUESTEIN
void map2alm_v6_set_phase1_method(int method) {
    switch (method) {
        case 0:
            g_phase1_method = Phase1Method::DFT;
            break;
        case 1:
            g_phase1_method = Phase1Method::FFT_EQUATORIAL;
            break;
        case 2:
            g_phase1_method = Phase1Method::BLUESTEIN;
            break;
        default:
            fprintf(stderr, "Warning: Unknown phase1 method %d, using DFT\n", method);
            g_phase1_method = Phase1Method::DFT;
            break;
    }
}

// Get current Phase 1 method
int map2alm_v6_get_phase1_method() {
    return static_cast<int>(g_phase1_method);
}

} // extern "C"

// ============================================================================
// SPIN-2 IMPLEMENTATION
// ============================================================================
// Spin-2 transforms for polarization (Q,U) <-> (E,B) modes
// Reference: jax_healpix/SPHT_jax.py and YLM_jax.py
//
// Key formulas:
//   ₂Y_{l,m} = norm * [(2(m²-l)/sin² - l(l-1)) * Y_{l,m} + 2*alpha*cos/sin² * Y_{l-1,m}]
//   ₋₂Y_{l,m} = norm * [2m/sin² * (alpha * Y_{l-1,m} - (l-1)*cos * Y_{l,m})]
//   where norm = sqrt((l-2)!/(l+2)!), alpha = sqrt((2l+1)(l²-m²)/(2l-1))
//
// map2alm spin-2:
//   E-mode: alm[2] = ∫Q×₂Y + i×∫U×₋₂Y
//   B-mode: alm[-2] = ∫Q×₋₂Y + i×∫U×₂Y
//   Post-processing: alm[-2] *= i, alm[2] *= -1
//
// South ring symmetry:
//   ₂Y(-θ) = (-1)^(l+m) × ₂Y(θ)
//   ₋₂Y(-θ) = (-1)^(l+m+1) × ₋₂Y(θ)  [extra sign flip!]
// ============================================================================

// Compute spin-2 normalization factor: sqrt((l-2)!/(l+2)!)
template<typename C>
__device__ __forceinline__ C compute_spin2_norm(int l) {
    // norm = sqrt((l-2)!/(l+2)!) = 1 / sqrt((l-1)*l*(l+1)*(l+2))
    if (l < 2) return C(0);
    C prod = C((l-1) * l) * C((l+1) * (l+2));
    return C(1.0) / sqrt(prod);
}

// Compute alpha_{l,m} = sqrt((2l+1)(l²-m²)/(2l-1))
template<typename C>
__device__ __forceinline__ C compute_alpha_lm(int l, int m) {
    if (l <= 1) return C(0);
    C l2 = C(l * l);
    C m2 = C(m * m);
    return sqrt(C(2*l + 1) * (l2 - m2) / C(2*l - 1));
}

// ============================================================================
// Spin-2 Phase 2 kernel: Reduce to E,B alm with inline spin-2 Ylm computation
// Sequential Q/U loading: halves shared memory (4 arrays instead of 8 per map)
// Takes Gm for Q and U maps, outputs E and B alm coefficients
// ============================================================================

template<typename T, typename R>
__global__ void reduce_to_alm_spin2_kernel_v6(
    int nside, int l_max, int n_maps, int n_north_rings,
    int ring_batch_size, int n_maps_parallel,
    const R* __restrict__ Gm_Q_north_re,   // Q map Gm [n_maps, lp1, n_north_rings]
    const R* __restrict__ Gm_Q_north_im,
    const R* __restrict__ Gm_Q_south_re,
    const R* __restrict__ Gm_Q_south_im,
    const R* __restrict__ Gm_U_north_re,   // U map Gm
    const R* __restrict__ Gm_U_north_im,
    const R* __restrict__ Gm_U_south_re,
    const R* __restrict__ Gm_U_south_im,
    const R* __restrict__ cos_theta,
    const R* __restrict__ sin_theta,
    R pix_area,
    T* __restrict__ alm_E_re,    // E-mode output
    T* __restrict__ alm_E_im,
    T* __restrict__ alm_B_re,    // B-mode output
    T* __restrict__ alm_B_im
) {
    using Traits = V6Traits<R>;
    using C = typename Traits::compute_t;

    int m = blockIdx.x;
    int lane = threadIdx.x;
    int lp1 = l_max + 1;

    if (m > l_max || lane >= 32) return;

    // Shared memory layout: geometry (2 arrays) + Gm (4 arrays per map for Q or U)
    // With sequential Q/U loading, we only need 4 Gm arrays per map at a time
    extern __shared__ char smem[];
    R* sh_cos_th = (R*)smem;
    R* sh_sin_th = sh_cos_th + ring_batch_size;
    R* sh_Gm_base = sh_sin_th + ring_batch_size;

    // Per-lane Ylm recurrence state
    C Ylm_prev1[MAX_RINGS_PER_LANE];
    C Ylm_prev2[MAX_RINGS_PER_LANE];
    // Save initial Ylm state for U pass
    C Ylm_saved1[MAX_RINGS_PER_LANE];
    C Ylm_saved2[MAX_RINGS_PER_LANE];

    // Accumulators for E and B modes
    C sum_E_re[MAX_PARALLEL_MAPS_F32];
    C sum_E_im[MAX_PARALLEL_MAPS_F32];
    C sum_B_re[MAX_PARALLEL_MAPS_F32];
    C sum_B_im[MAX_PARALLEL_MAPS_F32];

    for (int map_batch_start = 0; map_batch_start < n_maps; map_batch_start += n_maps_parallel) {
        int map_batch_end = min(map_batch_start + n_maps_parallel, n_maps);
        int n_maps_in_batch = map_batch_end - map_batch_start;

        for (int batch_start = 0; batch_start < n_north_rings; batch_start += ring_batch_size) {
            int batch_end = min(batch_start + ring_batch_size, n_north_rings);
            int batch_size = batch_end - batch_start;

            // Load geometry (shared across Q and U passes)
            for (int r = lane; r < batch_size; r += 32) {
                int global_r = batch_start + r;
                sh_cos_th[r] = cos_theta[global_r];
                sh_sin_th[r] = sin_theta[global_r];
            }
            __syncwarp();

            // Ring indices for this lane
            int k_start = (batch_start > lane) ? (batch_start - lane + 31) / 32 : 0;
            int k_end = (batch_end > lane) ? (batch_end - 1 - lane) / 32 + 1 : 0;
            int n_my_rings_total = (n_north_rings + 31 - lane) / 32;
            k_end = min(k_end, n_my_rings_total);

            // Initialize Y[m,m] and advance recurrence to l_start if needed
            int l_start = max(2, m);  // spin-2 starts at l=2

            for (int k = k_start; k < k_end; k++) {
                int global_r = lane + 32 * k;
                int local_r = global_r - batch_start;
                C cos_th = C(sh_cos_th[local_r]);
                C sin_th = C(sh_sin_th[local_r]);

                // Start with Y[m,m]
                C Ymm = C(1.0) / Traits::sqrt_d(C(4.0 * Traits::PI_VAL));
                for (int j = 1; j <= m; j++) {
                    Ymm *= -sin_th * Traits::sqrt_d(C(2*j + 1) / C(2*j));
                }

                // Advance to l_start-1 for spin-2 (l >= 2)
                C Yl_curr = Ymm;
                C Yl_prev = C(0);

                for (int l = m + 1; l < l_start; l++) {
                    if (l == m + 1) {
                        C Yl_new = cos_th * Traits::sqrt_d(C(2*m + 3)) * Yl_curr;
                        Yl_prev = Yl_curr;
                        Yl_curr = Yl_new;
                    } else {
                        C l2 = C(l * l);
                        C lm1_2 = C((l-1) * (l-1));
                        C m2 = C(m * m);
                        C A = Traits::sqrt_d((C(4)*l2 - C(1)) / (l2 - m2));
                        C B = Traits::sqrt_d((C(2*l + 1)) / (C(2*l - 3)) * (lm1_2 - m2) / (l2 - m2));
                        C Yl_new = A * cos_th * Yl_curr - B * Yl_prev;
                        Yl_prev = Yl_curr;
                        Yl_curr = Yl_new;
                    }
                }

                Ylm_prev1[k] = Yl_curr;
                Ylm_prev2[k] = Yl_prev;
                // Save for U pass
                Ylm_saved1[k] = Yl_curr;
                Ylm_saved2[k] = Yl_prev;
            }

            // ================================================================
            // Q PASS: Load Q_north, Q_south, compute Q contributions
            // E_re += Q × ₂Y (parity_s0), B_re += Q × ₋₂Y (parity_m2)
            // E_im += Q × ₂Y (parity_s0), B_im += Q × ₋₂Y (parity_m2)
            // ================================================================
            for (int t = 0; t < n_maps_in_batch; t++) {
                int global_t = map_batch_start + t;
                size_t base_idx = (size_t)global_t * lp1 * n_north_rings + (size_t)m * n_north_rings;
                R* sh_Gm_t = sh_Gm_base + t * 4 * ring_batch_size;

                for (int r = lane; r < batch_size; r += 32) {
                    size_t idx = base_idx + batch_start + r;
                    sh_Gm_t[0 * ring_batch_size + r] = Gm_Q_north_re[idx];
                    sh_Gm_t[1 * ring_batch_size + r] = Gm_Q_north_im[idx];
                    sh_Gm_t[2 * ring_batch_size + r] = Gm_Q_south_re[idx];
                    sh_Gm_t[3 * ring_batch_size + r] = Gm_Q_south_im[idx];
                }
            }
            __syncwarp();

            for (int l = l_start; l <= l_max; l++) {
                // Reset accumulators (partial - Q contribution only)
                for (int t = 0; t < n_maps_in_batch; t++) {
                    sum_E_re[t] = C(0);
                    sum_E_im[t] = C(0);
                    sum_B_re[t] = C(0);
                    sum_B_im[t] = C(0);
                }

                // Precompute l-dependent coefficients
                C norm = compute_spin2_norm<C>(l);
                C alpha = compute_alpha_lm<C>(l, m);
                C m2 = C(m * m);
                C ll1 = C(l * (l - 1));
                C two_alpha = C(2) * alpha;
                C two_m = C(2) * C(m);
                C lm1 = C(l - 1);
                C two_m2_minus_l = C(2) * (m2 - C(l));

                int parity_s0 = (l + m) & 1;
                int parity_m2 = (l + m + 1) & 1;
                C sign_s0 = parity_s0 ? C(-1) : C(1);
                C sign_m2 = parity_m2 ? C(-1) : C(1);

                C recur_A = C(0), recur_B = C(0), recur_C = C(0);
                if (l == m + 1) {
                    recur_C = Traits::sqrt_d(C(2*m + 3));
                } else if (l > m + 1) {
                    C l2 = C(l * l);
                    C lm1_2 = C((l-1) * (l-1));
                    recur_A = Traits::sqrt_d((C(4)*l2 - C(1)) / (l2 - m2));
                    recur_B = Traits::sqrt_d((C(2*l + 1)) / (C(2*l - 3)) * (lm1_2 - m2) / (l2 - m2));
                }

                for (int k = k_start; k < k_end; k++) {
                    int global_r = lane + 32 * k;
                    int local_r = global_r - batch_start;
                    C cos_th = C(sh_cos_th[local_r]);
                    C sin_th = C(sh_sin_th[local_r]);
                    C sin_th_sq = sin_th * sin_th;
                    C inv_sin_sq = (sin_th_sq > C(1e-20)) ? C(1.0) / sin_th_sq : C(0);

                    // Ylm recurrence
                    C Ylm, Ylm_prev;
                    if (l == m) {
                        Ylm = Ylm_prev1[k];
                        Ylm_prev = C(0);
                    } else if (l == m + 1) {
                        Ylm = cos_th * recur_C * Ylm_prev1[k];
                        Ylm_prev = Ylm_prev1[k];
                        Ylm_prev2[k] = Ylm_prev1[k];
                        Ylm_prev1[k] = Ylm;
                    } else {
                        Ylm = recur_A * cos_th * Ylm_prev1[k] - recur_B * Ylm_prev2[k];
                        Ylm_prev = Ylm_prev1[k];
                        Ylm_prev2[k] = Ylm_prev1[k];
                        Ylm_prev1[k] = Ylm;
                    }

                    // Compute ₂Y and ₋₂Y
                    C coeff1 = two_m2_minus_l * inv_sin_sq - ll1;
                    C coeff2 = two_alpha * cos_th * inv_sin_sq;
                    C Y2 = coeff1 * Ylm + coeff2 * Ylm_prev;

                    C inner = alpha * Ylm_prev - lm1 * cos_th * Ylm;
                    C Ym2 = two_m * inv_sin_sq * inner;

                    // Q contributions (E_re, E_im use Y2; B_re, B_im use Ym2)
                    for (int t = 0; t < n_maps_in_batch; t++) {
                        R* sh_Gm_t = sh_Gm_base + t * 4 * ring_batch_size;

                        C Q_north_re = C(sh_Gm_t[0 * ring_batch_size + local_r]);
                        C Q_north_im = C(sh_Gm_t[1 * ring_batch_size + local_r]);
                        C Q_south_re = C(sh_Gm_t[2 * ring_batch_size + local_r]);
                        C Q_south_im = C(sh_Gm_t[3 * ring_batch_size + local_r]);

                        // Q with parity_s0 for ₂Y → E
                        C gm_Q_s0_re = Q_north_re + sign_s0 * Q_south_re;
                        C gm_Q_s0_im = Q_north_im + sign_s0 * Q_south_im;
                        sum_E_re[t] += Y2 * gm_Q_s0_re;
                        sum_E_im[t] += Y2 * gm_Q_s0_im;

                        // Q with parity_m2 for ₋₂Y → B
                        C gm_Q_m2_re = Q_north_re + sign_m2 * Q_south_re;
                        C gm_Q_m2_im = Q_north_im + sign_m2 * Q_south_im;
                        sum_B_re[t] += Ym2 * gm_Q_m2_re;
                        sum_B_im[t] += Ym2 * gm_Q_m2_im;
                    }
                }

                // Warp reduce and write Q partial results
                for (int t = 0; t < n_maps_in_batch; t++) {
                    C sEr = sum_E_re[t], sEi = sum_E_im[t];
                    C sBr = sum_B_re[t], sBi = sum_B_im[t];

                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        sEr += __shfl_down_sync(0xffffffff, sEr, offset);
                        sEi += __shfl_down_sync(0xffffffff, sEi, offset);
                        sBr += __shfl_down_sync(0xffffffff, sBr, offset);
                        sBi += __shfl_down_sync(0xffffffff, sBi, offset);
                    }

                    if (lane == 0) {
                        int global_t = map_batch_start + t;
                        T* E_re_t = alm_E_re + (size_t)global_t * lp1 * lp1;
                        T* E_im_t = alm_E_im + (size_t)global_t * lp1 * lp1;
                        T* B_re_t = alm_B_re + (size_t)global_t * lp1 * lp1;
                        T* B_im_t = alm_B_im + (size_t)global_t * lp1 * lp1;

                        C scale = norm * C(pix_area);

                        // Q pass: write (first batch) or add (subsequent batches)
                        if (batch_start == 0) {
                            E_re_t[l * lp1 + m] = T(sEr * scale);
                            E_im_t[l * lp1 + m] = T(sEi * scale);
                            B_re_t[l * lp1 + m] = T(sBr * scale);
                            B_im_t[l * lp1 + m] = T(sBi * scale);
                        } else {
                            E_re_t[l * lp1 + m] = T(C(E_re_t[l * lp1 + m]) + sEr * scale);
                            E_im_t[l * lp1 + m] = T(C(E_im_t[l * lp1 + m]) + sEi * scale);
                            B_re_t[l * lp1 + m] = T(C(B_re_t[l * lp1 + m]) + sBr * scale);
                            B_im_t[l * lp1 + m] = T(C(B_im_t[l * lp1 + m]) + sBi * scale);
                        }
                    }
                }
            }

            // ================================================================
            // U PASS: Restore Ylm, load U_north, U_south, compute U contributions
            // E_re -= U × ₋₂Y (parity_m2) [imaginary part of i×U]
            // E_im += U × ₋₂Y (parity_m2) [real part of i×U]
            // B_re -= U × ₂Y (parity_s0) [imaginary part of i×U]
            // B_im += U × ₂Y (parity_s0) [real part of i×U]
            // ================================================================

            // Restore Ylm state
            for (int k = k_start; k < k_end; k++) {
                Ylm_prev1[k] = Ylm_saved1[k];
                Ylm_prev2[k] = Ylm_saved2[k];
            }

            // Load U data (reusing same shared memory)
            for (int t = 0; t < n_maps_in_batch; t++) {
                int global_t = map_batch_start + t;
                size_t base_idx = (size_t)global_t * lp1 * n_north_rings + (size_t)m * n_north_rings;
                R* sh_Gm_t = sh_Gm_base + t * 4 * ring_batch_size;

                for (int r = lane; r < batch_size; r += 32) {
                    size_t idx = base_idx + batch_start + r;
                    sh_Gm_t[0 * ring_batch_size + r] = Gm_U_north_re[idx];
                    sh_Gm_t[1 * ring_batch_size + r] = Gm_U_north_im[idx];
                    sh_Gm_t[2 * ring_batch_size + r] = Gm_U_south_re[idx];
                    sh_Gm_t[3 * ring_batch_size + r] = Gm_U_south_im[idx];
                }
            }
            __syncwarp();

            for (int l = l_start; l <= l_max; l++) {
                // Reset accumulators for U contributions
                for (int t = 0; t < n_maps_in_batch; t++) {
                    sum_E_re[t] = C(0);
                    sum_E_im[t] = C(0);
                    sum_B_re[t] = C(0);
                    sum_B_im[t] = C(0);
                }

                // Same l-dependent coefficients as Q pass
                C norm = compute_spin2_norm<C>(l);
                C alpha = compute_alpha_lm<C>(l, m);
                C m2 = C(m * m);
                C ll1 = C(l * (l - 1));
                C two_alpha = C(2) * alpha;
                C two_m = C(2) * C(m);
                C lm1 = C(l - 1);
                C two_m2_minus_l = C(2) * (m2 - C(l));

                int parity_s0 = (l + m) & 1;
                int parity_m2 = (l + m + 1) & 1;
                C sign_s0 = parity_s0 ? C(-1) : C(1);
                C sign_m2 = parity_m2 ? C(-1) : C(1);

                C recur_A = C(0), recur_B = C(0), recur_C = C(0);
                if (l == m + 1) {
                    recur_C = Traits::sqrt_d(C(2*m + 3));
                } else if (l > m + 1) {
                    C l2 = C(l * l);
                    C lm1_2 = C((l-1) * (l-1));
                    recur_A = Traits::sqrt_d((C(4)*l2 - C(1)) / (l2 - m2));
                    recur_B = Traits::sqrt_d((C(2*l + 1)) / (C(2*l - 3)) * (lm1_2 - m2) / (l2 - m2));
                }

                for (int k = k_start; k < k_end; k++) {
                    int global_r = lane + 32 * k;
                    int local_r = global_r - batch_start;
                    C cos_th = C(sh_cos_th[local_r]);
                    C sin_th = C(sh_sin_th[local_r]);
                    C sin_th_sq = sin_th * sin_th;
                    C inv_sin_sq = (sin_th_sq > C(1e-20)) ? C(1.0) / sin_th_sq : C(0);

                    // Ylm recurrence (duplicated from Q pass)
                    C Ylm, Ylm_prev;
                    if (l == m) {
                        Ylm = Ylm_prev1[k];
                        Ylm_prev = C(0);
                    } else if (l == m + 1) {
                        Ylm = cos_th * recur_C * Ylm_prev1[k];
                        Ylm_prev = Ylm_prev1[k];
                        Ylm_prev2[k] = Ylm_prev1[k];
                        Ylm_prev1[k] = Ylm;
                    } else {
                        Ylm = recur_A * cos_th * Ylm_prev1[k] - recur_B * Ylm_prev2[k];
                        Ylm_prev = Ylm_prev1[k];
                        Ylm_prev2[k] = Ylm_prev1[k];
                        Ylm_prev1[k] = Ylm;
                    }

                    // Compute ₂Y and ₋₂Y
                    C coeff1 = two_m2_minus_l * inv_sin_sq - ll1;
                    C coeff2 = two_alpha * cos_th * inv_sin_sq;
                    C Y2 = coeff1 * Ylm + coeff2 * Ylm_prev;

                    C inner = alpha * Ylm_prev - lm1 * cos_th * Ylm;
                    C Ym2 = two_m * inv_sin_sq * inner;

                    // U contributions via i×U: (i×U)_re = -U_im, (i×U)_im = U_re
                    for (int t = 0; t < n_maps_in_batch; t++) {
                        R* sh_Gm_t = sh_Gm_base + t * 4 * ring_batch_size;

                        C U_north_re = C(sh_Gm_t[0 * ring_batch_size + local_r]);
                        C U_north_im = C(sh_Gm_t[1 * ring_batch_size + local_r]);
                        C U_south_re = C(sh_Gm_t[2 * ring_batch_size + local_r]);
                        C U_south_im = C(sh_Gm_t[3 * ring_batch_size + local_r]);

                        // U with parity_s0 for ₂Y → B (via i×U)
                        C gm_U_s0_re = U_north_re + sign_s0 * U_south_re;
                        C gm_U_s0_im = U_north_im + sign_s0 * U_south_im;
                        // B: ₂Y × (i×U) → B_re -= Y2*U_im, B_im += Y2*U_re
                        sum_B_re[t] -= Y2 * gm_U_s0_im;
                        sum_B_im[t] += Y2 * gm_U_s0_re;

                        // U with parity_m2 for ₋₂Y → E (via i×U)
                        C gm_U_m2_re = U_north_re + sign_m2 * U_south_re;
                        C gm_U_m2_im = U_north_im + sign_m2 * U_south_im;
                        // E: ₋₂Y × (i×U) → E_re -= Ym2*U_im, E_im += Ym2*U_re
                        sum_E_re[t] -= Ym2 * gm_U_m2_im;
                        sum_E_im[t] += Ym2 * gm_U_m2_re;
                    }
                }

                // Warp reduce and ADD U contributions
                for (int t = 0; t < n_maps_in_batch; t++) {
                    C sEr = sum_E_re[t], sEi = sum_E_im[t];
                    C sBr = sum_B_re[t], sBi = sum_B_im[t];

                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        sEr += __shfl_down_sync(0xffffffff, sEr, offset);
                        sEi += __shfl_down_sync(0xffffffff, sEi, offset);
                        sBr += __shfl_down_sync(0xffffffff, sBr, offset);
                        sBi += __shfl_down_sync(0xffffffff, sBi, offset);
                    }

                    if (lane == 0) {
                        int global_t = map_batch_start + t;
                        T* E_re_t = alm_E_re + (size_t)global_t * lp1 * lp1;
                        T* E_im_t = alm_E_im + (size_t)global_t * lp1 * lp1;
                        T* B_re_t = alm_B_re + (size_t)global_t * lp1 * lp1;
                        T* B_im_t = alm_B_im + (size_t)global_t * lp1 * lp1;

                        C scale = norm * C(pix_area);

                        // U pass: always add (Q pass already wrote)
                        E_re_t[l * lp1 + m] = T(C(E_re_t[l * lp1 + m]) + sEr * scale);
                        E_im_t[l * lp1 + m] = T(C(E_im_t[l * lp1 + m]) + sEi * scale);
                        B_re_t[l * lp1 + m] = T(C(B_re_t[l * lp1 + m]) + sBr * scale);
                        B_im_t[l * lp1 + m] = T(C(B_im_t[l * lp1 + m]) + sBi * scale);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Spin-2 host wrapper
// ============================================================================

template<typename T, typename R>
void map2alm_cuda_v6_spin2_impl(
    int nside, int l_max, int n_maps,
    const T* map_Q, const T* map_U,
    T* alm_E_re, T* alm_E_im,
    T* alm_B_re, T* alm_B_im
) {
    int n_rings = 4 * nside - 1;
    int n_north_rings = 2 * nside;
    int lp1 = l_max + 1;

    // Check limits
    int rings_per_lane = (n_north_rings + 31) / 32;
    if (rings_per_lane > MAX_RINGS_PER_LANE) {
        fprintf(stderr, "Error: nside=%d requires %d rings per lane, max is %d\n",
                nside, rings_per_lane, MAX_RINGS_PER_LANE);
        return;
    }

    // Timing
    static bool timing_enabled = (getenv("SPHT_TIMING") != nullptr);
    cudaEvent_t start_p1, end_p1, start_p2, end_p2;
    if (timing_enabled) {
        cudaEventCreate(&start_p1);
        cudaEventCreate(&end_p1);
        cudaEventCreate(&start_p2);
        cudaEventCreate(&end_p2);
        cudaEventRecord(start_p1);
    }

    // Allocate Gm buffers for Q and U maps
    size_t gm_size = (size_t)n_maps * lp1 * n_north_rings * sizeof(R);
    size_t geom_size = n_north_rings * sizeof(R);

    R *Gm_Q_north_re, *Gm_Q_north_im, *Gm_Q_south_re, *Gm_Q_south_im;
    R *Gm_U_north_re, *Gm_U_north_im, *Gm_U_south_re, *Gm_U_south_im;
    R *cos_theta, *sin_theta;

    CUDA_CHECK(cudaMalloc(&Gm_Q_north_re, gm_size));
    CUDA_CHECK(cudaMalloc(&Gm_Q_north_im, gm_size));
    CUDA_CHECK(cudaMalloc(&Gm_Q_south_re, gm_size));
    CUDA_CHECK(cudaMalloc(&Gm_Q_south_im, gm_size));
    CUDA_CHECK(cudaMalloc(&Gm_U_north_re, gm_size));
    CUDA_CHECK(cudaMalloc(&Gm_U_north_im, gm_size));
    CUDA_CHECK(cudaMalloc(&Gm_U_south_re, gm_size));
    CUDA_CHECK(cudaMalloc(&Gm_U_south_im, gm_size));
    CUDA_CHECK(cudaMalloc(&cos_theta, geom_size));
    CUDA_CHECK(cudaMalloc(&sin_theta, geom_size));

    // Phase 1: Compute Gm for Q and U maps using DFT
    // Use two streams to process Q and U in parallel
    int block_size_p1 = min(256, lp1);

    cudaStream_t stream_Q, stream_U;
    CUDA_CHECK(cudaStreamCreate(&stream_Q));
    CUDA_CHECK(cudaStreamCreate(&stream_U));

    // Allocate geometry buffers for U stream (cos/sin already allocated for Q)
    R *cos_theta_U, *sin_theta_U;
    CUDA_CHECK(cudaMalloc(&cos_theta_U, geom_size));
    CUDA_CHECK(cudaMalloc(&sin_theta_U, geom_size));

    // Launch both Gm computations in parallel
    compute_gm_kernel_v6<T, R><<<n_north_rings, block_size_p1, 0, stream_Q>>>(
        nside, l_max, n_maps, n_rings, map_Q,
        Gm_Q_north_re, Gm_Q_north_im, Gm_Q_south_re, Gm_Q_south_im,
        cos_theta, sin_theta
    );
    CUDA_CHECK(cudaGetLastError());

    compute_gm_kernel_v6<T, R><<<n_north_rings, block_size_p1, 0, stream_U>>>(
        nside, l_max, n_maps, n_rings, map_U,
        Gm_U_north_re, Gm_U_north_im, Gm_U_south_re, Gm_U_south_im,
        cos_theta_U, sin_theta_U
    );
    CUDA_CHECK(cudaGetLastError());

    // Wait for both streams to complete before Phase 2
    CUDA_CHECK(cudaStreamSynchronize(stream_Q));
    CUDA_CHECK(cudaStreamSynchronize(stream_U));

    // Cleanup streams and extra geometry buffers
    CUDA_CHECK(cudaStreamDestroy(stream_Q));
    CUDA_CHECK(cudaStreamDestroy(stream_U));
    cudaFree(cos_theta_U);
    cudaFree(sin_theta_U);

    if (timing_enabled) {
        cudaEventRecord(end_p1);
        cudaEventRecord(start_p2);
    }

    // Phase 2: Reduce to E,B alm with spin-2 Ylm
    // Sequential Q/U loading uses 4 arrays per map at a time (Q_north_re/im, Q_south_re/im)
    // This is 2x spin-0's 2 arrays, so we need /=2 to fit in shared memory
    int ring_batch_size, n_maps_parallel;
    compute_v6_params<R>(n_maps, &ring_batch_size, &n_maps_parallel);
    n_maps_parallel = max(1, n_maps_parallel / 2);  // 4 arrays vs 2 for spin-0

    size_t smem_size = (2 + 4 * n_maps_parallel) * ring_batch_size * sizeof(R);
    R pix_area = R(4.0 * M_PI / (12.0 * nside * nside));

    reduce_to_alm_spin2_kernel_v6<T, R><<<lp1, 32, smem_size>>>(
        nside, l_max, n_maps, n_north_rings,
        ring_batch_size, n_maps_parallel,
        Gm_Q_north_re, Gm_Q_north_im, Gm_Q_south_re, Gm_Q_south_im,
        Gm_U_north_re, Gm_U_north_im, Gm_U_south_re, Gm_U_south_im,
        cos_theta, sin_theta, pix_area,
        alm_E_re, alm_E_im, alm_B_re, alm_B_im
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (timing_enabled) {
        cudaEventRecord(end_p2);
        cudaEventSynchronize(end_p2);

        float p1_ms, p2_ms;
        cudaEventElapsedTime(&p1_ms, start_p1, end_p1);
        cudaEventElapsedTime(&p2_ms, start_p2, end_p2);

        fprintf(stderr, "[SPHT_TIMING] spin2 map2alm nside=%d l_max=%d n_maps=%d Phase1=%.2fms Phase2=%.2fms Total=%.2fms\n",
                nside, l_max, n_maps, p1_ms, p2_ms, p1_ms + p2_ms);

        cudaEventDestroy(start_p1);
        cudaEventDestroy(end_p1);
        cudaEventDestroy(start_p2);
        cudaEventDestroy(end_p2);
    }

    // Cleanup
    cudaFree(Gm_Q_north_re);
    cudaFree(Gm_Q_north_im);
    cudaFree(Gm_Q_south_re);
    cudaFree(Gm_Q_south_im);
    cudaFree(Gm_U_north_re);
    cudaFree(Gm_U_north_im);
    cudaFree(Gm_U_south_re);
    cudaFree(Gm_U_south_im);
    cudaFree(cos_theta);
    cudaFree(sin_theta);
}

// ============================================================================
// Spin-2 C API entry points
// ============================================================================

extern "C" {

void map2alm_cuda_v6_spin2_f64_f64(int nside, int l_max, int n_maps,
                                    const double* map_Q, const double* map_U,
                                    double* alm_E_re, double* alm_E_im,
                                    double* alm_B_re, double* alm_B_im) {
    map2alm_cuda_v6_spin2_impl<double, double>(nside, l_max, n_maps,
                                                map_Q, map_U,
                                                alm_E_re, alm_E_im,
                                                alm_B_re, alm_B_im);
}

void map2alm_cuda_v6_spin2_f64_f32(int nside, int l_max, int n_maps,
                                    const double* map_Q, const double* map_U,
                                    double* alm_E_re, double* alm_E_im,
                                    double* alm_B_re, double* alm_B_im) {
    map2alm_cuda_v6_spin2_impl<double, float>(nside, l_max, n_maps,
                                               map_Q, map_U,
                                               alm_E_re, alm_E_im,
                                               alm_B_re, alm_B_im);
}

void map2alm_cuda_v6_spin2_f32_f64(int nside, int l_max, int n_maps,
                                    const float* map_Q, const float* map_U,
                                    float* alm_E_re, float* alm_E_im,
                                    float* alm_B_re, float* alm_B_im) {
    map2alm_cuda_v6_spin2_impl<float, double>(nside, l_max, n_maps,
                                               map_Q, map_U,
                                               alm_E_re, alm_E_im,
                                               alm_B_re, alm_B_im);
}

void map2alm_cuda_v6_spin2_f32_f32(int nside, int l_max, int n_maps,
                                    const float* map_Q, const float* map_U,
                                    float* alm_E_re, float* alm_E_im,
                                    float* alm_B_re, float* alm_B_im) {
    map2alm_cuda_v6_spin2_impl<float, float>(nside, l_max, n_maps,
                                              map_Q, map_U,
                                              alm_E_re, alm_E_im,
                                              alm_B_re, alm_B_im);
}

// Convenience aliases
void map2alm_cuda_v6_spin2_f64(int nside, int l_max, int n_maps,
                                const double* map_Q, const double* map_U,
                                double* alm_E_re, double* alm_E_im,
                                double* alm_B_re, double* alm_B_im) {
    map2alm_cuda_v6_spin2_f64_f64(nside, l_max, n_maps, map_Q, map_U,
                                   alm_E_re, alm_E_im, alm_B_re, alm_B_im);
}

void map2alm_cuda_v6_spin2_f32(int nside, int l_max, int n_maps,
                                const float* map_Q, const float* map_U,
                                float* alm_E_re, float* alm_E_im,
                                float* alm_B_re, float* alm_B_im) {
    map2alm_cuda_v6_spin2_f32_f32(nside, l_max, n_maps, map_Q, map_U,
                                   alm_E_re, alm_E_im, alm_B_re, alm_B_im);
}

} // extern "C" for spin-2

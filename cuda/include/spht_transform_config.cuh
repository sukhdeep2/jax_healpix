/**
 * spht_transform_config.cuh
 *
 * Shared configuration for map2alm and alm2map transforms.
 * Includes Phase1 method selection, memory budget constants, and kernel configuration.
 */

#ifndef SPHT_TRANSFORM_CONFIG_CUH
#define SPHT_TRANSFORM_CONFIG_CUH

#include <cuda_runtime.h>
#include <cufft.h>
#include <map>
#include <mutex>
#include <cstdio>

// ============================================================================
// Runtime configuration for Phase 1 method
// ============================================================================

enum class Phase1Method {
    DFT = 0,           // Direct DFT for all rings (default, simple, good for small nside)
    FFT_EQUATORIAL = 1, // FFT for equatorial rings, DFT for polar (hybrid)
    BLUESTEIN = 2       // Bluestein FFT for all rings (cuHPX-style)
};

// Global configuration - can be changed at runtime
// Defined in map2alm_v6.cu, used by both map2alm and alm2map
extern Phase1Method g_phase1_method;

// ============================================================================
// Memory Budget Constants
// ============================================================================
//
// Per ring Ylm state (log-space):
//   - log_Ylm_prev1, log_Ylm_prev2: 2 × sizeof(R)
//   - sign_prev1, sign_prev2: 2 × sizeof(int8_t)
//   - log_Ymm_saved, sign_Ymm_saved: sizeof(R) + sizeof(int8_t)
//   - log_cos_th_cached, sign_cos_th_cached: sizeof(R) + sizeof(int8_t)
//   Total: 4*sizeof(R) + 4 bytes = 20 bytes (f32) or 36 bytes (f64)
//
// Per T map: 2 accumulators (sum_re, sum_im) = 2*sizeof(R)
// Per pol pair: 4 accumulators (E_re, E_im, B_re, B_im) = 4*sizeof(R)
//

constexpr int MEMORY_BUDGET_BYTES = 1400;
constexpr int OVERHEAD_BYTES = 200;

// Legacy constant for non-templated code
#define MAX_RINGS_PER_LANE 64

// ============================================================================
// Kernel Configuration
// ============================================================================

struct KernelConfig {
    int rings_per_lane;     // Template parameter to use (16, 32, 64, or 128)
    int n_ring_passes;      // Number of passes to cover all rings
    int maps_per_launch;    // Maps processed per kernel launch
    int n_map_launches;     // Number of kernel launches for all maps
};

// Calculate Ylm state size per ring based on precision
template<typename R>
__host__ __device__ constexpr int ylm_state_bytes_per_ring() {
    // 4 arrays of R + 4 arrays of int8_t
    return 4 * sizeof(R) + 4;
}

// Calculate optimal RINGS_PER_LANE for joint T+QU kernel
template<typename R>
inline int select_rings_per_lane_joint(int n_T, int n_pol) {
    int ylm_bytes = ylm_state_bytes_per_ring<R>();
    int acc_bytes = n_T * 2 * sizeof(R) + n_pol * 4 * sizeof(R);
    int budget = MEMORY_BUDGET_BYTES - OVERHEAD_BYTES - acc_bytes;

    if (budget >= 64 * ylm_bytes) return 64;
    if (budget >= 32 * ylm_bytes) return 32;
    if (budget >= 16 * ylm_bytes) return 16;
    return 8;
}

// Calculate optimal kernel configuration based on runtime parameters
template<typename R>
inline KernelConfig calculate_kernel_config(int nside, int n_maps, int spin = 0) {
    int per_ring, per_map;

    if (spin == 0) {
        per_ring = ylm_state_bytes_per_ring<R>();
        per_map = 2 * sizeof(R);
    } else {
        per_ring = 4 * sizeof(R);
        per_map = 4 * sizeof(R);
    }

    int maps_per_launch = n_maps;
    int available = MEMORY_BUDGET_BYTES - OVERHEAD_BYTES - per_map * maps_per_launch;
    int max_rings_per_lane = available / per_ring;

    while (max_rings_per_lane < 1 && maps_per_launch > 1) {
        maps_per_launch--;
        available = MEMORY_BUDGET_BYTES - OVERHEAD_BYTES - per_map * maps_per_launch;
        max_rings_per_lane = available / per_ring;
        fprintf(stderr, "Warning: Reducing maps_per_launch to %d to fit memory budget (spin=%d)\n",
                maps_per_launch, spin);
    }

    int template_size;
    if (max_rings_per_lane >= 128) template_size = 128;
    else if (max_rings_per_lane >= 64) template_size = 64;
    else if (max_rings_per_lane >= 32) template_size = 32;
    else template_size = 16;

    int n_north_rings = 2 * nside;
    int rings_needed_per_lane = (n_north_rings + 31) / 32;
    int n_ring_passes = (rings_needed_per_lane + template_size - 1) / template_size;
    int n_map_launches = (n_maps + maps_per_launch - 1) / maps_per_launch;

    return {template_size, n_ring_passes, maps_per_launch, n_map_launches};
}

// Configuration for joint T+QU kernel
template<typename R>
inline KernelConfig calculate_kernel_config_joint(int nside, int n_T, int n_pol) {
    int rings_per_lane = select_rings_per_lane_joint<R>(n_T, n_pol);

    int n_north_rings = 2 * nside;
    int rings_needed_per_lane = (n_north_rings + 31) / 32;
    int n_ring_passes = (rings_needed_per_lane + rings_per_lane - 1) / rings_per_lane;

    return {rings_per_lane, n_ring_passes, n_T + n_pol, 1};
}

// ============================================================================
// cuFFT Plan Cache
// ============================================================================

struct CufftPlanKey {
    int fft_size;
    int batch;
    cufftType type;

    bool operator<(const CufftPlanKey& other) const {
        if (fft_size != other.fft_size) return fft_size < other.fft_size;
        if (batch != other.batch) return batch < other.batch;
        return type < other.type;
    }
};

// Singleton plan cache (implementation in map2alm_v6.cu)
cufftHandle get_cached_cufft_plan(int fft_size, int batch, cufftType type);

// ============================================================================
// Ring Geometry Helper
// ============================================================================

template<typename T>
__device__ __forceinline__ void compute_ring_geom(
    int ring_idx,     // 0-indexed ring (0 = north pole ring)
    int nside,
    T* cos_th,
    T* sin_th,
    int* first_pix,
    int* n_pix
) {
    int npix = 12 * nside * nside;
    int n_north_rings = 2 * nside;

    if (ring_idx < nside) {
        // North polar cap
        int r = ring_idx + 1;
        *n_pix = 4 * r;
        *first_pix = 2 * r * (r - 1);
        T z = T(1) - T(r * r) / T(3 * nside * nside);
        *cos_th = z;
        *sin_th = sqrt(T(1) - z * z);
    } else if (ring_idx < 3 * nside) {
        // Equatorial belt
        int r = ring_idx + 1;
        *n_pix = 4 * nside;
        int npix_north_cap = 2 * nside * (nside - 1);
        *first_pix = npix_north_cap + (r - nside) * 4 * nside;
        T z = T(2 * nside - r) * T(2) / T(3 * nside);
        *cos_th = z;
        *sin_th = sqrt(T(1) - z * z);
    } else {
        // South polar cap
        int r = ring_idx + 1;
        int s = 4 * nside - r;
        *n_pix = 4 * s;
        *first_pix = npix - 2 * s * (s + 1);
        T z = T(-1) + T(s * s) / T(3 * nside * nside);
        *cos_th = z;
        *sin_th = sqrt(T(1) - z * z);
    }
}

#endif // SPHT_TRANSFORM_CONFIG_CUH

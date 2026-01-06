/**
 * Bluestein FFT Core: cuFFT Plan Cache and Shared Kernels
 */

#include "bluestein_fft.cuh"
#include <unordered_map>
#include <mutex>
#include <stdio.h>

// ============================================================================
// cuFFT Plan Cache Implementation
// ============================================================================

struct CufftPlanKey {
    int size;
    int batch;
    cufftType type;

    bool operator==(const CufftPlanKey& other) const {
        return size == other.size && batch == other.batch && type == other.type;
    }
};

struct CufftPlanKeyHash {
    size_t operator()(const CufftPlanKey& k) const {
        size_t h = std::hash<int>()(k.size);
        h ^= std::hash<int>()(k.batch) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(static_cast<int>(k.type)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static std::unordered_map<CufftPlanKey, cufftHandle, CufftPlanKeyHash> g_cufft_cache;
static std::mutex g_cufft_cache_mutex;

cufftHandle get_cached_cufft_plan(int size, int batch, cufftType type) {
    CufftPlanKey key{size, batch, type};
    std::lock_guard<std::mutex> lock(g_cufft_cache_mutex);

    auto it = g_cufft_cache.find(key);
    if (it != g_cufft_cache.end()) {
        return it->second;
    }

    cufftHandle plan;
    cufftResult result = cufftPlan1d(&plan, size, type, batch);
    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT plan creation failed: size=%d, batch=%d, type=%d, error=%d\n",
                size, batch, static_cast<int>(type), result);
        return 0;
    }

    g_cufft_cache[key] = plan;
    return plan;
}

void clear_cufft_plan_cache() {
    std::lock_guard<std::mutex> lock(g_cufft_cache_mutex);
    for (auto& pair : g_cufft_cache) {
        cufftDestroy(pair.second);
    }
    g_cufft_cache.clear();
}

// Legacy alias for backward compatibility
cufftHandle get_cached_fft_plan(int size, int batch, cufftType type) {
    return get_cached_cufft_plan(size, batch, type);
}

// ============================================================================
// Conjugate Chirp Computation Kernels
// ============================================================================

__global__ void bluestein_compute_conj_chirp_f64(
    int nside, int M,
    cufftDoubleComplex* __restrict__ conj_chirp_fft
) {
    int size_idx = blockIdx.x;
    int N = 4 * (size_idx + 1);
    if (N > 4 * nside) return;

    cufftDoubleComplex* chirp = conj_chirp_fft + size_idx * M;
    double pi_over_N = PrecisionTraits<double>::PI_VAL / double(N);

    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        cufftDoubleComplex val;
        int j_eff;

        if (j < N) {
            j_eff = j;
        } else if (j >= M - N + 1) {
            j_eff = M - j;
        } else {
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

__global__ void bluestein_compute_conj_chirp_f32(
    int nside, int M,
    cufftComplex* __restrict__ conj_chirp_fft
) {
    int size_idx = blockIdx.x;
    int N = 4 * (size_idx + 1);
    if (N > 4 * nside) return;

    cufftComplex* chirp = conj_chirp_fft + size_idx * M;
    float pi_over_N = PrecisionTraits<float>::PI_VAL / float(N);

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

// ============================================================================
// Pointwise Multiplication Kernels
// ============================================================================

__global__ void bluestein_pointwise_mult_f64(
    int n_maps, int n_rings, int M,
    const int* __restrict__ ring_sizes,
    cufftDoubleComplex* __restrict__ fft_data,
    const cufftDoubleComplex* __restrict__ conj_chirp_fft
) {
    int ring_idx = blockIdx.x;
    int map_idx = blockIdx.y;
    if (ring_idx >= n_rings || map_idx >= n_maps) return;

    int N = ring_sizes[ring_idx];
    int size_idx = (N / 4) - 1;

    cufftDoubleComplex* data = fft_data + (size_t)map_idx * n_rings * M + ring_idx * M;
    const cufftDoubleComplex* chirp = conj_chirp_fft + size_idx * M;

    for (int k = threadIdx.x; k < M; k += blockDim.x) {
        cufftDoubleComplex d = data[k];
        cufftDoubleComplex h = chirp[k];
        cufftDoubleComplex result;
        result.x = d.x * h.x - d.y * h.y;
        result.y = d.x * h.y + d.y * h.x;
        data[k] = result;
    }
}

__global__ void bluestein_pointwise_mult_f32(
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

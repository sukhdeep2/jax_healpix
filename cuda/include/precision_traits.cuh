/**
 * Unified Precision Traits for SPHT CUDA kernels
 *
 * Provides type traits for multi-precision support (float64, float32, bfloat16).
 * Consolidates previously duplicated V6Traits and V6TraitsSynth.
 */

#ifndef PRECISION_TRAITS_CUH
#define PRECISION_TRAITS_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cufft.h>

// ============================================================================
// Precision Traits - unified type abstraction for all kernels
// ============================================================================

template<typename T>
struct PrecisionTraits;

template<>
struct PrecisionTraits<double> {
    // Storage and compute types
    using storage_t = double;
    using compute_t = double;
    using complex_storage_t = double2;
    using cufft_complex_t = cufftDoubleComplex;
    using cufft_real_t = cufftDoubleReal;
    static constexpr cufftType cufft_c2c_type = CUFFT_Z2Z;
    static constexpr cufftType cufft_r2c_type = CUFFT_D2Z;
    static constexpr cufftType cufft_c2r_type = CUFFT_Z2D;

    // Mathematical constants
    static constexpr double PI_VAL = 3.14159265358979323846;
    static constexpr double TWO_PI_VAL = 6.28318530717958647692;
    static constexpr double LOG_PI_VAL = 1.1447298858494002;   // log(pi)
    static constexpr double LOG_2_VAL = 0.6931471805599453;    // log(2)
    static constexpr double LOG_3_VAL = 1.0986122886681098;    // log(3)
    static constexpr double LOG_4_VAL = 1.3862943611198906;    // log(4)
    static constexpr double LOG_4PI_VAL = 2.5310242469692907;  // log(4*pi)

    // Precision limits
    static constexpr double LOG_MIN = -700.0;
    static constexpr double LOG_MAX = 700.0;
    static constexpr double EPS = 1e-15;

    // Math functions
    static __device__ __forceinline__ double sqrt_d(double x) { return sqrt(x); }
    static __device__ __forceinline__ double exp_d(double x) { return exp(x); }
    static __device__ __forceinline__ double log_d(double x) { return log(x); }
    static __device__ __forceinline__ double abs_d(double x) { return fabs(x); }
    static __device__ __forceinline__ double cos_d(double x) { return cos(x); }
    static __device__ __forceinline__ double sin_d(double x) { return sin(x); }
    static __device__ __forceinline__ void sincos_d(double x, double* s, double* c) {
        sincos(x, s, c);
    }

    // Memory access
    static __device__ __forceinline__ double load(const double* p) { return __ldg(p); }
    static __device__ __forceinline__ double2 load2(const double2* p) { return *p; }

    // Type conversion
    static __device__ __forceinline__ double from_storage(double x) { return x; }
    static __device__ __forceinline__ double to_storage(double x) { return x; }

    // Complex helpers
    static __device__ __forceinline__ cufftDoubleComplex make_complex(double re, double im) {
        cufftDoubleComplex c;
        c.x = re;
        c.y = im;
        return c;
    }
};

template<>
struct PrecisionTraits<float> {
    using storage_t = float;
    using compute_t = float;
    using complex_storage_t = float2;
    using cufft_complex_t = cufftComplex;
    using cufft_real_t = cufftReal;
    static constexpr cufftType cufft_c2c_type = CUFFT_C2C;
    static constexpr cufftType cufft_r2c_type = CUFFT_R2C;
    static constexpr cufftType cufft_c2r_type = CUFFT_C2R;

    static constexpr float PI_VAL = 3.14159265f;
    static constexpr float TWO_PI_VAL = 6.28318530f;
    static constexpr float LOG_PI_VAL = 1.14472988f;
    static constexpr float LOG_2_VAL = 0.69314718f;
    static constexpr float LOG_3_VAL = 1.09861228f;
    static constexpr float LOG_4_VAL = 1.38629436f;
    static constexpr float LOG_4PI_VAL = 2.53102425f;

    static constexpr float LOG_MIN = -87.0f;
    static constexpr float LOG_MAX = 88.0f;
    static constexpr float EPS = 1e-6f;

    static __device__ __forceinline__ float sqrt_d(float x) { return sqrtf(x); }
    static __device__ __forceinline__ float exp_d(float x) { return expf(x); }
    static __device__ __forceinline__ float log_d(float x) { return logf(x); }
    static __device__ __forceinline__ float abs_d(float x) { return fabsf(x); }
    static __device__ __forceinline__ float cos_d(float x) { return cosf(x); }
    static __device__ __forceinline__ float sin_d(float x) { return sinf(x); }
    static __device__ __forceinline__ void sincos_d(float x, float* s, float* c) {
        sincosf(x, s, c);
    }

    static __device__ __forceinline__ float load(const float* p) { return __ldg(p); }
    static __device__ __forceinline__ float2 load2(const float2* p) { return *p; }

    static __device__ __forceinline__ float from_storage(float x) { return x; }
    static __device__ __forceinline__ float to_storage(float x) { return x; }

    static __device__ __forceinline__ cufftComplex make_complex(float re, float im) {
        cufftComplex c;
        c.x = re;
        c.y = im;
        return c;
    }
};

template<>
struct PrecisionTraits<__nv_bfloat16> {
    using storage_t = __nv_bfloat16;
    using compute_t = float;  // Compute in float for numerical stability
    using complex_storage_t = __nv_bfloat162;
    using cufft_complex_t = cufftComplex;  // Use float FFT for bfloat16
    using cufft_real_t = cufftReal;
    static constexpr cufftType cufft_c2c_type = CUFFT_C2C;
    static constexpr cufftType cufft_r2c_type = CUFFT_R2C;
    static constexpr cufftType cufft_c2r_type = CUFFT_C2R;

    // Use float constants for compute
    static constexpr float PI_VAL = 3.14159265f;
    static constexpr float TWO_PI_VAL = 6.28318530f;
    static constexpr float LOG_PI_VAL = 1.14472988f;
    static constexpr float LOG_2_VAL = 0.69314718f;
    static constexpr float LOG_3_VAL = 1.09861228f;
    static constexpr float LOG_4_VAL = 1.38629436f;
    static constexpr float LOG_4PI_VAL = 2.53102425f;

    static constexpr float LOG_MIN = -87.0f;
    static constexpr float LOG_MAX = 88.0f;
    static constexpr float EPS = 1e-3f;  // Lower precision for bfloat16

    static __device__ __forceinline__ float sqrt_d(float x) { return sqrtf(x); }
    static __device__ __forceinline__ float exp_d(float x) { return expf(x); }
    static __device__ __forceinline__ float log_d(float x) { return logf(x); }
    static __device__ __forceinline__ float abs_d(float x) { return fabsf(x); }
    static __device__ __forceinline__ float cos_d(float x) { return cosf(x); }
    static __device__ __forceinline__ float sin_d(float x) { return sinf(x); }
    static __device__ __forceinline__ void sincos_d(float x, float* s, float* c) {
        sincosf(x, s, c);
    }

    static __device__ __forceinline__ float load(const __nv_bfloat16* p) {
        return __bfloat162float(__ldg(p));
    }
    static __device__ __forceinline__ float2 load2(const __nv_bfloat162* p) {
        __nv_bfloat162 v = *p;
        return make_float2(__bfloat162float(v.x), __bfloat162float(v.y));
    }

    static __device__ __forceinline__ float from_storage(__nv_bfloat16 x) {
        return __bfloat162float(x);
    }
    static __device__ __forceinline__ __nv_bfloat16 to_storage(float x) {
        return __float2bfloat16(x);
    }

    static __device__ __forceinline__ cufftComplex make_complex(float re, float im) {
        cufftComplex c;
        c.x = re;
        c.y = im;
        return c;
    }
};

// ============================================================================
// Helper type aliases for common patterns
// ============================================================================

// Get the appropriate cuFFT complex type for a given precision
template<typename T>
using CufftComplex = typename PrecisionTraits<T>::cufft_complex_t;

// Get the compute type (may differ from storage for bfloat16)
template<typename T>
using ComputeType = typename PrecisionTraits<T>::compute_t;

#endif // PRECISION_TRAITS_CUH

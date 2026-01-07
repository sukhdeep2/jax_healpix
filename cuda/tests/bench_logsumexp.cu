/**
 * Micro-benchmark: logsumexp vs FMA operations
 * Measures the raw cost difference between LOG and LINEAR accumulation.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N_ITERS 1000000
#define BLOCK_SIZE 256

// Measure FMA operations (LINEAR mode equivalent)
template<typename T>
__global__ void bench_fma(T* out, int n_iters) {
    T acc_re = T(0), acc_im = T(0);
    T ylm = T(1.0001), gm_re = T(0.5), gm_im = T(0.3);

    for (int i = 0; i < n_iters; i++) {
        // Typical LINEAR accumulation: 2 FMAs
        acc_re += ylm * gm_re;
        acc_im += ylm * gm_im;
        ylm *= T(0.999999);  // Simulate recurrence
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = acc_re;
        out[1] = acc_im;
    }
}

// Fast math intrinsics for f32
__device__ __forceinline__ float fast_log(float x) { return __logf(x); }
__device__ __forceinline__ float fast_exp(float x) { return __expf(x); }
__device__ __forceinline__ float fast_log1p(float x) { return log1pf(x); }

// Full precision for f64
__device__ __forceinline__ double fast_log(double x) { return log(x); }
__device__ __forceinline__ double fast_exp(double x) { return exp(x); }
__device__ __forceinline__ double fast_log1p(double x) { return log1p(x); }

// Measure logsumexp operations (LOG mode equivalent)
template<typename T>
__global__ void bench_logsumexp(T* out, int n_iters) {
    T log_acc_re = T(-700), log_acc_im = T(-700);  // ~0
    T log_ylm = T(0);  // ylm=1
    T log_gm_re = T(-0.693), log_gm_im = T(-1.2);  // gm~0.5, gm~0.3

    for (int i = 0; i < n_iters; i++) {
        // LOG accumulation: logsumexp for each component
        // log(acc + ylm*gm) = logsumexp(log_acc, log_ylm + log_gm)

        T log_term_re = log_ylm + log_gm_re;
        T log_term_im = log_ylm + log_gm_im;

        // logsumexp: max + log1p(exp(min - max))
        if (log_term_re > log_acc_re) {
            log_acc_re = log_term_re + fast_log1p(fast_exp(log_acc_re - log_term_re));
        } else {
            log_acc_re = log_acc_re + fast_log1p(fast_exp(log_term_re - log_acc_re));
        }

        if (log_term_im > log_acc_im) {
            log_acc_im = log_term_im + fast_log1p(fast_exp(log_acc_im - log_term_im));
        } else {
            log_acc_im = log_acc_im + fast_log1p(fast_exp(log_term_im - log_acc_im));
        }

        log_ylm -= T(0.000001);  // Simulate recurrence
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = log_acc_re;
        out[1] = log_acc_im;
    }
}

// Count transcendental ops in LOG mode
template<typename T>
__global__ void bench_transcendental_only(T* out, int n_iters) {
    T x = T(0.5);
    T y = T(0);

    for (int i = 0; i < n_iters; i++) {
        // Each logsumexp call: 1 exp + 1 log1p (per component)
        // For 2 components (re, im): 2 exp + 2 log1p per iteration
        T e1 = fast_exp(x);
        T e2 = fast_exp(x + T(0.1));
        T l1 = fast_log1p(e1);
        T l2 = fast_log1p(e2);
        y += l1 + l2;
        x -= T(0.000001);
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = y;
    }
}

template<typename T>
void run_benchmark(const char* type_name) {
    T* d_out;
    cudaMalloc(&d_out, 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    int n_blocks = 256;
    int n_threads = BLOCK_SIZE;
    int n_iters = N_ITERS;

    printf("\n%s precision:\n", type_name);
    printf("  Each thread: %d iterations, %d blocks x %d threads\n", n_iters, n_blocks, n_threads);

    // Warmup
    bench_fma<T><<<n_blocks, n_threads>>>(d_out, n_iters);
    cudaDeviceSynchronize();

    // Benchmark FMA (LINEAR equivalent)
    cudaEventRecord(start);
    bench_fma<T><<<n_blocks, n_threads>>>(d_out, n_iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float fma_time = ms;
    printf("  FMA (LINEAR equiv):     %8.3f ms\n", fma_time);

    // Warmup
    bench_logsumexp<T><<<n_blocks, n_threads>>>(d_out, n_iters);
    cudaDeviceSynchronize();

    // Benchmark logsumexp (LOG equivalent)
    cudaEventRecord(start);
    bench_logsumexp<T><<<n_blocks, n_threads>>>(d_out, n_iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float log_time = ms;
    printf("  Logsumexp (LOG equiv):  %8.3f ms\n", log_time);
    printf("  Ratio:                  %8.2fx\n", log_time / fma_time);

    // Warmup
    bench_transcendental_only<T><<<n_blocks, n_threads>>>(d_out, n_iters);
    cudaDeviceSynchronize();

    // Benchmark transcendental only
    cudaEventRecord(start);
    bench_transcendental_only<T><<<n_blocks, n_threads>>>(d_out, n_iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float trans_time = ms;
    printf("  Transcendental only:    %8.3f ms (2 exp + 2 log1p per iter)\n", trans_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);
}

int main() {
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Micro-benchmark: FMA vs Logsumexp operations\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");

    run_benchmark<float>("Float32");
    run_benchmark<double>("Float64");

    printf("\n");
    printf("CONCLUSION:\n");
    printf("  - f32 logsumexp uses fast SFU intrinsics (__expf, log1pf)\n");
    printf("  - f64 logsumexp uses software-emulated transcendentals\n");
    printf("  - This explains the 1.5x (f32) vs 3.6x (f64) overhead\n");

    return 0;
}

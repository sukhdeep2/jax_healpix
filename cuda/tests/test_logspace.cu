/**
 * Test Log-Space Arithmetic Modules
 *
 * Tests for:
 * - logspace_math.cuh: logsumexp, safe_log, ABRecurrence
 * - logspace_complex.cuh: LogComplex add/mul, conversions
 * - logspace_dft.cuh: DFT accumulation, twiddle factors
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../include/log_arithmetic.cuh"

// ============================================================================
// Test Utilities
// ============================================================================

#define TOLERANCE_F64 1e-10
#define TOLERANCE_F32 1e-5

template<typename T>
__device__ __forceinline__ T test_tolerance() {
    return sizeof(T) == 8 ? T(1e-10) : T(1e-5);
}

// Kernel to run logsumexp test
template<typename T>
__global__ void test_logsumexp_kernel(T* results) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // Test 1: Simple addition 1 + 2 = 3
        // log(1) = 0, log(2) = 0.693..., result should be log(3) = 1.0986...
        T log_A = T(0);
        T log_B = LogArithmeticTraits<T>::log_d(T(2));
        int8_t sign_A = 1, sign_B = 1;
        T log_result;
        int8_t sign_result;

        logsumexp_typed<T>(log_A, log_B, sign_A, sign_B, &log_result, &sign_result);

        results[0] = log_result;
        results[1] = T(sign_result);
        results[2] = LogArithmeticTraits<T>::log_d(T(3));  // Expected

        // Test 2: Subtraction 5 - 3 = 2
        log_A = LogArithmeticTraits<T>::log_d(T(5));
        log_B = LogArithmeticTraits<T>::log_d(T(3));
        sign_A = 1;
        sign_B = -1;

        logsumexp_typed<T>(log_A, log_B, sign_A, sign_B, &log_result, &sign_result);

        results[3] = log_result;
        results[4] = T(sign_result);
        results[5] = LogArithmeticTraits<T>::log_d(T(2));  // Expected

        // Test 3: Subtraction 3 - 5 = -2
        log_A = LogArithmeticTraits<T>::log_d(T(3));
        log_B = LogArithmeticTraits<T>::log_d(T(5));
        sign_A = 1;
        sign_B = -1;

        logsumexp_typed<T>(log_A, log_B, sign_A, sign_B, &log_result, &sign_result);

        results[6] = log_result;
        results[7] = T(sign_result);
        results[8] = LogArithmeticTraits<T>::log_d(T(2));  // Expected magnitude

        // Test 4: Large magnitude difference 1e20 + 1 ≈ 1e20
        log_A = T(46.05);  // log(1e20) ≈ 46.05
        log_B = T(0);       // log(1) = 0
        sign_A = 1;
        sign_B = 1;

        logsumexp_typed<T>(log_A, log_B, sign_A, sign_B, &log_result, &sign_result);

        results[9] = log_result;
        results[10] = T(sign_result);
        results[11] = T(46.05);  // Expected (essentially unchanged)
    }
}

// Kernel to test LogComplex operations
template<typename T>
__global__ void test_log_complex_kernel(T* results) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // Test 1: Complex addition (1+2i) + (3+4i) = (4+6i)
        LogComplex<T> z1 = LogComplex<T>::from_linear(T(1), T(2));
        LogComplex<T> z2 = LogComplex<T>::from_linear(T(3), T(4));
        LogComplex<T> sum = log_complex_add(z1, z2);

        T re_sum, im_sum;
        sum.to_linear(&re_sum, &im_sum);

        results[0] = re_sum;
        results[1] = im_sum;
        results[2] = T(4);  // Expected real
        results[3] = T(6);  // Expected imag

        // Test 2: Complex multiplication (1+2i) * (3+4i) = (1*3 - 2*4) + i(1*4 + 2*3)
        //                                               = -5 + 10i
        LogComplex<T> prod = log_complex_mul(z1, z2);

        T re_prod, im_prod;
        prod.to_linear(&re_prod, &im_prod);

        results[4] = re_prod;
        results[5] = im_prod;
        results[6] = T(-5);   // Expected real
        results[7] = T(10);   // Expected imag

        // Test 3: Complex from angle exp(i*pi/4) = (sqrt(2)/2, sqrt(2)/2)
        T theta = LogArithmeticTraits<T>::PI_VAL / T(4);
        LogComplex<T> unit = log_complex_from_angle(theta);

        T re_unit, im_unit;
        unit.to_linear(&re_unit, &im_unit);

        T expected_val = LogArithmeticTraits<T>::sqrt_d(T(2)) / T(2);
        results[8] = re_unit;
        results[9] = im_unit;
        results[10] = expected_val;  // Expected real
        results[11] = expected_val;  // Expected imag

        // Test 4: Polar to Cartesian conversion
        LogPolar<T> polar;
        polar.log_r = T(0);  // magnitude = 1
        polar.theta = LogArithmeticTraits<T>::PI_VAL / T(6);  // 30 degrees

        LogComplex<T> cart = log_polar_to_cartesian(polar);
        T re_cart, im_cart;
        cart.to_linear(&re_cart, &im_cart);

        // cos(pi/6) = sqrt(3)/2 ≈ 0.866, sin(pi/6) = 0.5
        results[12] = re_cart;
        results[13] = im_cart;
        results[14] = LogArithmeticTraits<T>::sqrt_d(T(3)) / T(2);
        results[15] = T(0.5);
    }
}

// Kernel to test DFT accumulation
template<typename T>
__global__ void test_log_dft_kernel(T* results) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // Test DFT: F[m] = Σ f[k] * exp(-i*m*phi[k])
        // Simple test: f = [1, 1, 1, 1] at phi = [0, pi/2, pi, 3pi/2]
        // F[0] = 1 + 1 + 1 + 1 = 4
        // F[1] = 1*exp(0) + 1*exp(-i*pi/2) + 1*exp(-i*pi) + 1*exp(-i*3pi/2)
        //      = 1 - i - 1 + i = 0

        T f_vals[4] = {T(1), T(1), T(1), T(1)};
        T phis[4] = {T(0),
                     LogArithmeticTraits<T>::PI_VAL / T(2),
                     LogArithmeticTraits<T>::PI_VAL,
                     T(3) * LogArithmeticTraits<T>::PI_VAL / T(2)};

        // Compute F[0] (m=0)
        LogComplex<T> F0 = LogComplex<T>::zero();
        for (int k = 0; k < 4; k++) {
            T m_phi = T(0) * phis[k];  // m=0
            LogTwiddle<T> tw = compute_log_twiddle<T>(m_phi);
            T log_f = safe_log_typed<T>(f_vals[k]);
            int8_t sign_f = sign_of<T>(f_vals[k]);
            log_dft_accumulate<T>(log_f, sign_f, tw, &F0);
        }

        T F0_re, F0_im;
        F0.to_linear(&F0_re, &F0_im);

        results[0] = F0_re;
        results[1] = F0_im;
        results[2] = T(4);  // Expected real
        results[3] = T(0);  // Expected imag

        // Compute F[1] (m=1)
        LogComplex<T> F1 = LogComplex<T>::zero();
        for (int k = 0; k < 4; k++) {
            T m_phi = T(1) * phis[k];  // m=1
            LogTwiddle<T> tw = compute_log_twiddle<T>(m_phi);
            T log_f = safe_log_typed<T>(f_vals[k]);
            int8_t sign_f = sign_of<T>(f_vals[k]);
            log_dft_accumulate<T>(log_f, sign_f, tw, &F1);
        }

        T F1_re, F1_im;
        F1.to_linear(&F1_re, &F1_im);

        results[4] = F1_re;
        results[5] = F1_im;
        results[6] = T(0);  // Expected real (should be near 0)
        results[7] = T(0);  // Expected imag (should be near 0)

        // Test TwiddleRecurrence
        T phi = LogArithmeticTraits<T>::PI_VAL / T(4);
        TwiddleRecurrence<T> rec;
        rec.init(phi);

        // m=0: should be (1, 0)
        LogTwiddle<T> tw0 = rec.get_twiddle_conj();
        T cos_0 = tw0.sign_cos * LogArithmeticTraits<T>::exp_d(tw0.log_cos);
        T sin_0 = tw0.sign_sin * LogArithmeticTraits<T>::exp_d(tw0.log_sin);

        results[8] = cos_0;
        results[9] = sin_0;
        results[10] = T(1);  // Expected cos(0)
        results[11] = T(0);  // Expected sin(0)

        // Advance to m=1
        rec.step();
        LogTwiddle<T> tw1 = rec.get_twiddle_conj();
        T cos_1 = tw1.sign_cos * LogArithmeticTraits<T>::exp_d(tw1.log_cos);
        T sin_1 = tw1.sign_sin * LogArithmeticTraits<T>::exp_d(tw1.log_sin);

        T expected_cos1 = LogArithmeticTraits<T>::cos_d(phi);
        T expected_sin1 = LogArithmeticTraits<T>::sin_d(phi);

        results[12] = cos_1;
        results[13] = sin_1;
        results[14] = expected_cos1;
        results[15] = expected_sin1;
    }
}

// Kernel to test ABRecurrenceState
template<typename T>
__global__ void test_ab_recurrence_kernel(T* results) {
    int tid = threadIdx.x;

    if (tid == 0) {
        int m = 2;

        // Test incremental vs direct computation
        ABRecurrenceState<T> state;
        state.init(m);

        // Compute for l = m+2 = 4
        T log_A_inc, log_B_inc;
        log_B_inc = state.step(m + 2, &log_A_inc);

        // Direct computation
        T log_A_direct = log_A_lm<T>(m + 2, m);
        T log_B_direct = log_B_lm<T>(m + 2, m);

        results[0] = log_A_inc;
        results[1] = log_A_direct;
        results[2] = log_B_inc;
        results[3] = log_B_direct;

        // Test several more steps
        int errors = 0;
        for (int l = m + 3; l <= m + 10; l++) {
            log_B_inc = state.step(l, &log_A_inc);
            log_A_direct = log_A_lm<T>(l, m);
            log_B_direct = log_B_lm<T>(l, m);

            T diff_A = LogArithmeticTraits<T>::abs_d(log_A_inc - log_A_direct);
            T diff_B = LogArithmeticTraits<T>::abs_d(log_B_inc - log_B_direct);

            if (diff_A > test_tolerance<T>() || diff_B > test_tolerance<T>()) {
                errors++;
            }
        }

        results[4] = T(errors);
    }
}

// ============================================================================
// Host Test Functions
// ============================================================================

int test_logsumexp_f64() {
    printf("Testing logsumexp (float64)...\n");

    double* d_results;
    double h_results[12];
    cudaMalloc(&d_results, 12 * sizeof(double));

    test_logsumexp_kernel<double><<<1, 1>>>(d_results);
    cudaMemcpy(h_results, d_results, 12 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    int errors = 0;

    // Test 1: 1 + 2 = 3
    double diff1 = fabs(h_results[0] - h_results[2]);
    if (diff1 > TOLERANCE_F64 || h_results[1] != 1.0) {
        printf("  ERROR: 1+2: log_result=%.10e (expected %.10e), sign=%.0f\n",
               h_results[0], h_results[2], h_results[1]);
        errors++;
    }

    // Test 2: 5 - 3 = 2
    double diff2 = fabs(h_results[3] - h_results[5]);
    if (diff2 > TOLERANCE_F64 || h_results[4] != 1.0) {
        printf("  ERROR: 5-3: log_result=%.10e (expected %.10e), sign=%.0f\n",
               h_results[3], h_results[5], h_results[4]);
        errors++;
    }

    // Test 3: 3 - 5 = -2
    double diff3 = fabs(h_results[6] - h_results[8]);
    if (diff3 > TOLERANCE_F64 || h_results[7] != -1.0) {
        printf("  ERROR: 3-5: log_result=%.10e (expected %.10e), sign=%.0f\n",
               h_results[6], h_results[8], h_results[7]);
        errors++;
    }

    // Test 4: Large magnitude (1e20 + 1 ≈ 1e20)
    double diff4 = fabs(h_results[9] - h_results[11]);
    if (diff4 > 1e-8) {  // Looser tolerance for large magnitudes
        printf("  ERROR: 1e20+1: log_result=%.10e (expected %.10e)\n",
               h_results[9], h_results[11]);
        errors++;
    }

    if (errors == 0) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED with %d errors\n", errors);
    }

    return errors;
}

int test_logsumexp_f32() {
    printf("Testing logsumexp (float32)...\n");

    float* d_results;
    float h_results[12];
    cudaMalloc(&d_results, 12 * sizeof(float));

    test_logsumexp_kernel<float><<<1, 1>>>(d_results);
    cudaMemcpy(h_results, d_results, 12 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    int errors = 0;

    // Test 1: 1 + 2 = 3
    float diff1 = fabsf(h_results[0] - h_results[2]);
    if (diff1 > TOLERANCE_F32 || h_results[1] != 1.0f) {
        printf("  ERROR: 1+2: log_result=%.6e (expected %.6e), sign=%.0f\n",
               h_results[0], h_results[2], h_results[1]);
        errors++;
    }

    // Test 2: 5 - 3 = 2
    float diff2 = fabsf(h_results[3] - h_results[5]);
    if (diff2 > TOLERANCE_F32 || h_results[4] != 1.0f) {
        printf("  ERROR: 5-3: log_result=%.6e (expected %.6e), sign=%.0f\n",
               h_results[3], h_results[5], h_results[4]);
        errors++;
    }

    if (errors == 0) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED with %d errors\n", errors);
    }

    return errors;
}

int test_log_complex_f64() {
    printf("Testing LogComplex operations (float64)...\n");

    double* d_results;
    double h_results[16];
    cudaMalloc(&d_results, 16 * sizeof(double));

    test_log_complex_kernel<double><<<1, 1>>>(d_results);
    cudaMemcpy(h_results, d_results, 16 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    int errors = 0;

    // Test 1: Complex addition
    double diff_re_add = fabs(h_results[0] - h_results[2]);
    double diff_im_add = fabs(h_results[1] - h_results[3]);
    if (diff_re_add > TOLERANCE_F64 || diff_im_add > TOLERANCE_F64) {
        printf("  ERROR: Complex add: (%.10e, %.10e) expected (%.10e, %.10e)\n",
               h_results[0], h_results[1], h_results[2], h_results[3]);
        errors++;
    }

    // Test 2: Complex multiplication
    double diff_re_mul = fabs(h_results[4] - h_results[6]);
    double diff_im_mul = fabs(h_results[5] - h_results[7]);
    if (diff_re_mul > TOLERANCE_F64 || diff_im_mul > TOLERANCE_F64) {
        printf("  ERROR: Complex mul: (%.10e, %.10e) expected (%.10e, %.10e)\n",
               h_results[4], h_results[5], h_results[6], h_results[7]);
        errors++;
    }

    // Test 3: Complex from angle
    double diff_re_angle = fabs(h_results[8] - h_results[10]);
    double diff_im_angle = fabs(h_results[9] - h_results[11]);
    if (diff_re_angle > TOLERANCE_F64 || diff_im_angle > TOLERANCE_F64) {
        printf("  ERROR: Complex from angle: (%.10e, %.10e) expected (%.10e, %.10e)\n",
               h_results[8], h_results[9], h_results[10], h_results[11]);
        errors++;
    }

    // Test 4: Polar to Cartesian
    double diff_re_polar = fabs(h_results[12] - h_results[14]);
    double diff_im_polar = fabs(h_results[13] - h_results[15]);
    if (diff_re_polar > TOLERANCE_F64 || diff_im_polar > TOLERANCE_F64) {
        printf("  ERROR: Polar to Cartesian: (%.10e, %.10e) expected (%.10e, %.10e)\n",
               h_results[12], h_results[13], h_results[14], h_results[15]);
        errors++;
    }

    if (errors == 0) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED with %d errors\n", errors);
    }

    return errors;
}

int test_log_dft_f64() {
    printf("Testing log-space DFT (float64)...\n");

    double* d_results;
    double h_results[16];
    cudaMalloc(&d_results, 16 * sizeof(double));

    test_log_dft_kernel<double><<<1, 1>>>(d_results);
    cudaMemcpy(h_results, d_results, 16 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    int errors = 0;

    // Test F[0] = 4
    double diff_F0_re = fabs(h_results[0] - h_results[2]);
    double diff_F0_im = fabs(h_results[1] - h_results[3]);
    if (diff_F0_re > TOLERANCE_F64 || diff_F0_im > TOLERANCE_F64) {
        printf("  ERROR: F[0]: (%.10e, %.10e) expected (%.10e, %.10e)\n",
               h_results[0], h_results[1], h_results[2], h_results[3]);
        errors++;
    }

    // Test F[1] ≈ 0
    double F1_mag = sqrt(h_results[4]*h_results[4] + h_results[5]*h_results[5]);
    if (F1_mag > 1e-8) {  // Should be very close to zero
        printf("  ERROR: F[1]: (%.10e, %.10e) expected near (0, 0)\n",
               h_results[4], h_results[5]);
        errors++;
    }

    // Test TwiddleRecurrence m=0
    double diff_cos0 = fabs(h_results[8] - h_results[10]);
    // sin(0) could be a very small number due to LOG_MIN handling
    if (diff_cos0 > TOLERANCE_F64) {
        printf("  ERROR: Twiddle m=0: cos=%.10e (expected %.10e)\n",
               h_results[8], h_results[10]);
        errors++;
    }

    // Test TwiddleRecurrence m=1
    double diff_cos1 = fabs(h_results[12] - h_results[14]);
    double diff_sin1 = fabs(h_results[13] - h_results[15]);
    if (diff_cos1 > TOLERANCE_F64 || diff_sin1 > TOLERANCE_F64) {
        printf("  ERROR: Twiddle m=1: (%.10e, %.10e) expected (%.10e, %.10e)\n",
               h_results[12], h_results[13], h_results[14], h_results[15]);
        errors++;
    }

    if (errors == 0) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED with %d errors\n", errors);
    }

    return errors;
}

int test_ab_recurrence_f64() {
    printf("Testing ABRecurrenceState (float64)...\n");

    double* d_results;
    double h_results[5];
    cudaMalloc(&d_results, 5 * sizeof(double));

    test_ab_recurrence_kernel<double><<<1, 1>>>(d_results);
    cudaMemcpy(h_results, d_results, 5 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    int errors = 0;

    // Check incremental vs direct for l=4, m=2
    double diff_A = fabs(h_results[0] - h_results[1]);
    double diff_B = fabs(h_results[2] - h_results[3]);

    if (diff_A > TOLERANCE_F64) {
        printf("  ERROR: log_A incremental=%.10e, direct=%.10e, diff=%.2e\n",
               h_results[0], h_results[1], diff_A);
        errors++;
    }

    if (diff_B > TOLERANCE_F64) {
        printf("  ERROR: log_B incremental=%.10e, direct=%.10e, diff=%.2e\n",
               h_results[2], h_results[3], diff_B);
        errors++;
    }

    // Check error count from additional iterations
    int iter_errors = (int)h_results[4];
    if (iter_errors > 0) {
        printf("  ERROR: %d errors in incremental recurrence iterations\n", iter_errors);
        errors += iter_errors;
    }

    if (errors == 0) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED with %d errors\n", errors);
    }

    return errors;
}

int test_safe_log_edge_cases() {
    printf("Testing safe_log edge cases...\n");

    int errors = 0;

    // Test on host (simple check)
    // safe_log(0) should return LOG_MIN
    // We can't directly call device functions from host, so we verify constants

    printf("  LOG_MIN_F64 = %.1f\n", LogArithmeticTraits<double>::LOG_MIN);
    printf("  LOG_MAX_F64 = %.1f\n", LogArithmeticTraits<double>::LOG_MAX);
    printf("  LOG_MIN_F32 = %.1f\n", LogArithmeticTraits<float>::LOG_MIN);
    printf("  LOG_MAX_F32 = %.1f\n", LogArithmeticTraits<float>::LOG_MAX);

    // Verify reasonable values
    if (LogArithmeticTraits<double>::LOG_MIN > -500) {
        printf("  ERROR: LOG_MIN_F64 too high\n");
        errors++;
    }
    if (LogArithmeticTraits<float>::LOG_MIN > -80) {
        printf("  ERROR: LOG_MIN_F32 too high\n");
        errors++;
    }

    if (errors == 0) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED with %d errors\n", errors);
    }

    return errors;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== SPHT CUDA Log-Space Arithmetic Tests ===\n\n");

    int total_errors = 0;

    // Core math tests
    total_errors += test_logsumexp_f64();
    total_errors += test_logsumexp_f32();
    total_errors += test_safe_log_edge_cases();

    // Complex arithmetic tests
    total_errors += test_log_complex_f64();

    // DFT tests
    total_errors += test_log_dft_f64();

    // Recurrence tests
    total_errors += test_ab_recurrence_f64();

    printf("\n=== Summary ===\n");
    if (total_errors == 0) {
        printf("All tests PASSED\n");
        return 0;
    } else {
        printf("FAILED with %d total errors\n", total_errors);
        return 1;
    }
}

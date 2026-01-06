/**
 * alm2cl.cu - CUDA implementation of power spectrum computation
 *
 * Computes angular power spectra C_l from spherical harmonic coefficients a_lm.
 * Supports auto-spectra and cross-spectra for multiple fields.
 *
 * Formula: C_l = (1/(2l+1)) * Σ_m |a_{l,m}|² for auto-spectra
 *          C_l = (1/(2l+1)) * Re(Σ_m a1_{l,m} * conj(a2_{l,m})) for cross-spectra
 *
 * Note: For m>0, contributions are doubled since we don't store m<0 coefficients.
 */

#include <cuda_runtime.h>
#include <cstdio>

// =============================================================================
// Auto-spectrum kernel: C_l from single alm array
// =============================================================================

template<typename T>
__global__ void alm2cl_auto_kernel(
    int l_max,
    int n_fields,
    const T* __restrict__ alm_real,    // [n_fields, l_max+1, l_max+1]
    const T* __restrict__ alm_imag,
    T* __restrict__ cl_out             // [n_fields, l_max+1]
) {
    // Each thread handles one (field, l) pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_l = l_max + 1;
    int total = n_fields * n_l;

    if (idx >= total) return;

    int field_idx = idx / n_l;
    int l = idx % n_l;

    // Compute sum of |a_lm|^2 for m = 0..l
    T sum = T(0);

    int base_idx = field_idx * n_l * n_l + l * n_l;

    // m = 0 term (count once)
    {
        T re = alm_real[base_idx];
        T im = alm_imag[base_idx];
        sum += re * re + im * im;
    }

    // m > 0 terms (multiply by 2 since we don't store m < 0)
    for (int m = 1; m <= l; m++) {
        T re = alm_real[base_idx + m];
        T im = alm_imag[base_idx + m];
        sum += T(2) * (re * re + im * im);
    }

    // Normalize by 2l+1
    cl_out[field_idx * n_l + l] = sum / T(2 * l + 1);
}


// =============================================================================
// Cross-spectrum kernel: C_l from two alm arrays
// =============================================================================

template<typename T>
__global__ void alm2cl_cross_kernel(
    int l_max,
    int n_pairs,
    const T* __restrict__ alm1_real,   // [n_pairs, l_max+1, l_max+1]
    const T* __restrict__ alm1_imag,
    const T* __restrict__ alm2_real,
    const T* __restrict__ alm2_imag,
    T* __restrict__ cl_out             // [n_pairs, l_max+1]
) {
    // Each thread handles one (pair, l) combination
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_l = l_max + 1;
    int total = n_pairs * n_l;

    if (idx >= total) return;

    int pair_idx = idx / n_l;
    int l = idx % n_l;

    // Compute Re(sum of a1_lm * conj(a2_lm)) for m = 0..l
    T sum = T(0);

    int base_idx = pair_idx * n_l * n_l + l * n_l;

    // m = 0 term (count once)
    // Re(a1 * conj(a2)) = re1*re2 + im1*im2
    {
        T re1 = alm1_real[base_idx];
        T im1 = alm1_imag[base_idx];
        T re2 = alm2_real[base_idx];
        T im2 = alm2_imag[base_idx];
        sum += re1 * re2 + im1 * im2;
    }

    // m > 0 terms (multiply by 2 since we don't store m < 0)
    for (int m = 1; m <= l; m++) {
        T re1 = alm1_real[base_idx + m];
        T im1 = alm1_imag[base_idx + m];
        T re2 = alm2_real[base_idx + m];
        T im2 = alm2_imag[base_idx + m];
        sum += T(2) * (re1 * re2 + im1 * im2);
    }

    // Normalize by 2l+1
    cl_out[pair_idx * n_l + l] = sum / T(2 * l + 1);
}


// =============================================================================
// Batch auto and cross spectra kernel
// Computes all n_fields*(n_fields+1)/2 spectra in one launch
// =============================================================================

template<typename T>
__global__ void alm2cl_all_pairs_kernel(
    int l_max,
    int n_fields,
    const T* __restrict__ alm_real,    // [n_fields, l_max+1, l_max+1]
    const T* __restrict__ alm_imag,
    T* __restrict__ cl_out             // [n_pairs, l_max+1] where n_pairs = n_fields*(n_fields+1)/2
) {
    // Each thread handles one (pair, l) combination
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_l = l_max + 1;
    int n_pairs = n_fields * (n_fields + 1) / 2;
    int total = n_pairs * n_l;

    if (idx >= total) return;

    int pair_idx = idx / n_l;
    int l = idx % n_l;

    // Convert linear pair_idx to (field1, field2) where field1 <= field2
    // Pairs are ordered as: (0,0), (0,1), (0,2), ..., (1,1), (1,2), ..., (n-1, n-1)
    int field1 = 0;
    int count = 0;
    for (int f = 0; f < n_fields; f++) {
        int pairs_from_f = n_fields - f;
        if (count + pairs_from_f > pair_idx) {
            field1 = f;
            break;
        }
        count += pairs_from_f;
    }
    int field2 = field1 + (pair_idx - count);

    // Base indices for the two fields
    int base1 = field1 * n_l * n_l + l * n_l;
    int base2 = field2 * n_l * n_l + l * n_l;

    // Compute cross-spectrum (or auto-spectrum if field1 == field2)
    T sum = T(0);

    // m = 0 term
    {
        T re1 = alm_real[base1];
        T im1 = alm_imag[base1];
        T re2 = alm_real[base2];
        T im2 = alm_imag[base2];
        sum += re1 * re2 + im1 * im2;
    }

    // m > 0 terms (multiply by 2)
    for (int m = 1; m <= l; m++) {
        T re1 = alm_real[base1 + m];
        T im1 = alm_imag[base1 + m];
        T re2 = alm_real[base2 + m];
        T im2 = alm_imag[base2 + m];
        sum += T(2) * (re1 * re2 + im1 * im2);
    }

    // Normalize by 2l+1
    cl_out[pair_idx * n_l + l] = sum / T(2 * l + 1);
}


// =============================================================================
// C API - Auto-spectrum
// =============================================================================

extern "C"
void alm2cl_cuda_auto_f64(
    int l_max,
    int n_fields,
    const double* d_alm_real,
    const double* d_alm_imag,
    double* d_cl_out
) {
    int n_l = l_max + 1;
    int total = n_fields * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    alm2cl_auto_kernel<double><<<blocks, threads>>>(
        l_max, n_fields, d_alm_real, d_alm_imag, d_cl_out
    );
    cudaDeviceSynchronize();
}

extern "C"
void alm2cl_cuda_auto_f32(
    int l_max,
    int n_fields,
    const float* d_alm_real,
    const float* d_alm_imag,
    float* d_cl_out
) {
    int n_l = l_max + 1;
    int total = n_fields * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    alm2cl_auto_kernel<float><<<blocks, threads>>>(
        l_max, n_fields, d_alm_real, d_alm_imag, d_cl_out
    );
    cudaDeviceSynchronize();
}


// =============================================================================
// C API - Cross-spectrum (paired arrays)
// =============================================================================

extern "C"
void alm2cl_cuda_cross_f64(
    int l_max,
    int n_pairs,
    const double* d_alm1_real,
    const double* d_alm1_imag,
    const double* d_alm2_real,
    const double* d_alm2_imag,
    double* d_cl_out
) {
    int n_l = l_max + 1;
    int total = n_pairs * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    alm2cl_cross_kernel<double><<<blocks, threads>>>(
        l_max, n_pairs, d_alm1_real, d_alm1_imag, d_alm2_real, d_alm2_imag, d_cl_out
    );
    cudaDeviceSynchronize();
}

extern "C"
void alm2cl_cuda_cross_f32(
    int l_max,
    int n_pairs,
    const float* d_alm1_real,
    const float* d_alm1_imag,
    const float* d_alm2_real,
    const float* d_alm2_imag,
    float* d_cl_out
) {
    int n_l = l_max + 1;
    int total = n_pairs * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    alm2cl_cross_kernel<float><<<blocks, threads>>>(
        l_max, n_pairs, d_alm1_real, d_alm1_imag, d_alm2_real, d_alm2_imag, d_cl_out
    );
    cudaDeviceSynchronize();
}


// =============================================================================
// C API - All pairs (auto + cross spectra)
// =============================================================================

extern "C"
void alm2cl_cuda_all_pairs_f64(
    int l_max,
    int n_fields,
    const double* d_alm_real,
    const double* d_alm_imag,
    double* d_cl_out
) {
    int n_l = l_max + 1;
    int n_pairs = n_fields * (n_fields + 1) / 2;
    int total = n_pairs * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    alm2cl_all_pairs_kernel<double><<<blocks, threads>>>(
        l_max, n_fields, d_alm_real, d_alm_imag, d_cl_out
    );
    cudaDeviceSynchronize();
}

extern "C"
void alm2cl_cuda_all_pairs_f32(
    int l_max,
    int n_fields,
    const float* d_alm_real,
    const float* d_alm_imag,
    float* d_cl_out
) {
    int n_l = l_max + 1;
    int n_pairs = n_fields * (n_fields + 1) / 2;
    int total = n_pairs * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    alm2cl_all_pairs_kernel<float><<<blocks, threads>>>(
        l_max, n_fields, d_alm_real, d_alm_imag, d_cl_out
    );
    cudaDeviceSynchronize();
}

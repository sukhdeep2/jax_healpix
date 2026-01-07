/**
 * Synalm: Generate Random ALM from Power Spectrum
 *
 * Generates Gaussian-distributed spherical harmonic coefficients from Cl.
 *
 * HEALPix Formula:
 *   a_lm = sqrt(C_l) * (g_r + i*g_i) / sqrt(2)   for m > 0
 *   a_l0 = sqrt(C_l) * g_r                        for m = 0
 * where g_r, g_i ~ N(0,1)
 *
 * Log-Space Implementation:
 *   log_scale = 0.5 * log(C_l)
 *   a_lm = exp(log_scale) * gaussian_sample
 *
 * Reference: healpy.synalm and IMPLEMENTATION_PLAN.md
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// RNG State Initialization
// ============================================================================

__global__ void init_curand_states(
    curandState* states,
    unsigned long long seed,
    int n_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// ============================================================================
// Synalm Kernel
// ============================================================================

/**
 * Generate random alm from Cl power spectrum
 *
 * Each thread handles one (field, l, m) triplet.
 * Uses log-space for Cl values that span many orders of magnitude.
 */
template<typename T>
__global__ void synalm_kernel(
    int l_max,
    int n_fields,
    const T* __restrict__ d_cl,           // [n_fields, l_max+1]
    const T* __restrict__ d_log_cl,       // [n_fields, l_max+1] - log(Cl) for accuracy
    curandState* __restrict__ rng_states, // Per-thread RNG
    T* __restrict__ d_alm_real,           // [n_fields, l_max+1, l_max+1]
    T* __restrict__ d_alm_imag,
    bool use_log_space                     // Runtime flag for log-space path
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;

    if (idx >= total) return;

    int field = idx / (n_l * n_l);
    int lm_idx = idx % (n_l * n_l);
    int l = lm_idx / n_l;
    int m = lm_idx % n_l;

    if (m > l) return;  // Upper triangular only

    // Get Cl value
    int cl_idx = field * n_l + l;
    T cl_val = d_cl[cl_idx];

    // Generate Gaussian samples
    curandState local_state = rng_states[idx];
    T g_r = curand_normal(&local_state);
    T g_i = (m > 0) ? curand_normal(&local_state) : T(0);
    rng_states[idx] = local_state;

    // Compute scale factor
    T scale;
    if (use_log_space && cl_val > T(0)) {
        // Log-space path for very small/large Cl
        T log_cl = d_log_cl[cl_idx];
        T log_scale = T(0.5) * log_cl;  // sqrt in log space
        scale = exp(log_scale);
    } else if (cl_val > T(0)) {
        scale = sqrt(cl_val);
    } else {
        scale = T(0);
    }

    // Apply scaling
    T re, im;
    if (m == 0) {
        re = scale * g_r;
        im = T(0);
    } else {
        T inv_sqrt2 = T(0.7071067811865476);
        re = scale * g_r * inv_sqrt2;
        im = scale * g_i * inv_sqrt2;
    }

    // Store result
    int out_idx = field * n_l * n_l + l * n_l + m;
    d_alm_real[out_idx] = re;
    d_alm_imag[out_idx] = im;
}

/**
 * Synalm for correlated fields using Cholesky decomposition
 *
 * For correlated fields (e.g., T, E, B with cross-correlations):
 *   C = L * L^T  (Cholesky decomposition)
 *   alm = L * g   (g is vector of independent Gaussians)
 *
 * The Cholesky matrix L should be precomputed on the host.
 */
template<typename T>
__global__ void synalm_correlated_kernel(
    int l_max,
    int n_fields,                          // e.g., 3 for T, E, B
    const T* __restrict__ d_cholesky,      // [l_max+1, n_fields, n_fields] - L matrix (lower triangular)
    curandState* __restrict__ rng_states,
    T* __restrict__ d_alm_real,            // [n_fields, l_max+1, l_max+1]
    T* __restrict__ d_alm_imag
) {
    // Each block handles one (l, m) pair for all fields
    int lm_idx = blockIdx.x;
    int n_l = l_max + 1;

    if (lm_idx >= n_l * n_l) return;

    int l = lm_idx / n_l;
    int m = lm_idx % n_l;
    if (m > l) return;

    // Thread idx within block handles different parts of the correlation
    int tid = threadIdx.x;

    // Shared memory for Gaussian samples (use char[] for template compatibility)
    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    T* g_r = smem;                          // [n_fields]
    T* g_i = smem + n_fields;               // [n_fields]

    // Generate independent Gaussian samples for each field
    if (tid < n_fields) {
        int rng_idx = lm_idx * n_fields + tid;
        curandState local_state = rng_states[rng_idx];
        g_r[tid] = curand_normal(&local_state);
        g_i[tid] = (m > 0) ? curand_normal(&local_state) : T(0);
        rng_states[rng_idx] = local_state;
    }
    __syncthreads();

    // Apply Cholesky: alm[i] = sum_j L[l,i,j] * g[j]
    if (tid < n_fields) {
        T sum_r = T(0);
        T sum_i = T(0);

        // L is lower triangular: L[i,j] = 0 for j > i
        for (int j = 0; j <= tid; j++) {
            int L_idx = l * n_fields * n_fields + tid * n_fields + j;
            T L_ij = d_cholesky[L_idx];
            sum_r += L_ij * g_r[j];
            sum_i += L_ij * g_i[j];
        }

        // Apply 1/sqrt(2) factor for m > 0
        if (m > 0) {
            T inv_sqrt2 = T(0.7071067811865476);
            sum_r *= inv_sqrt2;
            sum_i *= inv_sqrt2;
        } else {
            sum_i = T(0);
        }

        // Store result
        int out_idx = tid * n_l * n_l + l * n_l + m;
        d_alm_real[out_idx] = sum_r;
        d_alm_imag[out_idx] = sum_i;
    }
}

// ============================================================================
// Host-side Cholesky Decomposition
// ============================================================================

/**
 * Compute Cholesky decomposition of Cl covariance matrix
 *
 * Input: cl_matrix[l, i, j] = cross-power between fields i and j at multipole l
 * Output: cholesky[l, i, j] = L[i,j] where C = L * L^T
 *
 * This is called on the host before the correlated synalm kernel.
 */
template<typename T>
void compute_cholesky_host(
    int l_max,
    int n_fields,
    const T* cl_matrix,    // [l_max+1, n_fields, n_fields]
    T* cholesky            // [l_max+1, n_fields, n_fields]
) {
    int n_l = l_max + 1;

    for (int l = 0; l < n_l; l++) {
        // Get pointer to this l's matrices
        const T* C = cl_matrix + l * n_fields * n_fields;
        T* L = cholesky + l * n_fields * n_fields;

        // Initialize L to zero
        for (int i = 0; i < n_fields * n_fields; i++) {
            L[i] = T(0);
        }

        // Cholesky decomposition: L * L^T = C
        for (int i = 0; i < n_fields; i++) {
            for (int j = 0; j <= i; j++) {
                T sum = C[i * n_fields + j];

                for (int k = 0; k < j; k++) {
                    sum -= L[i * n_fields + k] * L[j * n_fields + k];
                }

                if (i == j) {
                    if (sum > T(0)) {
                        L[i * n_fields + j] = sqrt(sum);
                    } else {
                        L[i * n_fields + j] = T(0);  // Handle non-positive-definite
                    }
                } else {
                    T L_jj = L[j * n_fields + j];
                    if (L_jj > T(1e-30)) {
                        L[i * n_fields + j] = sum / L_jj;
                    } else {
                        L[i * n_fields + j] = T(0);
                    }
                }
            }
        }
    }
}

// ============================================================================
// C API Entry Points
// ============================================================================

extern "C" {

// Initialize RNG states
void synalm_init_rng_cuda(curandState** d_states, int n_states, unsigned long long seed) {
    cudaMalloc(d_states, n_states * sizeof(curandState));

    int threads = 256;
    int blocks = (n_states + threads - 1) / threads;
    init_curand_states<<<blocks, threads>>>(*d_states, seed, n_states);
}

// Free RNG states
void synalm_free_rng_cuda(curandState* d_states) {
    if (d_states) cudaFree(d_states);
}

// Synalm - f64
void synalm_cuda_f64(
    int l_max, int n_fields,
    const double* d_cl, const double* d_log_cl,
    curandState* rng_states,
    double* d_alm_real, double* d_alm_imag,
    bool use_log_space
) {
    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    synalm_kernel<double><<<blocks, threads>>>(
        l_max, n_fields, d_cl, d_log_cl,
        rng_states, d_alm_real, d_alm_imag,
        use_log_space
    );
}

// Synalm - f32
void synalm_cuda_f32(
    int l_max, int n_fields,
    const float* d_cl, const float* d_log_cl,
    curandState* rng_states,
    float* d_alm_real, float* d_alm_imag,
    bool use_log_space
) {
    int n_l = l_max + 1;
    int total = n_fields * n_l * n_l;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    synalm_kernel<float><<<blocks, threads>>>(
        l_max, n_fields, d_cl, d_log_cl,
        rng_states, d_alm_real, d_alm_imag,
        use_log_space
    );
}

// Synalm correlated - f64
void synalm_correlated_cuda_f64(
    int l_max, int n_fields,
    const double* d_cholesky,
    curandState* rng_states,
    double* d_alm_real, double* d_alm_imag
) {
    int n_l = l_max + 1;
    int n_lm = n_l * n_l;

    // Each block handles one (l,m), needs n_fields threads
    int threads = ((n_fields + 31) / 32) * 32;  // Round up to warp size
    int blocks = n_lm;
    size_t smem_size = 2 * n_fields * sizeof(double);

    synalm_correlated_kernel<double><<<blocks, threads, smem_size>>>(
        l_max, n_fields, d_cholesky,
        rng_states, d_alm_real, d_alm_imag
    );
}

// Synalm correlated - f32
void synalm_correlated_cuda_f32(
    int l_max, int n_fields,
    const float* d_cholesky,
    curandState* rng_states,
    float* d_alm_real, float* d_alm_imag
) {
    int n_l = l_max + 1;
    int n_lm = n_l * n_l;

    int threads = ((n_fields + 31) / 32) * 32;
    int blocks = n_lm;
    size_t smem_size = 2 * n_fields * sizeof(float);

    synalm_correlated_kernel<float><<<blocks, threads, smem_size>>>(
        l_max, n_fields, d_cholesky,
        rng_states, d_alm_real, d_alm_imag
    );
}

// Host Cholesky - f64
void compute_cholesky_f64(int l_max, int n_fields,
                           const double* cl_matrix, double* cholesky) {
    compute_cholesky_host<double>(l_max, n_fields, cl_matrix, cholesky);
}

// Host Cholesky - f32
void compute_cholesky_f32(int l_max, int n_fields,
                           const float* cl_matrix, float* cholesky) {
    compute_cholesky_host<float>(l_max, n_fields, cl_matrix, cholesky);
}

} // extern "C"

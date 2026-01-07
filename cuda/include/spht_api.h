#ifndef SPHT_API_H
#define SPHT_API_H

#include "spht_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * SPHT CUDA Context - holds precomputed data and buffers
 */
typedef struct {
    int nside;
    int l_max;
    ring_geometry_t* ring_geom;
    // Preallocated buffers can be added here
} spht_context_t;

/**
 * Create a new SPHT context
 *
 * @param nside HEALPix nside parameter
 * @param l_max Maximum l value (default: 3*nside if 0)
 * @return Pointer to context, or NULL on error
 */
spht_context_t* spht_create_context(int nside, int l_max);

/**
 * Destroy SPHT context and free resources
 */
void spht_destroy_context(spht_context_t* ctx);

/**
 * Perform map2alm transform (analysis)
 *
 * @param ctx SPHT context
 * @param n_maps Number of maps
 * @param map_in Input maps on device [n_maps, 4*nside-1, 4*nside]
 * @param alm_out Output alm on device [n_maps, l_max+1, l_max+1]
 * @return 0 on success, non-zero on error
 */
int spht_map2alm(spht_context_t* ctx, int n_maps,
                 const real_t* map_in, complex_t* alm_out);

/**
 * Perform alm2map transform (synthesis)
 *
 * @param ctx SPHT context
 * @param n_maps Number of maps
 * @param alm_in Input alm on device [n_maps, l_max+1, l_max+1]
 * @param map_out Output maps on device [n_maps, 4*nside-1, 4*nside]
 * @return 0 on success, non-zero on error
 */
int spht_alm2map(spht_context_t* ctx, int n_maps,
                 const complex_t* alm_in, real_t* map_out);

/**
 * Standalone map2alm v6 - LINEAR mode (no context needed)
 */
void map2alm_cuda_v6(int nside, int l_max, int n_maps,
                      const real_t* map_in, complex_t* alm_out);

/**
 * Standalone alm2map v6 - LINEAR mode (no context needed)
 */
void alm2map_cuda_v6(int nside, int l_max, int n_maps,
                      const complex_t* alm_in, real_t* map_out);

/**
 * V6 precision variants - LINEAR mode
 */
void map2alm_cuda_v6_f64_f64(int nside, int l_max, int n_maps,
                              const double* map_in, double* alm_re, double* alm_im);
void map2alm_cuda_v6_f64_f32(int nside, int l_max, int n_maps,
                              const double* map_in, double* alm_re, double* alm_im);
void map2alm_cuda_v6_f32_f64(int nside, int l_max, int n_maps,
                              const float* map_in, float* alm_re, float* alm_im);
void map2alm_cuda_v6_f32_f32(int nside, int l_max, int n_maps,
                              const float* map_in, float* alm_re, float* alm_im);

void alm2map_cuda_v6_f64_f64(int nside, int l_max, int n_maps,
                              const double* alm_re, const double* alm_im, double* map_out);
void alm2map_cuda_v6_f64_f32(int nside, int l_max, int n_maps,
                              const double* alm_re, const double* alm_im, double* map_out);
void alm2map_cuda_v6_f32_f64(int nside, int l_max, int n_maps,
                              const float* alm_re, const float* alm_im, float* map_out);
void alm2map_cuda_v6_f32_f32(int nside, int l_max, int n_maps,
                              const float* alm_re, const float* alm_im, float* map_out);

/**
 * V6 precision variants - LOG mode (logsumexp accumulation)
 *
 * LOG mode uses log-space arithmetic for numerical stability.
 * Required for bf16 precision, optional for f32/f64.
 * ~5-10x slower than LINEAR mode but handles extreme dynamic range.
 */
void map2alm_cuda_v6_log_f64_f64(int nside, int l_max, int n_maps,
                                  const double* map_in, double* alm_re, double* alm_im);
void map2alm_cuda_v6_log_f64_f32(int nside, int l_max, int n_maps,
                                  const double* map_in, double* alm_re, double* alm_im);
void map2alm_cuda_v6_log_f32_f64(int nside, int l_max, int n_maps,
                                  const float* map_in, float* alm_re, float* alm_im);
void map2alm_cuda_v6_log_f32_f32(int nside, int l_max, int n_maps,
                                  const float* map_in, float* alm_re, float* alm_im);

/**
 * Allocate device memory for maps
 *
 * @param nside HEALPix nside
 * @param n_maps Number of maps
 * @return Device pointer, or NULL on error
 */
real_t* spht_allocate_map(int nside, int n_maps);

/**
 * Allocate device memory for alm
 *
 * @param l_max Maximum l
 * @param n_maps Number of fields
 * @return Device pointer, or NULL on error
 */
complex_t* spht_allocate_alm(int l_max, int n_maps);

/**
 * Free device memory
 */
void spht_free(void* ptr);

/**
 * Copy map from host to device
 */
int spht_map_to_device(int nside, int n_maps, const real_t* h_map, real_t* d_map);

/**
 * Copy map from device to host
 */
int spht_map_to_host(int nside, int n_maps, const real_t* d_map, real_t* h_map);

/**
 * Copy alm from host to device
 */
int spht_alm_to_device(int l_max, int n_maps, const complex_t* h_alm, complex_t* d_alm);

/**
 * Copy alm from device to host
 */
int spht_alm_to_host(int l_max, int n_maps, const complex_t* d_alm, complex_t* h_alm);

/**
 * Compute YLM for given log_beta values (for testing/debugging)
 *
 * @param l_max Maximum l
 * @param n_rings Number of rings
 * @param h_log_beta Host array of log(cos(theta)) values [n_rings]
 * @param h_ylm_out Host output array [l_max+1, l_max+1, n_rings]
 */
void spht_compute_ylm_debug(int l_max, int n_rings,
                             const real_t* h_log_beta,
                             real_t* h_ylm_out);

#ifdef __cplusplus
}
#endif

#endif // SPHT_API_H

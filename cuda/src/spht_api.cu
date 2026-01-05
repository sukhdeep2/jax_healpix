/**
 * SPHT CUDA API Implementation
 */

#include "../include/spht_api.h"
#include "../include/ylm_recurrence.cuh"
#include "../include/ring_processing.cuh"
#include <stdio.h>

// External declarations from transform files
extern "C" void map2alm_cuda(int nside, int l_max, int n_maps,
                              const real_t* map_in, complex_t* alm_out);
extern "C" void alm2map_cuda(int nside, int l_max, int n_maps,
                              const complex_t* alm_in, real_t* map_out);

spht_context_t* spht_create_context(int nside, int l_max) {
    spht_context_t* ctx = (spht_context_t*)malloc(sizeof(spht_context_t));
    if (!ctx) return NULL;

    ctx->nside = nside;
    ctx->l_max = (l_max > 0) ? l_max : 3 * nside;

    // Precompute ring geometry
    ctx->ring_geom = allocate_ring_geometry(nside);
    precompute_ring_geometry(nside, ctx->ring_geom);

    return ctx;
}

void spht_destroy_context(spht_context_t* ctx) {
    if (ctx) {
        free_ring_geometry(ctx->ring_geom);
        free(ctx);
    }
}

int spht_map2alm(spht_context_t* ctx, int n_maps,
                 const real_t* map_in, complex_t* alm_out) {
    if (!ctx || !map_in || !alm_out) return -1;

    map2alm_cuda(ctx->nside, ctx->l_max, n_maps, map_in, alm_out);
    return 0;
}

int spht_alm2map(spht_context_t* ctx, int n_maps,
                 const complex_t* alm_in, real_t* map_out) {
    if (!ctx || !alm_in || !map_out) return -1;

    alm2map_cuda(ctx->nside, ctx->l_max, n_maps, alm_in, map_out);
    return 0;
}

real_t* spht_allocate_map(int nside, int n_maps) {
    int n_rings = 4 * nside - 1;
    size_t size = (size_t)n_maps * n_rings * 4 * nside * sizeof(real_t);

    real_t* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate map: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
}

complex_t* spht_allocate_alm(int l_max, int n_maps) {
    int lp1 = l_max + 1;
    size_t size = (size_t)n_maps * lp1 * lp1 * sizeof(complex_t);

    complex_t* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate alm: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
}

void spht_free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

int spht_map_to_device(int nside, int n_maps, const real_t* h_map, real_t* d_map) {
    int n_rings = 4 * nside - 1;
    size_t size = (size_t)n_maps * n_rings * 4 * nside * sizeof(real_t);

    cudaError_t err = cudaMemcpy(d_map, h_map, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

int spht_map_to_host(int nside, int n_maps, const real_t* d_map, real_t* h_map) {
    int n_rings = 4 * nside - 1;
    size_t size = (size_t)n_maps * n_rings * 4 * nside * sizeof(real_t);

    cudaError_t err = cudaMemcpy(h_map, d_map, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

int spht_alm_to_device(int l_max, int n_maps, const complex_t* h_alm, complex_t* d_alm) {
    int lp1 = l_max + 1;
    size_t size = (size_t)n_maps * lp1 * lp1 * sizeof(complex_t);

    cudaError_t err = cudaMemcpy(d_alm, h_alm, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

int spht_alm_to_host(int l_max, int n_maps, const complex_t* d_alm, complex_t* h_alm) {
    int lp1 = l_max + 1;
    size_t size = (size_t)n_maps * lp1 * lp1 * sizeof(complex_t);

    cudaError_t err = cudaMemcpy(h_alm, d_alm, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

extern "C"
void spht_get_ring_geometry(int nside,
                             real_t* h_log_beta,
                             int8_t* h_beta_sign) {
    // Allocate and compute ring geometry on GPU
    ring_geometry_t* geom = allocate_ring_geometry(nside);
    precompute_ring_geometry(nside, geom);

    int n_rings = 4 * nside - 1;

    // Copy to host
    cudaMemcpy(h_log_beta, geom->log_beta, n_rings * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_beta_sign, geom->beta_sign, n_rings * sizeof(int8_t), cudaMemcpyDeviceToHost);

    free_ring_geometry(geom);
}

extern "C"
void spht_compute_ylm_debug(int l_max, int n_rings,
                             const real_t* h_log_beta,
                             real_t* h_ylm_out) {
    // Copy log_beta to device
    real_t* d_log_beta;
    cudaMalloc(&d_log_beta, n_rings * sizeof(real_t));
    cudaMemcpy(d_log_beta, h_log_beta, n_rings * sizeof(real_t), cudaMemcpyHostToDevice);

    // Create sign array (assume all positive cos(theta) for north hemisphere)
    int8_t* h_beta_sign = new int8_t[n_rings];
    for (int i = 0; i < n_rings; i++) {
        h_beta_sign[i] = 1;  // Positive cos(theta)
    }
    int8_t* d_beta_sign;
    cudaMalloc(&d_beta_sign, n_rings * sizeof(int8_t));
    cudaMemcpy(d_beta_sign, h_beta_sign, n_rings * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Allocate output
    int lp1 = l_max + 1;
    size_t ylm_size = (size_t)lp1 * lp1 * n_rings;
    real_t* d_ylm;
    cudaMalloc(&d_ylm, ylm_size * sizeof(real_t));

    // Compute YLM
    compute_ylm_spin0(l_max, n_rings, d_log_beta, d_beta_sign, d_ylm);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_ylm_out, d_ylm, ylm_size * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] h_beta_sign;
    cudaFree(d_log_beta);
    cudaFree(d_beta_sign);
    cudaFree(d_ylm);
}

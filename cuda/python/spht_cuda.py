"""
SPHT CUDA - Python Frontend for CUDA-accelerated Spherical Harmonic Transforms

This module provides a Python interface to the CUDA implementation of
spherical harmonic transforms on the HEALPix grid.
"""

import numpy as np
import ctypes
from ctypes import c_int, c_double, c_void_p, POINTER, Structure
from pathlib import Path
import os

# Type definitions matching CUDA types
class Complex64(Structure):
    _fields_ = [("x", c_double), ("y", c_double)]

# Load the shared library
def _load_library():
    """Load the SPHT CUDA shared library."""
    lib_paths = [
        Path(__file__).parent.parent / "build" / "libspht_cuda.so",
        Path(__file__).parent.parent / "libspht_cuda.so",
        "libspht_cuda.so",
    ]

    for path in lib_paths:
        try:
            return ctypes.CDLL(str(path))
        except OSError:
            continue

    raise ImportError(
        "Could not load libspht_cuda.so. Please build the CUDA library first.\n"
        "Run: cd cuda && mkdir build && cd build && cmake .. && make"
    )

# Global library handle (lazy loaded)
_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        _lib = _load_library()
        _setup_functions(_lib)
    return _lib

def _setup_functions(lib):
    """Set up function signatures for the C library."""
    # map2alm_cuda
    lib.map2alm_cuda.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p]
    lib.map2alm_cuda.restype = None

    # alm2map_cuda
    lib.alm2map_cuda.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p]
    lib.alm2map_cuda.restype = None

    # Memory allocation
    lib.spht_allocate_map.argtypes = [c_int, c_int]
    lib.spht_allocate_map.restype = c_void_p

    lib.spht_allocate_alm.argtypes = [c_int, c_int]
    lib.spht_allocate_alm.restype = c_void_p

    lib.spht_free.argtypes = [c_void_p]
    lib.spht_free.restype = None

    # Memory transfer
    lib.spht_map_to_device.argtypes = [c_int, c_int, c_void_p, c_void_p]
    lib.spht_map_to_device.restype = c_int

    lib.spht_map_to_host.argtypes = [c_int, c_int, c_void_p, c_void_p]
    lib.spht_map_to_host.restype = c_int

    lib.spht_alm_to_device.argtypes = [c_int, c_int, c_void_p, c_void_p]
    lib.spht_alm_to_device.restype = c_int

    lib.spht_alm_to_host.argtypes = [c_int, c_int, c_void_p, c_void_p]
    lib.spht_alm_to_host.restype = c_int

    # YLM debug function
    lib.spht_compute_ylm_debug.argtypes = [c_int, c_int, c_void_p, c_void_p]
    lib.spht_compute_ylm_debug.restype = None

    # Ring geometry debug function
    lib.spht_get_ring_geometry.argtypes = [c_int, c_void_p, c_void_p]
    lib.spht_get_ring_geometry.restype = None


def get_ring_geometry(nside: int):
    """
    Get the CUDA-computed ring geometry for debugging.

    Args:
        nside: HEALPix nside parameter

    Returns:
        log_beta: Array of log(|cos(theta)|) values [n_rings]
        beta_sign: Array of sign(cos(theta)) values [n_rings]
    """
    lib = _get_lib()
    n_rings = 4 * nside - 1

    log_beta = np.zeros(n_rings, dtype=np.float64)
    beta_sign = np.zeros(n_rings, dtype=np.int8)

    lib.spht_get_ring_geometry(
        nside,
        log_beta.ctypes.data_as(c_void_p),
        beta_sign.ctypes.data_as(c_void_p)
    )

    return log_beta, beta_sign


def compute_ylm_cuda(l_max: int, log_beta: np.ndarray) -> np.ndarray:
    """
    Compute YLM values using CUDA (for debugging/testing).

    Args:
        l_max: Maximum l value
        log_beta: Array of log(cos(theta)) values, shape [n_rings]

    Returns:
        ylm: Array of shape [l_max+1, l_max+1, n_rings]
    """
    lib = _get_lib()
    log_beta = np.ascontiguousarray(log_beta, dtype=np.float64)
    n_rings = len(log_beta)

    ylm_out = np.zeros((l_max + 1, l_max + 1, n_rings), dtype=np.float64)

    lib.spht_compute_ylm_debug(
        l_max, n_rings,
        log_beta.ctypes.data_as(c_void_p),
        ylm_out.ctypes.data_as(c_void_p)
    )

    return ylm_out


RING_BATCH_SIZE = 16  # Match CUDA constant


class SPHTCuda:
    """CUDA-accelerated Spherical Harmonic Transforms on HEALPix grid."""

    def __init__(self, nside: int, l_max: int = None):
        """
        Initialize SPHT CUDA context.

        Args:
            nside: HEALPix nside parameter (must be power of 2)
            l_max: Maximum l value. Defaults to 3*nside.
        """
        self.nside = nside
        self.l_max = l_max if l_max is not None else 3 * nside
        self.n_rings = 4 * nside - 1
        self._lib = _get_lib()

    def map2alm(self, maps: dict, spins: tuple = (0,)) -> dict:
        """
        Transform HEALPix maps to spherical harmonic coefficients.

        Args:
            maps: Dict with keys in {0, 2, -2} for spin types.
                  Each value is array of shape [n_maps, n_rings, 4*nside]
                  or [n_maps, npix] in 1D HEALPix format.
            spins: Tuple of spins to compute, e.g., (0,) or (0, 2)

        Returns:
            alm: Dict with same keys, each of shape [n_maps, l_max+1, l_max+1]
        """
        if 0 not in spins:
            raise NotImplementedError("Only spin-0 is currently implemented")

        alm_out = {}

        for s in spins:
            if s not in maps:
                continue

            map_data = np.asarray(maps[s], dtype=np.float64)

            # Handle 1D HEALPix format
            if map_data.ndim == 1:
                map_data = map_data.reshape(1, -1)
            if map_data.ndim == 2 and map_data.shape[1] == 12 * self.nside**2:
                map_data = self._reshape_maps_to_2d(map_data)

            n_maps = map_data.shape[0]

            # Ensure contiguous
            map_data = np.ascontiguousarray(map_data)

            # Allocate device memory
            d_map = self._lib.spht_allocate_map(self.nside, n_maps)
            d_alm = self._lib.spht_allocate_alm(self.l_max, n_maps)

            if d_map is None or d_alm is None:
                raise RuntimeError("Failed to allocate GPU memory")

            try:
                # Copy to device
                ret = self._lib.spht_map_to_device(
                    self.nside, n_maps,
                    map_data.ctypes.data_as(c_void_p),
                    d_map
                )
                if ret != 0:
                    raise RuntimeError("Failed to copy map to device")

                # Run transform
                self._lib.map2alm_cuda(self.nside, self.l_max, n_maps, d_map, d_alm)

                # Copy result back
                alm_shape = (n_maps, self.l_max + 1, self.l_max + 1)
                alm_data = np.zeros(alm_shape, dtype=np.complex128)

                ret = self._lib.spht_alm_to_host(
                    self.l_max, n_maps,
                    d_alm,
                    alm_data.ctypes.data_as(c_void_p)
                )
                if ret != 0:
                    raise RuntimeError("Failed to copy alm from device")

                alm_out[s] = alm_data

            finally:
                self._lib.spht_free(d_map)
                self._lib.spht_free(d_alm)

        return alm_out

    def alm2map(self, alm: dict, spins: tuple = (0,)) -> dict:
        """
        Transform spherical harmonic coefficients to HEALPix maps.

        Args:
            alm: Dict with keys in {0, 2, -2} for spin types.
                 Each value is array of shape [n_maps, l_max+1, l_max+1]
            spins: Tuple of spins to compute

        Returns:
            maps: Dict with same keys, each of shape [n_maps, n_rings, 4*nside]
        """
        if 0 not in spins:
            raise NotImplementedError("Only spin-0 is currently implemented")

        maps_out = {}

        for s in spins:
            if s not in alm:
                continue

            alm_data = np.asarray(alm[s], dtype=np.complex128)

            if alm_data.ndim == 2:
                alm_data = alm_data.reshape(1, alm_data.shape[0], alm_data.shape[1])

            n_maps = alm_data.shape[0]

            # Ensure contiguous
            alm_data = np.ascontiguousarray(alm_data)

            # Allocate device memory
            d_alm = self._lib.spht_allocate_alm(self.l_max, n_maps)
            d_map = self._lib.spht_allocate_map(self.nside, n_maps)

            if d_alm is None or d_map is None:
                raise RuntimeError("Failed to allocate GPU memory")

            try:
                # Copy to device
                ret = self._lib.spht_alm_to_device(
                    self.l_max, n_maps,
                    alm_data.ctypes.data_as(c_void_p),
                    d_alm
                )
                if ret != 0:
                    raise RuntimeError("Failed to copy alm to device")

                # Run transform
                self._lib.alm2map_cuda(self.nside, self.l_max, n_maps, d_alm, d_map)

                # Copy result back
                map_shape = (n_maps, self.n_rings, 4 * self.nside)
                map_data = np.zeros(map_shape, dtype=np.float64)

                ret = self._lib.spht_map_to_host(
                    self.nside, n_maps,
                    d_map,
                    map_data.ctypes.data_as(c_void_p)
                )
                if ret != 0:
                    raise RuntimeError("Failed to copy map from device")

                maps_out[s] = map_data

            finally:
                self._lib.spht_free(d_alm)
                self._lib.spht_free(d_map)

        return maps_out

    def _reshape_maps_to_2d(self, maps_1d):
        """
        Convert 1D HEALPix format [n_maps, npix] to 2D ring format [n_maps, n_rings, 4*nside].

        Reference: jax_healpix/reshape_utils.py
        """
        n_maps = maps_1d.shape[0]
        npix = 12 * self.nside**2
        n_rings = 4 * self.nside - 1

        maps_2d = np.zeros((n_maps, n_rings, 4 * self.nside), dtype=np.float64)

        for ring_i in range(1, n_rings + 1):
            # Get ring info
            start_pix, npix_ring = self._get_ring_info(ring_i)

            for t in range(n_maps):
                maps_2d[t, ring_i - 1, :npix_ring] = maps_1d[t, start_pix:start_pix + npix_ring]

        return maps_2d

    def _get_ring_info(self, ring_i):
        """Get start pixel and number of pixels for a ring (1-indexed)."""
        nside = self.nside

        if ring_i < nside:
            # North polar cap
            npix_ring = 4 * ring_i
            start_pix = 2 * ring_i * (ring_i - 1)
        elif ring_i <= 3 * nside:
            # Equatorial belt
            npix_ring = 4 * nside
            north_cap_pix = 2 * nside * (nside - 1)
            start_pix = north_cap_pix + (ring_i - nside) * 4 * nside
        else:
            # South polar cap
            ring_from_south = 4 * nside - ring_i
            npix_ring = 4 * ring_from_south
            total_pix = 12 * nside * nside
            south_remaining = 2 * ring_from_south * (ring_from_south + 1)
            start_pix = total_pix - south_remaining

        return start_pix, npix_ring


def map2alm_cuda(nside: int, l_max: int, maps: np.ndarray) -> np.ndarray:
    """
    Convenience function for spin-0 map2alm transform.

    Args:
        nside: HEALPix nside
        l_max: Maximum l
        maps: Input maps [n_maps, n_rings, 4*nside] or [n_maps, npix]

    Returns:
        alm: Output coefficients [n_maps, l_max+1, l_max+1]
    """
    spht = SPHTCuda(nside, l_max)
    result = spht.map2alm({0: maps}, spins=(0,))
    return result[0]


def alm2map_cuda(nside: int, l_max: int, alm: np.ndarray) -> np.ndarray:
    """
    Convenience function for spin-0 alm2map transform.

    Args:
        nside: HEALPix nside
        l_max: Maximum l
        alm: Input coefficients [n_maps, l_max+1, l_max+1]

    Returns:
        maps: Output maps [n_maps, n_rings, 4*nside]
    """
    spht = SPHTCuda(nside, l_max)
    result = spht.alm2map({0: alm}, spins=(0,))
    return result[0]

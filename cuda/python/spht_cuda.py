"""
SPHT CUDA - Python Frontend for CUDA-accelerated Spherical Harmonic Transforms

This module provides a Python interface to the CUDA implementation of
spherical harmonic transforms on the HEALPix grid.
"""

import numpy as np
import ctypes
from ctypes import c_int, c_float, c_double, c_void_p, POINTER, Structure
from pathlib import Path
import os


# ============================================================================
# Configuration (JAX-style precision flags)
# ============================================================================

class _SPHTConfig:
    """Global configuration for SPHT CUDA precision settings."""

    def __init__(self):
        self._storage_precision = "float64"
        self._recurrence_precision = "float64"

    @property
    def storage_precision(self) -> str:
        """Precision for map/alm storage: 'float64' or 'float32'"""
        return self._storage_precision

    @property
    def recurrence_precision(self) -> str:
        """Precision for Ylm recurrence: 'float64' or 'float32'"""
        return self._recurrence_precision

    def update(self, key: str, value):
        """Update a configuration value.

        Args:
            key: One of 'spht_storage_precision' or 'spht_recurrence_precision'
            value: 'float64' or 'float32'
        """
        if key == "spht_storage_precision":
            if value not in ("float64", "float32"):
                raise ValueError(f"storage_precision must be 'float64' or 'float32', got {value}")
            self._storage_precision = value
        elif key == "spht_recurrence_precision":
            if value not in ("float64", "float32"):
                raise ValueError(f"recurrence_precision must be 'float64' or 'float32', got {value}")
            self._recurrence_precision = value
        else:
            raise KeyError(f"Unknown config key: {key}")


# Global config instance
config = _SPHTConfig()


def set_precision(storage: str = None, recurrence: str = None):
    """Set precision for SPHT CUDA computations.

    Similar to jax.config.update("jax_enable_x64", True).

    Args:
        storage: Precision for map/alm data - 'float64' or 'float32'
        recurrence: Precision for Ylm recurrence - 'float64' or 'float32'

    Examples:
        # Full float64 (default, highest accuracy)
        set_precision(storage="float64", recurrence="float64")

        # Full float32 (fastest, may lose accuracy at high l)
        set_precision(storage="float32", recurrence="float32")

        # Mixed: float32 storage with float64 recurrence (balanced)
        set_precision(storage="float32", recurrence="float64")
    """
    if storage is not None:
        config.update("spht_storage_precision", storage)
    if recurrence is not None:
        config.update("spht_recurrence_precision", recurrence)


# Type definitions matching CUDA types
class Complex64(Structure):
    _fields_ = [("x", c_double), ("y", c_double)]

class Complex32(Structure):
    _fields_ = [("x", c_float), ("y", c_float)]

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

    # Add optimized version (cuFFT + cuBLAS)
    lib.map2alm_cuda_v2.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p]
    lib.map2alm_cuda_v2.restype = None

    # Add fused per-ring parallel version
    lib.map2alm_cuda_v3.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p]
    lib.map2alm_cuda_v3.restype = None

    # Add tiled version for large nside
    lib.map2alm_cuda_v4.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p]
    lib.map2alm_cuda_v4.restype = None

    # Multi-precision version (v5)
    lib.map2alm_cuda_v5.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p]
    lib.map2alm_cuda_v5.restype = None

    # v5 precision combinations: map2alm_cuda_v5_{storage}_{recurrence}
    # f64_f64: float64 storage, float64 recurrence (default, highest accuracy)
    lib.map2alm_cuda_v5_f64_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v5_f64_f64.restype = None

    # f64_f32: float64 storage, float32 recurrence
    lib.map2alm_cuda_v5_f64_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v5_f64_f32.restype = None

    # f32_f64: float32 storage, float64 recurrence
    lib.map2alm_cuda_v5_f32_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v5_f32_f64.restype = None

    # f32_f32: float32 storage, float32 recurrence (fastest)
    lib.map2alm_cuda_v5_f32_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v5_f32_f32.restype = None

    # Backwards-compatible aliases
    lib.map2alm_cuda_v5_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v5_f64.restype = None
    lib.map2alm_cuda_v5_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v5_f32.restype = None

    # ========== V6: Optimal warp-per-m with NO atomics ==========
    # v6 precision combinations: map2alm_cuda_v6_{storage}_{recurrence}
    lib.map2alm_cuda_v6_f64_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_f64_f64.restype = None

    lib.map2alm_cuda_v6_f64_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_f64_f32.restype = None

    lib.map2alm_cuda_v6_f32_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_f32_f64.restype = None

    lib.map2alm_cuda_v6_f32_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_f32_f32.restype = None

    # Backwards-compatible v6 aliases
    lib.map2alm_cuda_v6_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_f64.restype = None
    lib.map2alm_cuda_v6_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_f32.restype = None

    # alm2map_cuda_v5 precision combinations
    # f64_f64: float64 storage, float64 recurrence (default, highest accuracy)
    lib.alm2map_cuda_v5_f64_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v5_f64_f64.restype = None

    # f64_f32: float64 storage, float32 recurrence
    lib.alm2map_cuda_v5_f64_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v5_f64_f32.restype = None

    # f32_f64: float32 storage, float64 recurrence
    lib.alm2map_cuda_v5_f32_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v5_f32_f64.restype = None

    # f32_f32: float32 storage, float32 recurrence (fastest)
    lib.alm2map_cuda_v5_f32_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v5_f32_f32.restype = None

    # Backwards-compatible aliases for alm2map_v5
    lib.alm2map_cuda_v5_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v5_f64.restype = None
    lib.alm2map_cuda_v5_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v5_f32.restype = None

    # alm2map_cuda
    lib.alm2map_cuda.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p]
    lib.alm2map_cuda.restype = None

    # Memory allocation (float64)
    lib.spht_allocate_map.argtypes = [c_int, c_int]
    lib.spht_allocate_map.restype = c_void_p

    lib.spht_allocate_alm.argtypes = [c_int, c_int]
    lib.spht_allocate_alm.restype = c_void_p

    lib.spht_free.argtypes = [c_void_p]
    lib.spht_free.restype = None

    # Memory transfer (float64)
    lib.spht_map_to_device.argtypes = [c_int, c_int, c_void_p, c_void_p]
    lib.spht_map_to_device.restype = c_int

    lib.spht_map_to_host.argtypes = [c_int, c_int, c_void_p, c_void_p]
    lib.spht_map_to_host.restype = c_int

    lib.spht_alm_to_device.argtypes = [c_int, c_int, c_void_p, c_void_p]
    lib.spht_alm_to_device.restype = c_int

    lib.spht_alm_to_host.argtypes = [c_int, c_int, c_void_p, c_void_p]
    lib.spht_alm_to_host.restype = c_int

    # Float32 memory allocation and transfer (if available)
    try:
        lib.spht_allocate_map_f32.argtypes = [c_int, c_int]
        lib.spht_allocate_map_f32.restype = c_void_p

        lib.spht_allocate_alm_f32.argtypes = [c_int, c_int]
        lib.spht_allocate_alm_f32.restype = c_void_p

        lib.spht_map_to_device_f32.argtypes = [c_int, c_int, c_void_p, c_void_p]
        lib.spht_map_to_device_f32.restype = c_int

        lib.spht_alm_to_host_f32.argtypes = [c_int, c_int, c_void_p, c_void_p]
        lib.spht_alm_to_host_f32.restype = c_int
    except AttributeError:
        pass  # Float32 API not available in this build

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

    def __init__(self, nside: int, l_max: int = None, version: str = "v5",
                 storage_precision: str = None, recurrence_precision: str = None):
        """
        Initialize SPHT CUDA context.

        Args:
            nside: HEALPix nside parameter (must be power of 2)
            l_max: Maximum l value. Defaults to 3*nside.
            version: Implementation version to use:
                - "v1": Original batched implementation
                - "v2": cuFFT + cuBLAS implementation
                - "v3": Fused per-ring parallel (fast for small nside)
                - "v4": Tiled per-ring (scales to large nside)
                - "v5": Multi-precision templated version
                - "v6": Optimal warp-per-m with NO atomics (default)
            storage_precision: Precision for map/alm data - 'float64' or 'float32'
                               If None, uses global config.storage_precision
            recurrence_precision: Precision for Ylm recurrence - 'float64' or 'float32'
                                  If None, uses global config.recurrence_precision
        """
        self.nside = nside
        self.l_max = l_max if l_max is not None else 3 * nside
        self.n_rings = 4 * nside - 1
        self.version = version
        # Use provided precision or fall back to global config
        self.storage_precision = storage_precision or config.storage_precision
        self.recurrence_precision = recurrence_precision or config.recurrence_precision
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
                 dtype is complex128 for float64 storage, complex64 for float32
        """
        if 0 not in spins:
            raise NotImplementedError("Only spin-0 is currently implemented")

        alm_out = {}
        use_f32 = (self.storage_precision == "float32")

        for s in spins:
            if s not in maps:
                continue

            # Convert to appropriate dtype
            dtype = np.float32 if use_f32 else np.float64
            map_data = np.asarray(maps[s], dtype=dtype)

            # Handle 1D HEALPix format
            if map_data.ndim == 1:
                map_data = map_data.reshape(1, -1)
            if map_data.ndim == 2 and map_data.shape[1] == 12 * self.nside**2:
                map_data = self._reshape_maps_to_2d(map_data, dtype=dtype)

            n_maps = map_data.shape[0]
            map_data = np.ascontiguousarray(map_data)

            # For v5/v6 with precision control, use direct kernel calls
            if self.version == "v6":
                alm_data = self._map2alm_v6(map_data, n_maps)
            elif self.version == "v5":
                alm_data = self._map2alm_v5(map_data, n_maps)
            else:
                # Legacy versions use float64 only
                if use_f32:
                    map_data = map_data.astype(np.float64)
                alm_data = self._map2alm_legacy(map_data, n_maps)

            alm_out[s] = alm_data

        return alm_out

    def _map2alm_v5(self, map_data: np.ndarray, n_maps: int) -> np.ndarray:
        """Run v5 transform with precision control."""
        use_f32 = (self.storage_precision == "float32")
        use_f32_recur = (self.recurrence_precision == "float32")
        lp1 = self.l_max + 1

        # Allocate output arrays on host
        alm_real = np.zeros((n_maps, lp1, lp1), dtype=map_data.dtype)
        alm_imag = np.zeros((n_maps, lp1, lp1), dtype=map_data.dtype)

        # Allocate device memory
        map_size = n_maps * self.n_rings * 4 * self.nside * map_data.itemsize
        alm_size = n_maps * lp1 * lp1 * map_data.itemsize

        import ctypes
        d_map = ctypes.c_void_p()
        d_alm_real = ctypes.c_void_p()
        d_alm_imag = ctypes.c_void_p()

        # cudaMalloc
        cuda_rt = ctypes.CDLL("libcudart.so")
        cuda_rt.cudaMalloc(ctypes.byref(d_map), map_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_alm_real), alm_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_alm_imag), alm_size)
        cuda_rt.cudaMemset(d_alm_real, 0, alm_size)
        cuda_rt.cudaMemset(d_alm_imag, 0, alm_size)

        # Copy map to device
        cuda_rt.cudaMemcpy(d_map, map_data.ctypes.data_as(c_void_p),
                          map_size, 1)  # cudaMemcpyHostToDevice = 1

        try:
            # Select kernel based on precision combination
            if use_f32:
                if use_f32_recur:
                    self._lib.map2alm_cuda_v5_f32_f32(
                        self.nside, self.l_max, n_maps,
                        d_map, d_alm_real, d_alm_imag)
                else:
                    self._lib.map2alm_cuda_v5_f32_f64(
                        self.nside, self.l_max, n_maps,
                        d_map, d_alm_real, d_alm_imag)
            else:
                if use_f32_recur:
                    self._lib.map2alm_cuda_v5_f64_f32(
                        self.nside, self.l_max, n_maps,
                        d_map, d_alm_real, d_alm_imag)
                else:
                    self._lib.map2alm_cuda_v5_f64_f64(
                        self.nside, self.l_max, n_maps,
                        d_map, d_alm_real, d_alm_imag)

            # Copy results back
            cuda_rt.cudaMemcpy(alm_real.ctypes.data_as(c_void_p),
                              d_alm_real, alm_size, 2)  # cudaMemcpyDeviceToHost = 2
            cuda_rt.cudaMemcpy(alm_imag.ctypes.data_as(c_void_p),
                              d_alm_imag, alm_size, 2)
        finally:
            cuda_rt.cudaFree(d_map)
            cuda_rt.cudaFree(d_alm_real)
            cuda_rt.cudaFree(d_alm_imag)

        # Combine to complex
        if use_f32:
            alm_data = (alm_real + 1j * alm_imag).astype(np.complex64)
        else:
            alm_data = alm_real + 1j * alm_imag

        return alm_data

    def _map2alm_v6(self, map_data: np.ndarray, n_maps: int) -> np.ndarray:
        """Run v6 transform with precision control (optimal warp-per-m, no atomics)."""
        use_f32 = (self.storage_precision == "float32")
        use_f32_recur = (self.recurrence_precision == "float32")
        lp1 = self.l_max + 1

        # Allocate output arrays on host
        alm_real = np.zeros((n_maps, lp1, lp1), dtype=map_data.dtype)
        alm_imag = np.zeros((n_maps, lp1, lp1), dtype=map_data.dtype)

        # Allocate device memory
        map_size = n_maps * self.n_rings * 4 * self.nside * map_data.itemsize
        alm_size = n_maps * lp1 * lp1 * map_data.itemsize

        import ctypes
        d_map = ctypes.c_void_p()
        d_alm_real = ctypes.c_void_p()
        d_alm_imag = ctypes.c_void_p()

        # cudaMalloc
        cuda_rt = ctypes.CDLL("libcudart.so")
        cuda_rt.cudaMalloc(ctypes.byref(d_map), map_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_alm_real), alm_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_alm_imag), alm_size)
        cuda_rt.cudaMemset(d_alm_real, 0, alm_size)
        cuda_rt.cudaMemset(d_alm_imag, 0, alm_size)

        # Copy map to device
        cuda_rt.cudaMemcpy(d_map, map_data.ctypes.data_as(c_void_p),
                          map_size, 1)  # cudaMemcpyHostToDevice = 1

        try:
            # Select kernel based on precision combination
            if use_f32:
                if use_f32_recur:
                    self._lib.map2alm_cuda_v6_f32_f32(
                        self.nside, self.l_max, n_maps,
                        d_map, d_alm_real, d_alm_imag)
                else:
                    self._lib.map2alm_cuda_v6_f32_f64(
                        self.nside, self.l_max, n_maps,
                        d_map, d_alm_real, d_alm_imag)
            else:
                if use_f32_recur:
                    self._lib.map2alm_cuda_v6_f64_f32(
                        self.nside, self.l_max, n_maps,
                        d_map, d_alm_real, d_alm_imag)
                else:
                    self._lib.map2alm_cuda_v6_f64_f64(
                        self.nside, self.l_max, n_maps,
                        d_map, d_alm_real, d_alm_imag)

            # Copy results back
            cuda_rt.cudaMemcpy(alm_real.ctypes.data_as(c_void_p),
                              d_alm_real, alm_size, 2)  # cudaMemcpyDeviceToHost = 2
            cuda_rt.cudaMemcpy(alm_imag.ctypes.data_as(c_void_p),
                              d_alm_imag, alm_size, 2)
        finally:
            cuda_rt.cudaFree(d_map)
            cuda_rt.cudaFree(d_alm_real)
            cuda_rt.cudaFree(d_alm_imag)

        # Combine to complex
        if use_f32:
            alm_data = (alm_real + 1j * alm_imag).astype(np.complex64)
        else:
            alm_data = alm_real + 1j * alm_imag

        return alm_data

    def _map2alm_legacy(self, map_data: np.ndarray, n_maps: int) -> np.ndarray:
        """Run legacy (v1-v4) transform."""
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
            if self.version == "v4":
                self._lib.map2alm_cuda_v4(self.nside, self.l_max, n_maps, d_map, d_alm)
            elif self.version == "v3":
                self._lib.map2alm_cuda_v3(self.nside, self.l_max, n_maps, d_map, d_alm)
            elif self.version == "v2":
                self._lib.map2alm_cuda_v2(self.nside, self.l_max, n_maps, d_map, d_alm)
            else:
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

            return alm_data

        finally:
            self._lib.spht_free(d_map)
            self._lib.spht_free(d_alm)

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
        use_f32 = (self.storage_precision == "float32")

        for s in spins:
            if s not in alm:
                continue

            # For v5 with precision control
            if self.version == "v5":
                map_data = self._alm2map_v5(alm[s])
            else:
                map_data = self._alm2map_legacy(alm[s])

            maps_out[s] = map_data

        return maps_out

    def _alm2map_v5(self, alm_data: np.ndarray) -> np.ndarray:
        """Run v5 alm2map transform with precision control."""
        use_f32 = (self.storage_precision == "float32")
        use_f32_recur = (self.recurrence_precision == "float32")
        lp1 = self.l_max + 1

        # Convert to appropriate dtype
        if use_f32:
            alm_data = np.asarray(alm_data, dtype=np.complex64)
        else:
            alm_data = np.asarray(alm_data, dtype=np.complex128)

        if alm_data.ndim == 2:
            alm_data = alm_data.reshape(1, alm_data.shape[0], alm_data.shape[1])

        n_maps = alm_data.shape[0]
        alm_data = np.ascontiguousarray(alm_data)

        # Separate into real and imag parts
        if use_f32:
            alm_real = np.ascontiguousarray(alm_data.real.astype(np.float32))
            alm_imag = np.ascontiguousarray(alm_data.imag.astype(np.float32))
            map_dtype = np.float32
        else:
            alm_real = np.ascontiguousarray(alm_data.real.astype(np.float64))
            alm_imag = np.ascontiguousarray(alm_data.imag.astype(np.float64))
            map_dtype = np.float64

        # Allocate output
        map_shape = (n_maps, self.n_rings, 4 * self.nside)
        map_out = np.zeros(map_shape, dtype=map_dtype)

        # Allocate device memory
        alm_size = n_maps * lp1 * lp1 * alm_real.itemsize
        map_size = n_maps * self.n_rings * 4 * self.nside * map_out.itemsize

        import ctypes
        d_alm_real = ctypes.c_void_p()
        d_alm_imag = ctypes.c_void_p()
        d_map = ctypes.c_void_p()

        cuda_rt = ctypes.CDLL("libcudart.so")
        cuda_rt.cudaMalloc(ctypes.byref(d_alm_real), alm_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_alm_imag), alm_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_map), map_size)
        cuda_rt.cudaMemset(d_map, 0, map_size)

        # Copy alm to device
        cuda_rt.cudaMemcpy(d_alm_real, alm_real.ctypes.data_as(c_void_p),
                          alm_size, 1)  # cudaMemcpyHostToDevice = 1
        cuda_rt.cudaMemcpy(d_alm_imag, alm_imag.ctypes.data_as(c_void_p),
                          alm_size, 1)

        try:
            # Select kernel based on precision combination
            if use_f32:
                if use_f32_recur:
                    self._lib.alm2map_cuda_v5_f32_f32(
                        self.nside, self.l_max, n_maps,
                        d_alm_real, d_alm_imag, d_map)
                else:
                    self._lib.alm2map_cuda_v5_f32_f64(
                        self.nside, self.l_max, n_maps,
                        d_alm_real, d_alm_imag, d_map)
            else:
                if use_f32_recur:
                    self._lib.alm2map_cuda_v5_f64_f32(
                        self.nside, self.l_max, n_maps,
                        d_alm_real, d_alm_imag, d_map)
                else:
                    self._lib.alm2map_cuda_v5_f64_f64(
                        self.nside, self.l_max, n_maps,
                        d_alm_real, d_alm_imag, d_map)

            # Copy result back
            cuda_rt.cudaMemcpy(map_out.ctypes.data_as(c_void_p),
                              d_map, map_size, 2)  # cudaMemcpyDeviceToHost = 2
        finally:
            cuda_rt.cudaFree(d_alm_real)
            cuda_rt.cudaFree(d_alm_imag)
            cuda_rt.cudaFree(d_map)

        return map_out

    def _alm2map_legacy(self, alm_data: np.ndarray) -> np.ndarray:
        """Run legacy alm2map transform."""
        alm_data = np.asarray(alm_data, dtype=np.complex128)

        if alm_data.ndim == 2:
            alm_data = alm_data.reshape(1, alm_data.shape[0], alm_data.shape[1])

        n_maps = alm_data.shape[0]
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

            return map_data

        finally:
            self._lib.spht_free(d_alm)
            self._lib.spht_free(d_map)

    def _reshape_maps_to_2d(self, maps_1d, dtype=np.float64):
        """
        Convert 1D HEALPix format [n_maps, npix] to 2D ring format [n_maps, n_rings, 4*nside].

        Reference: jax_healpix/reshape_utils.py
        """
        n_maps = maps_1d.shape[0]
        npix = 12 * self.nside**2
        n_rings = 4 * self.nside - 1

        maps_2d = np.zeros((n_maps, n_rings, 4 * self.nside), dtype=dtype)

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

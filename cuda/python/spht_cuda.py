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

VALID_ACCUMULATION_MODES = ("linear", "log")


class _SPHTConfig:
    """Global configuration for SPHT CUDA precision settings."""

    def __init__(self):
        self._storage_precision = "float64"
        self._recurrence_precision = "float64"
        self._accumulation_mode = "linear"

    @property
    def storage_precision(self) -> str:
        """Precision for map/alm storage: 'float64' or 'float32'"""
        return self._storage_precision

    @property
    def recurrence_precision(self) -> str:
        """Precision for Ylm recurrence: 'float64' or 'float32'"""
        return self._recurrence_precision

    @property
    def accumulation_mode(self) -> str:
        """Accumulation mode: 'linear' (fast FMA) or 'log' (logsumexp)"""
        return self._accumulation_mode

    def get_effective_accumulation_mode(self) -> str:
        """Get the effective accumulation mode (may be forced by precision)."""
        # bf16 always requires log mode
        if self._storage_precision == "bfloat16" or self._recurrence_precision == "bfloat16":
            return "log"
        return self._accumulation_mode

    def update(self, key: str, value):
        """Update a configuration value.

        Args:
            key: One of 'spht_storage_precision', 'spht_recurrence_precision', or 'spht_accumulation_mode'
            value: 'float64', 'float32', 'bfloat16', 'linear', or 'log'
        """
        if key == "spht_storage_precision":
            if value not in ("float64", "float32", "bfloat16"):
                raise ValueError(f"storage_precision must be 'float64', 'float32', or 'bfloat16', got {value}")
            self._storage_precision = value
            # bf16 requires log accumulation
            if value == "bfloat16" and self._accumulation_mode == "linear":
                self._accumulation_mode = "log"
        elif key == "spht_recurrence_precision":
            if value not in ("float64", "float32", "bfloat16"):
                raise ValueError(f"recurrence_precision must be 'float64', 'float32', or 'bfloat16', got {value}")
            self._recurrence_precision = value
            # bf16 requires log accumulation
            if value == "bfloat16" and self._accumulation_mode == "linear":
                self._accumulation_mode = "log"
        elif key == "spht_accumulation_mode":
            if value not in VALID_ACCUMULATION_MODES:
                raise ValueError(f"accumulation_mode must be one of {VALID_ACCUMULATION_MODES}, got {value}")
            # Cannot use linear with bf16
            if value == "linear" and (self._storage_precision == "bfloat16" or
                                       self._recurrence_precision == "bfloat16"):
                raise ValueError("accumulation_mode='linear' not supported with bfloat16 precision")
            self._accumulation_mode = value
        else:
            raise KeyError(f"Unknown config key: {key}")


# Global config instance
config = _SPHTConfig()


def set_precision(storage: str = None, recurrence: str = None, accumulation: str = None):
    """Set precision for SPHT CUDA computations.

    Similar to jax.config.update("jax_enable_x64", True).

    Args:
        storage: Precision for map/alm data - 'float64', 'float32', or 'bfloat16'
        recurrence: Precision for Ylm recurrence - 'float64', 'float32', or 'bfloat16'
        accumulation: Accumulation mode - 'linear' (fast, default) or 'log' (numerically stable)

    Note:
        - bfloat16 requires accumulation='log' (set automatically)
        - 'linear' mode uses fast FMA-based accumulation
        - 'log' mode uses logsumexp (5-10x slower but handles extreme dynamic range)

    Examples:
        # Full float64 (default, highest accuracy)
        set_precision(storage="float64", recurrence="float64")

        # Full float32 (fastest, may lose accuracy at high l)
        set_precision(storage="float32", recurrence="float32")

        # Mixed: float32 storage with float64 recurrence (balanced)
        set_precision(storage="float32", recurrence="float64")

        # LOG mode for extreme dynamic range
        set_precision(storage="float64", recurrence="float64", accumulation="log")
    """
    if storage is not None:
        config.update("spht_storage_precision", storage)
    if recurrence is not None:
        config.update("spht_recurrence_precision", recurrence)
    if accumulation is not None:
        config.update("spht_accumulation_mode", accumulation)


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
_cuda_rt = None
_pinned_buffers = {}  # Cache for pinned memory buffers: (size, dtype) -> (h_real, h_imag)

# Device buffer cache for I/O buffers (eliminates cudaMalloc/cudaFree overhead)
# Keys: (buffer_name, use_f32), Values: (ptr, current_size)
_device_buffers = {}

def _get_lib():
    global _lib
    if _lib is None:
        _lib = _load_library()
        _setup_functions(_lib)
    return _lib

def _get_cuda_rt():
    """Get cached CUDA runtime library handle."""
    global _cuda_rt
    if _cuda_rt is None:
        _cuda_rt = ctypes.CDLL("libcudart.so")
    return _cuda_rt

def _get_pinned_buffers(size, use_f32):
    """Get or allocate cached pinned memory buffers."""
    global _pinned_buffers
    key = (size, use_f32)
    if key not in _pinned_buffers:
        cuda_rt = _get_cuda_rt()
        h_real = ctypes.c_void_p()
        h_imag = ctypes.c_void_p()
        cuda_rt.cudaHostAlloc(ctypes.byref(h_real), size, 0)
        cuda_rt.cudaHostAlloc(ctypes.byref(h_imag), size, 0)
        _pinned_buffers[key] = (h_real, h_imag)
    return _pinned_buffers[key]


def _get_device_buffer(name: str, required_size: int, use_f32: bool):
    """Get or allocate a cached device buffer using high-water mark pattern.

    Args:
        name: Buffer identifier (e.g., 'alm_real', 'alm_imag', 'map')
        required_size: Required size in bytes
        use_f32: Whether this is for f32 or f64 precision

    Returns:
        ctypes.c_void_p: Device pointer to buffer
    """
    global _device_buffers
    key = (name, use_f32)
    cuda_rt = _get_cuda_rt()

    if key in _device_buffers:
        ptr, current_size = _device_buffers[key]
        if current_size >= required_size:
            # Existing buffer is large enough
            return ptr
        else:
            # Need larger buffer - free old one first
            cuda_rt.cudaFree(ptr)

    # Allocate new buffer
    ptr = ctypes.c_void_p()
    cuda_rt.cudaMalloc(ctypes.byref(ptr), required_size)
    _device_buffers[key] = (ptr, required_size)
    return ptr


def clear_device_buffer_cache():
    """Free all cached device buffers. Call this to release GPU memory."""
    global _device_buffers
    cuda_rt = _get_cuda_rt()
    for key, (ptr, size) in _device_buffers.items():
        cuda_rt.cudaFree(ptr)
    _device_buffers.clear()

def _setup_functions(lib):
    """Set up function signatures for the C library."""
    # ========== alm2cl functions ==========
    # Auto-spectrum
    lib.alm2cl_cuda_auto_f64.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2cl_cuda_auto_f64.restype = None
    lib.alm2cl_cuda_auto_f32.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2cl_cuda_auto_f32.restype = None

    # Cross-spectrum
    lib.alm2cl_cuda_cross_f64.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.alm2cl_cuda_cross_f64.restype = None
    lib.alm2cl_cuda_cross_f32.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.alm2cl_cuda_cross_f32.restype = None

    # All pairs (auto + cross)
    lib.alm2cl_cuda_all_pairs_f64.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2cl_cuda_all_pairs_f64.restype = None
    lib.alm2cl_cuda_all_pairs_f32.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2cl_cuda_all_pairs_f32.restype = None

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

    # ========== V6: alm2map optimal warp-per-m ==========
    # alm2map_cuda_v6 precision combinations
    lib.alm2map_cuda_v6_f64_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_f64_f64.restype = None

    lib.alm2map_cuda_v6_f64_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_f64_f32.restype = None

    lib.alm2map_cuda_v6_f32_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_f32_f64.restype = None

    lib.alm2map_cuda_v6_f32_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_f32_f32.restype = None

    # Backwards-compatible v6 aliases for alm2map
    lib.alm2map_cuda_v6_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_f64.restype = None
    lib.alm2map_cuda_v6_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_f32.restype = None

    # ========== V6 Spin-2: map2alm Q,U -> E,B ==========
    # Args: (nside, l_max, n_maps, d_map_Q, d_map_U, d_alm_E_real, d_alm_E_imag, d_alm_B_real, d_alm_B_imag)
    lib.map2alm_cuda_v6_spin2_f64_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_spin2_f64_f64.restype = None

    lib.map2alm_cuda_v6_spin2_f64_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_spin2_f64_f32.restype = None

    lib.map2alm_cuda_v6_spin2_f32_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_spin2_f32_f64.restype = None

    lib.map2alm_cuda_v6_spin2_f32_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_spin2_f32_f32.restype = None

    # ========== V6 Spin-2: alm2map E,B -> Q,U ==========
    # Args: (nside, l_max, n_maps, d_alm_E_real, d_alm_E_imag, d_alm_B_real, d_alm_B_imag, d_map_Q, d_map_U)
    lib.alm2map_cuda_v6_spin2_f64_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_spin2_f64_f64.restype = None

    lib.alm2map_cuda_v6_spin2_f64_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_spin2_f64_f32.restype = None

    lib.alm2map_cuda_v6_spin2_f32_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_spin2_f32_f64.restype = None

    lib.alm2map_cuda_v6_spin2_f32_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_spin2_f32_f32.restype = None

    # ========== V6 LOG mode: map2alm with logsumexp accumulation ==========
    lib.map2alm_cuda_v6_log_f64_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_log_f64_f64.restype = None

    lib.map2alm_cuda_v6_log_f64_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_log_f64_f32.restype = None

    lib.map2alm_cuda_v6_log_f32_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_log_f32_f64.restype = None

    lib.map2alm_cuda_v6_log_f32_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_log_f32_f32.restype = None

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

    # v6 Phase 1 method configuration
    lib.map2alm_v6_set_phase1_method.argtypes = [c_int]
    lib.map2alm_v6_set_phase1_method.restype = None
    lib.map2alm_v6_get_phase1_method.argtypes = []
    lib.map2alm_v6_get_phase1_method.restype = c_int


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


# ============================================================================
# Phase 1 Method Configuration for v6
# ============================================================================

# Method constants
PHASE1_DFT = 0            # Direct DFT for all rings (default, simple)
PHASE1_FFT_EQUATORIAL = 1  # FFT for equatorial rings, DFT for polar
PHASE1_BLUESTEIN = 2       # Bluestein FFT for all rings (cuHPX-style)


def set_phase1_method(method: int):
    """
    Set the Phase 1 (Gm computation) method for v6 map2alm.

    Args:
        method: One of:
            - PHASE1_DFT (0): Direct DFT for all rings (default, simple, good for small nside)
            - PHASE1_FFT_EQUATORIAL (1): FFT for equatorial rings, DFT for polar
            - PHASE1_BLUESTEIN (2): Bluestein FFT for all rings (cuHPX-style)

    Examples:
        # Use DFT (default)
        set_phase1_method(PHASE1_DFT)

        # Use Bluestein FFT (cuHPX-style)
        set_phase1_method(PHASE1_BLUESTEIN)
    """
    lib = _get_lib()
    lib.map2alm_v6_set_phase1_method(method)


def get_phase1_method() -> int:
    """
    Get the current Phase 1 method for v6 map2alm.

    Returns:
        Current method: PHASE1_DFT, PHASE1_FFT_EQUATORIAL, or PHASE1_BLUESTEIN
    """
    lib = _get_lib()
    return lib.map2alm_v6_get_phase1_method()


def get_phase1_method_name() -> str:
    """Get the name of the current Phase 1 method."""
    method = get_phase1_method()
    names = {
        PHASE1_DFT: "DFT",
        PHASE1_FFT_EQUATORIAL: "FFT_EQUATORIAL",
        PHASE1_BLUESTEIN: "BLUESTEIN"
    }
    return names.get(method, f"UNKNOWN({method})")


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
                 storage_precision: str = None, recurrence_precision: str = None,
                 accumulation_mode: str = None):
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
            accumulation_mode: Accumulation mode - 'linear' (fast) or 'log' (numerically stable)
                               If None, uses global config. bf16 precision forces 'log'.
        """
        self.nside = nside
        self.l_max = l_max if l_max is not None else 3 * nside
        self.n_rings = 4 * nside - 1
        self.version = version
        # Use provided precision or fall back to global config
        self.storage_precision = storage_precision or config.storage_precision
        self.recurrence_precision = recurrence_precision or config.recurrence_precision
        self.accumulation_mode = accumulation_mode or config.get_effective_accumulation_mode()
        self._lib = _get_lib()

    def map2alm(self, maps: dict, spins: tuple = (0,), return_split: bool = False) -> dict:
        """
        Transform HEALPix maps to spherical harmonic coefficients.

        Args:
            maps: Dict with keys in {0, 2} for spin types.
                  For spin-0: Each value is array of shape [n_maps, n_rings, 4*nside]
                              or [n_maps, npix] in 1D HEALPix format.
                  For spin-2: Each value is array of shape [n_maps, n_rings, 4*nside, 2]
                              where last dimension is (Q, U).
            spins: Tuple of spins to compute, e.g., (0,) or (0, 2) or (2,)
            return_split: If True, return (alm_real, alm_imag) tuple instead of
                         complex array. This avoids the ~50ms combine overhead.
                         Use combine_to_complex() to convert to complex when needed.

        Returns:
            alm: Dict with same keys.
                 For spin-0: shape [n_maps, l_max+1, l_max+1] complex array (or split)
                 For spin-2: shape [n_maps, l_max+1, l_max+1, 2] where last dim is (E, B)
        """
        alm_out = {}
        use_f32 = (self.storage_precision == "float32")

        for s in spins:
            if s not in maps:
                continue

            if s == 2:
                # Spin-2 transform (Q, U) -> (E, B)
                if self.version != "v6":
                    raise NotImplementedError("Spin-2 only supported in v6")

                map_data = np.asarray(maps[s])
                if map_data.ndim == 3:
                    # Shape [n_maps, npix, 2] - need to reshape
                    raise ValueError("Spin-2 maps must have shape [n_maps, n_rings, 4*nside, 2]")
                if map_data.ndim != 4 or map_data.shape[-1] != 2:
                    raise ValueError(f"Spin-2 maps must have shape [n_maps, n_rings, 4*nside, 2], got {map_data.shape}")

                dtype = np.float32 if use_f32 else np.float64
                map_data = map_data.astype(dtype)
                n_maps = map_data.shape[0]

                alm_data = self._map2alm_spin2_v6(map_data, n_maps, return_split=return_split)
                alm_out[s] = alm_data
            elif s == 0:
                # Spin-0 transform
                dtype = np.float32 if use_f32 else np.float64
                map_data = np.asarray(maps[s], dtype=dtype)

                # Handle 1D HEALPix format
                if map_data.ndim == 1:
                    map_data = map_data.reshape(1, -1)
                if map_data.ndim == 2 and map_data.shape[1] == 12 * self.nside**2:
                    map_data = self._reshape_maps_to_2d(map_data, dtype=dtype)

                n_maps = map_data.shape[0]
                map_data = np.ascontiguousarray(map_data)

                # Use v6 with precision control
                alm_data = self._map2alm_v6(map_data, n_maps, return_split=return_split)

                alm_out[s] = alm_data
            else:
                raise NotImplementedError(f"Spin {s} not supported. Use spin=0 or spin=2.")

        return alm_out

    def _map2alm_v6(self, map_data: np.ndarray, n_maps: int, return_split: bool = False):
        """Run v6 transform with precision control (optimal warp-per-m, no atomics).

        Args:
            map_data: Input map array
            n_maps: Number of maps
            return_split: If True, return (alm_real, alm_imag) tuple instead of complex
        """
        import time
        import os
        timing_enabled = os.environ.get('SPHT_PY_TIMING') is not None

        use_f32 = (self.storage_precision == "float32")
        use_f32_recur = (self.recurrence_precision == "float32")
        use_log_mode = (self.accumulation_mode == "log")
        lp1 = self.l_max + 1

        if timing_enabled:
            t0 = time.perf_counter()

        # Sizes
        map_size = n_maps * self.n_rings * 4 * self.nside * map_data.itemsize
        alm_size = n_maps * lp1 * lp1 * map_data.itemsize

        # Use cached CUDA runtime
        cuda_rt = _get_cuda_rt()

        # Get cached pinned host memory for faster transfers
        h_alm_real, h_alm_imag = _get_pinned_buffers(alm_size, use_f32)

        if timing_enabled:
            t_alloc_host = time.perf_counter()

        # Get cached device buffers (no malloc/free per call)
        d_map = _get_device_buffer('m2a_map', map_size, use_f32)
        d_alm_real = _get_device_buffer('m2a_alm_real', alm_size, use_f32)
        d_alm_imag = _get_device_buffer('m2a_alm_imag', alm_size, use_f32)

        if timing_enabled:
            t_get_rt = time.perf_counter()
            t_alloc_dev = time.perf_counter()

        cuda_rt.cudaMemset(d_alm_real, 0, alm_size)
        cuda_rt.cudaMemset(d_alm_imag, 0, alm_size)

        if timing_enabled:
            t_memset = time.perf_counter()

        # Copy map to device
        cuda_rt.cudaMemcpy(d_map, map_data.ctypes.data_as(c_void_p),
                          map_size, 1)  # cudaMemcpyHostToDevice = 1

        if timing_enabled:
            t_h2d = time.perf_counter()

        # Select kernel based on precision combination and accumulation mode
        if use_log_mode:
            # LOG mode (logsumexp accumulation)
            if use_f32:
                if use_f32_recur:
                    self._lib.map2alm_cuda_v6_log_f32_f32(
                        self.nside, self.l_max, n_maps,
                        d_map, d_alm_real, d_alm_imag)
                else:
                    self._lib.map2alm_cuda_v6_log_f32_f64(
                        self.nside, self.l_max, n_maps,
                        d_map, d_alm_real, d_alm_imag)
            else:
                if use_f32_recur:
                    self._lib.map2alm_cuda_v6_log_f64_f32(
                        self.nside, self.l_max, n_maps,
                        d_map, d_alm_real, d_alm_imag)
                else:
                    self._lib.map2alm_cuda_v6_log_f64_f64(
                        self.nside, self.l_max, n_maps,
                        d_map, d_alm_real, d_alm_imag)
        else:
            # LINEAR mode (FMA accumulation - default)
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

        if timing_enabled:
            t_kernel = time.perf_counter()

        # Copy results to pinned host memory
        cuda_rt.cudaMemcpy(h_alm_real, d_alm_real, alm_size, 2)  # cudaMemcpyDeviceToHost = 2
        cuda_rt.cudaMemcpy(h_alm_imag, d_alm_imag, alm_size, 2)

        if timing_enabled:
            t_d2h = time.perf_counter()

        # No cudaFree - buffers are cached for reuse

        if timing_enabled:
            t_free = time.perf_counter()

        # Create numpy arrays from pinned memory
        c_type = ctypes.c_float if use_f32 else ctypes.c_double
        np_dtype = np.float32 if use_f32 else np.float64

        if return_split:
            # Copy from pinned memory to regular numpy arrays (avoid aliasing issues)
            alm_real_view = np.ctypeslib.as_array(ctypes.cast(h_alm_real, ctypes.POINTER(c_type)),
                                                  shape=(n_maps * lp1 * lp1,))
            alm_imag_view = np.ctypeslib.as_array(ctypes.cast(h_alm_imag, ctypes.POINTER(c_type)),
                                                  shape=(n_maps * lp1 * lp1,))
            alm_real = alm_real_view.copy().reshape(n_maps, lp1, lp1)
            alm_imag = alm_imag_view.copy().reshape(n_maps, lp1, lp1)

            if timing_enabled:
                t_combine = time.perf_counter()
                print(f"[PY_TIMING] alloc_host={1000*(t_alloc_host-t0):.2f}ms "
                      f"get_rt={1000*(t_get_rt-t_alloc_host):.2f}ms "
                      f"alloc_dev={1000*(t_alloc_dev-t_get_rt):.2f}ms "
                      f"memset={1000*(t_memset-t_alloc_dev):.2f}ms "
                      f"h2d={1000*(t_h2d-t_memset):.2f}ms "
                      f"kernel={1000*(t_kernel-t_h2d):.2f}ms "
                      f"d2h={1000*(t_d2h-t_kernel):.2f}ms "
                      f"free={1000*(t_free-t_d2h):.2f}ms "
                      f"copy={1000*(t_combine-t_free):.2f}ms "
                      f"TOTAL={1000*(t_combine-t0):.2f}ms")

            return (alm_real, alm_imag)
        else:
            # Combine to complex
            alm_real = np.ctypeslib.as_array(ctypes.cast(h_alm_real, ctypes.POINTER(c_type)),
                                              shape=(n_maps * lp1 * lp1,))
            alm_imag = np.ctypeslib.as_array(ctypes.cast(h_alm_imag, ctypes.POINTER(c_type)),
                                              shape=(n_maps * lp1 * lp1,))

            # Efficient interleaving: create [N, 2] array, then view as complex
            interleaved = np.empty((n_maps * lp1 * lp1, 2), dtype=alm_real.dtype)
            interleaved[:, 0] = alm_real
            interleaved[:, 1] = alm_imag
            complex_dtype = np.complex64 if use_f32 else np.complex128
            alm_data = interleaved.view(complex_dtype).reshape(n_maps, lp1, lp1)

            if timing_enabled:
                t_combine = time.perf_counter()
                print(f"[PY_TIMING] alloc_host={1000*(t_alloc_host-t0):.2f}ms "
                      f"get_rt={1000*(t_get_rt-t_alloc_host):.2f}ms "
                      f"alloc_dev={1000*(t_alloc_dev-t_get_rt):.2f}ms "
                      f"memset={1000*(t_memset-t_alloc_dev):.2f}ms "
                      f"h2d={1000*(t_h2d-t_memset):.2f}ms "
                      f"kernel={1000*(t_kernel-t_h2d):.2f}ms "
                      f"d2h={1000*(t_d2h-t_kernel):.2f}ms "
                      f"free={1000*(t_free-t_d2h):.2f}ms "
                      f"combine={1000*(t_combine-t_free):.2f}ms "
                      f"TOTAL={1000*(t_combine-t0):.2f}ms")

            return alm_data

    def _map2alm_spin2_v6(self, map_data: np.ndarray, n_maps: int, return_split: bool = False):
        """Run v6 spin-2 transform (Q, U) -> (E, B).

        Args:
            map_data: Input map array of shape [n_maps, n_rings, 4*nside, 2]
                      where last dimension is (Q, U)
            n_maps: Number of maps
            return_split: If True, return split real/imag arrays

        Returns:
            If return_split=False: complex array of shape [n_maps, l_max+1, l_max+1, 2]
                                   where last dim is (E, B)
            If return_split=True: tuple ((E_real, E_imag), (B_real, B_imag))
        """
        use_f32 = (self.storage_precision == "float32")
        use_f32_recur = (self.recurrence_precision == "float32")
        lp1 = self.l_max + 1

        # Extract Q and U maps - shape [n_maps, n_rings, max_pix]
        map_Q = np.ascontiguousarray(map_data[..., 0])
        map_U = np.ascontiguousarray(map_data[..., 1])

        # Sizes
        map_size = n_maps * self.n_rings * 4 * self.nside * map_Q.itemsize
        alm_size = n_maps * lp1 * lp1 * map_Q.itemsize

        # Use cached CUDA runtime
        cuda_rt = _get_cuda_rt()

        # Allocate device memory
        d_map_Q = ctypes.c_void_p()
        d_map_U = ctypes.c_void_p()
        d_alm_E_real = ctypes.c_void_p()
        d_alm_E_imag = ctypes.c_void_p()
        d_alm_B_real = ctypes.c_void_p()
        d_alm_B_imag = ctypes.c_void_p()

        cuda_rt.cudaMalloc(ctypes.byref(d_map_Q), map_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_map_U), map_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_alm_E_real), alm_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_alm_E_imag), alm_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_alm_B_real), alm_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_alm_B_imag), alm_size)

        cuda_rt.cudaMemset(d_alm_E_real, 0, alm_size)
        cuda_rt.cudaMemset(d_alm_E_imag, 0, alm_size)
        cuda_rt.cudaMemset(d_alm_B_real, 0, alm_size)
        cuda_rt.cudaMemset(d_alm_B_imag, 0, alm_size)

        # Copy maps to device
        cuda_rt.cudaMemcpy(d_map_Q, map_Q.ctypes.data_as(c_void_p), map_size, 1)
        cuda_rt.cudaMemcpy(d_map_U, map_U.ctypes.data_as(c_void_p), map_size, 1)

        try:
            # Select kernel based on precision combination
            if use_f32:
                if use_f32_recur:
                    self._lib.map2alm_cuda_v6_spin2_f32_f32(
                        self.nside, self.l_max, n_maps,
                        d_map_Q, d_map_U,
                        d_alm_E_real, d_alm_E_imag, d_alm_B_real, d_alm_B_imag)
                else:
                    self._lib.map2alm_cuda_v6_spin2_f32_f64(
                        self.nside, self.l_max, n_maps,
                        d_map_Q, d_map_U,
                        d_alm_E_real, d_alm_E_imag, d_alm_B_real, d_alm_B_imag)
            else:
                if use_f32_recur:
                    self._lib.map2alm_cuda_v6_spin2_f64_f32(
                        self.nside, self.l_max, n_maps,
                        d_map_Q, d_map_U,
                        d_alm_E_real, d_alm_E_imag, d_alm_B_real, d_alm_B_imag)
                else:
                    self._lib.map2alm_cuda_v6_spin2_f64_f64(
                        self.nside, self.l_max, n_maps,
                        d_map_Q, d_map_U,
                        d_alm_E_real, d_alm_E_imag, d_alm_B_real, d_alm_B_imag)

            # Allocate host arrays
            np_dtype = np.float32 if use_f32 else np.float64
            E_real = np.zeros((n_maps, lp1, lp1), dtype=np_dtype)
            E_imag = np.zeros((n_maps, lp1, lp1), dtype=np_dtype)
            B_real = np.zeros((n_maps, lp1, lp1), dtype=np_dtype)
            B_imag = np.zeros((n_maps, lp1, lp1), dtype=np_dtype)

            # Copy results back
            cuda_rt.cudaMemcpy(E_real.ctypes.data_as(c_void_p), d_alm_E_real, alm_size, 2)
            cuda_rt.cudaMemcpy(E_imag.ctypes.data_as(c_void_p), d_alm_E_imag, alm_size, 2)
            cuda_rt.cudaMemcpy(B_real.ctypes.data_as(c_void_p), d_alm_B_real, alm_size, 2)
            cuda_rt.cudaMemcpy(B_imag.ctypes.data_as(c_void_p), d_alm_B_imag, alm_size, 2)

        finally:
            cuda_rt.cudaFree(d_map_Q)
            cuda_rt.cudaFree(d_map_U)
            cuda_rt.cudaFree(d_alm_E_real)
            cuda_rt.cudaFree(d_alm_E_imag)
            cuda_rt.cudaFree(d_alm_B_real)
            cuda_rt.cudaFree(d_alm_B_imag)

        # Apply JAX-compatible post-processing:
        # JAX does: alm[2] *= -1, alm[-2] *= 1j
        # E_final = -E_raw = -(E_re + i*E_im)
        # B_final = i*B_raw = i*(B_re + i*B_im) = -B_im + i*B_re
        E_real_final = -E_real
        E_imag_final = -E_imag
        B_real_final = -B_imag
        B_imag_final = B_real

        if return_split:
            return ((E_real_final, E_imag_final), (B_real_final, B_imag_final))
        else:
            # Combine to complex arrays with shape [n_maps, lp1, lp1, 2]
            complex_dtype = np.complex64 if use_f32 else np.complex128
            alm_E = (E_real_final + 1j * E_imag_final).astype(complex_dtype)
            alm_B = (B_real_final + 1j * B_imag_final).astype(complex_dtype)
            # Stack E and B along last dimension
            alm_out = np.stack([alm_E, alm_B], axis=-1)
            return alm_out

    def alm2map(self, alm: dict, spins: tuple = (0,)) -> dict:
        """
        Transform spherical harmonic coefficients to HEALPix maps.

        Args:
            alm: Dict with keys in {0, 2} for spin types.
                 For spin-0: Each value is array of shape [n_maps, l_max+1, l_max+1]
                 For spin-2: Each value is array of shape [n_maps, l_max+1, l_max+1, 2]
                             where last dimension is (E, B)
            spins: Tuple of spins to compute, e.g., (0,) or (0, 2) or (2,)

        Returns:
            maps: Dict with same keys.
                  For spin-0: shape [n_maps, n_rings, 4*nside]
                  For spin-2: shape [n_maps, n_rings, 4*nside, 2] where last dim is (Q, U)
        """
        maps_out = {}
        use_f32 = (self.storage_precision == "float32")

        for s in spins:
            if s not in alm:
                continue

            if s == 2:
                # Spin-2 transform (E, B) -> (Q, U)
                if self.version != "v6":
                    raise NotImplementedError("Spin-2 only supported in v6")

                alm_data = np.asarray(alm[s])
                if alm_data.ndim != 4 or alm_data.shape[-1] != 2:
                    raise ValueError(f"Spin-2 alm must have shape [n_maps, l_max+1, l_max+1, 2], got {alm_data.shape}")

                map_data = self._alm2map_spin2_v6(alm_data)
                maps_out[s] = map_data
            elif s == 0:
                # Spin-0 transform using v6
                map_data = self._alm2map_v6(alm[s])
                maps_out[s] = map_data
            else:
                raise NotImplementedError(f"Spin {s} not supported. Use spin=0 or spin=2.")

        return maps_out

    def _alm2map_v6(self, alm_data: np.ndarray) -> np.ndarray:
        """Run v6 alm2map transform with precision control (optimal warp-per-m).

        Args:
            alm_data: Input alm array

        Returns:
            map_out: Output map array
        """
        import time
        import os
        timing_enabled = os.environ.get('SPHT_PY_TIMING') is not None

        use_f32 = (self.storage_precision == "float32")
        use_f32_recur = (self.recurrence_precision == "float32")
        lp1 = self.l_max + 1

        if timing_enabled:
            t0 = time.perf_counter()

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

        if timing_enabled:
            t_prep = time.perf_counter()

        # Sizes
        alm_size = n_maps * lp1 * lp1 * alm_real.itemsize
        map_size = n_maps * self.n_rings * 4 * self.nside * alm_real.itemsize

        # Use cached CUDA runtime
        cuda_rt = _get_cuda_rt()

        if timing_enabled:
            t_get_rt = time.perf_counter()

        # Get cached device buffers (no malloc/free per call)
        d_alm_real = _get_device_buffer('a2m_alm_real', alm_size, use_f32)
        d_alm_imag = _get_device_buffer('a2m_alm_imag', alm_size, use_f32)
        d_map = _get_device_buffer('a2m_map', map_size, use_f32)

        if timing_enabled:
            t_alloc = time.perf_counter()

        # Copy alm to device
        cuda_rt.cudaMemcpy(d_alm_real, alm_real.ctypes.data_as(c_void_p),
                          alm_size, 1)  # cudaMemcpyHostToDevice = 1
        cuda_rt.cudaMemcpy(d_alm_imag, alm_imag.ctypes.data_as(c_void_p),
                          alm_size, 1)

        if timing_enabled:
            t_h2d = time.perf_counter()

        # Select kernel based on precision combination
        if use_f32:
            if use_f32_recur:
                self._lib.alm2map_cuda_v6_f32_f32(
                    self.nside, self.l_max, n_maps,
                    d_alm_real, d_alm_imag, d_map)
            else:
                self._lib.alm2map_cuda_v6_f32_f64(
                    self.nside, self.l_max, n_maps,
                    d_alm_real, d_alm_imag, d_map)
        else:
            if use_f32_recur:
                self._lib.alm2map_cuda_v6_f64_f32(
                    self.nside, self.l_max, n_maps,
                    d_alm_real, d_alm_imag, d_map)
            else:
                self._lib.alm2map_cuda_v6_f64_f64(
                    self.nside, self.l_max, n_maps,
                    d_alm_real, d_alm_imag, d_map)

        if timing_enabled:
            t_kernel = time.perf_counter()

        # Allocate output
        map_shape = (n_maps, self.n_rings, 4 * self.nside)
        map_out = np.zeros(map_shape, dtype=map_dtype)

        # Copy result back
        cuda_rt.cudaMemcpy(map_out.ctypes.data_as(c_void_p),
                          d_map, map_size, 2)  # cudaMemcpyDeviceToHost = 2

        if timing_enabled:
            t_d2h = time.perf_counter()

        # No cudaFree - buffers are cached for reuse

        if timing_enabled:
            t_free = time.perf_counter()
            print(f"[PY_TIMING alm2map] prep={1000*(t_prep-t0):.2f}ms "
                  f"get_rt={1000*(t_get_rt-t_prep):.2f}ms "
                  f"alloc={1000*(t_alloc-t_get_rt):.2f}ms "
                  f"h2d={1000*(t_h2d-t_alloc):.2f}ms "
                  f"kernel={1000*(t_kernel-t_h2d):.2f}ms "
                  f"d2h={1000*(t_d2h-t_kernel):.2f}ms "
                  f"free={1000*(t_free-t_d2h):.2f}ms "
                  f"TOTAL={1000*(t_free-t0):.2f}ms")

        return map_out

    def _alm2map_spin2_v6(self, alm_data: np.ndarray) -> np.ndarray:
        """Run v6 spin-2 alm2map transform (E, B) -> (Q, U).

        Args:
            alm_data: Input alm array of shape [n_maps, l_max+1, l_max+1, 2]
                      where last dimension is (E, B) in JAX convention:
                      - E_alm = -E_raw (from map2alm post-processing)
                      - B_alm = i*B_raw (from map2alm post-processing)

        Returns:
            map_out: Output map array of shape [n_maps, n_rings, 4*nside, 2]
                     where last dimension is (Q, U)
        """
        use_f32 = (self.storage_precision == "float32")
        use_f32_recur = (self.recurrence_precision == "float32")
        lp1 = self.l_max + 1

        # Convert to appropriate dtype
        if use_f32:
            alm_data = np.asarray(alm_data, dtype=np.complex64)
            np_dtype = np.float32
        else:
            alm_data = np.asarray(alm_data, dtype=np.complex128)
            np_dtype = np.float64

        if alm_data.ndim == 3:
            alm_data = alm_data.reshape(1, alm_data.shape[0], alm_data.shape[1], alm_data.shape[2])

        n_maps = alm_data.shape[0]

        # Extract E and B alm - shape [n_maps, lp1, lp1]
        # Input is in JAX convention: E_alm = -E_raw, B_alm = i*B_raw
        alm_E = np.ascontiguousarray(alm_data[..., 0])
        alm_B = np.ascontiguousarray(alm_data[..., 1])

        # Pass JAX-formatted alm directly to CUDA kernel
        # The kernel will handle the conventions internally
        E_real = np.ascontiguousarray(alm_E.real.astype(np_dtype))
        E_imag = np.ascontiguousarray(alm_E.imag.astype(np_dtype))
        B_real = np.ascontiguousarray(alm_B.real.astype(np_dtype))
        B_imag = np.ascontiguousarray(alm_B.imag.astype(np_dtype))

        # Sizes
        alm_size = n_maps * lp1 * lp1 * E_real.itemsize
        map_size = n_maps * self.n_rings * 4 * self.nside * E_real.itemsize

        # Use cached CUDA runtime
        cuda_rt = _get_cuda_rt()

        # Allocate device memory
        d_alm_E_real = ctypes.c_void_p()
        d_alm_E_imag = ctypes.c_void_p()
        d_alm_B_real = ctypes.c_void_p()
        d_alm_B_imag = ctypes.c_void_p()
        d_map_Q = ctypes.c_void_p()
        d_map_U = ctypes.c_void_p()

        cuda_rt.cudaMalloc(ctypes.byref(d_alm_E_real), alm_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_alm_E_imag), alm_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_alm_B_real), alm_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_alm_B_imag), alm_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_map_Q), map_size)
        cuda_rt.cudaMalloc(ctypes.byref(d_map_U), map_size)

        # Copy alm to device
        cuda_rt.cudaMemcpy(d_alm_E_real, E_real.ctypes.data_as(c_void_p), alm_size, 1)
        cuda_rt.cudaMemcpy(d_alm_E_imag, E_imag.ctypes.data_as(c_void_p), alm_size, 1)
        cuda_rt.cudaMemcpy(d_alm_B_real, B_real.ctypes.data_as(c_void_p), alm_size, 1)
        cuda_rt.cudaMemcpy(d_alm_B_imag, B_imag.ctypes.data_as(c_void_p), alm_size, 1)

        try:
            # Select kernel based on precision combination
            if use_f32:
                if use_f32_recur:
                    self._lib.alm2map_cuda_v6_spin2_f32_f32(
                        self.nside, self.l_max, n_maps,
                        d_alm_E_real, d_alm_E_imag, d_alm_B_real, d_alm_B_imag,
                        d_map_Q, d_map_U)
                else:
                    self._lib.alm2map_cuda_v6_spin2_f32_f64(
                        self.nside, self.l_max, n_maps,
                        d_alm_E_real, d_alm_E_imag, d_alm_B_real, d_alm_B_imag,
                        d_map_Q, d_map_U)
            else:
                if use_f32_recur:
                    self._lib.alm2map_cuda_v6_spin2_f64_f32(
                        self.nside, self.l_max, n_maps,
                        d_alm_E_real, d_alm_E_imag, d_alm_B_real, d_alm_B_imag,
                        d_map_Q, d_map_U)
                else:
                    self._lib.alm2map_cuda_v6_spin2_f64_f64(
                        self.nside, self.l_max, n_maps,
                        d_alm_E_real, d_alm_E_imag, d_alm_B_real, d_alm_B_imag,
                        d_map_Q, d_map_U)

            # Allocate output arrays
            map_Q = np.zeros((n_maps, self.n_rings, 4 * self.nside), dtype=np_dtype)
            map_U = np.zeros((n_maps, self.n_rings, 4 * self.nside), dtype=np_dtype)

            # Copy results back
            cuda_rt.cudaMemcpy(map_Q.ctypes.data_as(c_void_p), d_map_Q, map_size, 2)
            cuda_rt.cudaMemcpy(map_U.ctypes.data_as(c_void_p), d_map_U, map_size, 2)

        finally:
            cuda_rt.cudaFree(d_alm_E_real)
            cuda_rt.cudaFree(d_alm_E_imag)
            cuda_rt.cudaFree(d_alm_B_real)
            cuda_rt.cudaFree(d_alm_B_imag)
            cuda_rt.cudaFree(d_map_Q)
            cuda_rt.cudaFree(d_map_U)

        # Stack Q and U along last dimension
        map_out = np.stack([map_Q, map_U], axis=-1)
        return map_out

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


def combine_to_complex(alm_real: np.ndarray, alm_imag: np.ndarray) -> np.ndarray:
    """
    Combine split real/imag alm arrays into complex array.

    This is a helper function to convert split arrays (returned by map2alm with
    return_split=True) into complex format for comparison with JAX or other uses.

    Args:
        alm_real: Real part array, shape [n_maps, l_max+1, l_max+1]
        alm_imag: Imaginary part array, shape [n_maps, l_max+1, l_max+1]

    Returns:
        alm: Complex array [n_maps, l_max+1, l_max+1]
             dtype is complex64 for float32 input, complex128 for float64

    Example:
        # Get split arrays
        alm_real, alm_imag = spht.map2alm({0: maps}, spins=(0,), return_split=True)[0]

        # Combine when needed for JAX comparison
        alm_complex = combine_to_complex(alm_real, alm_imag)
    """
    # Efficient interleaving: create [N, 2] array, then view as complex
    flat_shape = alm_real.size
    interleaved = np.empty((flat_shape, 2), dtype=alm_real.dtype)
    interleaved[:, 0] = alm_real.ravel()
    interleaved[:, 1] = alm_imag.ravel()
    complex_dtype = np.complex64 if alm_real.dtype == np.float32 else np.complex128
    return interleaved.view(complex_dtype).reshape(alm_real.shape)


# =============================================================================
# Power Spectrum Functions
# =============================================================================

def alm2cl_cuda(l_max: int, alm_real: np.ndarray, alm_imag: np.ndarray,
                alm2_real: np.ndarray = None, alm2_imag: np.ndarray = None) -> np.ndarray:
    """
    Compute angular power spectrum C_l from alm coefficients on GPU.

    Args:
        l_max: Maximum multipole
        alm_real: Real part of alm, shape [n_fields, l_max+1, l_max+1]
        alm_imag: Imaginary part of alm, shape [n_fields, l_max+1, l_max+1]
        alm2_real: Optional second alm real part for cross-spectrum
        alm2_imag: Optional second alm imaginary part for cross-spectrum

    Returns:
        cl: Power spectra, shape [n_fields, l_max+1] for auto-spectra
            or [n_fields, l_max+1] for cross-spectra
    """
    lib = _get_lib()
    cuda_rt = _get_cuda_rt()

    use_f32 = (alm_real.dtype == np.float32)
    n_fields = alm_real.shape[0]
    lp1 = l_max + 1

    # Ensure contiguous
    alm_real = np.ascontiguousarray(alm_real)
    alm_imag = np.ascontiguousarray(alm_imag)

    # Sizes
    alm_size = n_fields * lp1 * lp1 * alm_real.itemsize
    cl_size = n_fields * lp1 * alm_real.itemsize

    # Allocate device memory
    d_alm_real = ctypes.c_void_p()
    d_alm_imag = ctypes.c_void_p()
    d_cl = ctypes.c_void_p()

    cuda_rt.cudaMalloc(ctypes.byref(d_alm_real), alm_size)
    cuda_rt.cudaMalloc(ctypes.byref(d_alm_imag), alm_size)
    cuda_rt.cudaMalloc(ctypes.byref(d_cl), cl_size)

    # Copy alm to device
    cuda_rt.cudaMemcpy(d_alm_real, alm_real.ctypes.data_as(c_void_p), alm_size, 1)
    cuda_rt.cudaMemcpy(d_alm_imag, alm_imag.ctypes.data_as(c_void_p), alm_size, 1)

    try:
        if alm2_real is None:
            # Auto-spectrum
            if use_f32:
                lib.alm2cl_cuda_auto_f32(l_max, n_fields, d_alm_real, d_alm_imag, d_cl)
            else:
                lib.alm2cl_cuda_auto_f64(l_max, n_fields, d_alm_real, d_alm_imag, d_cl)
        else:
            # Cross-spectrum
            alm2_real = np.ascontiguousarray(alm2_real)
            alm2_imag = np.ascontiguousarray(alm2_imag)

            d_alm2_real = ctypes.c_void_p()
            d_alm2_imag = ctypes.c_void_p()
            cuda_rt.cudaMalloc(ctypes.byref(d_alm2_real), alm_size)
            cuda_rt.cudaMalloc(ctypes.byref(d_alm2_imag), alm_size)
            cuda_rt.cudaMemcpy(d_alm2_real, alm2_real.ctypes.data_as(c_void_p), alm_size, 1)
            cuda_rt.cudaMemcpy(d_alm2_imag, alm2_imag.ctypes.data_as(c_void_p), alm_size, 1)

            try:
                if use_f32:
                    lib.alm2cl_cuda_cross_f32(l_max, n_fields, d_alm_real, d_alm_imag,
                                              d_alm2_real, d_alm2_imag, d_cl)
                else:
                    lib.alm2cl_cuda_cross_f64(l_max, n_fields, d_alm_real, d_alm_imag,
                                              d_alm2_real, d_alm2_imag, d_cl)
            finally:
                cuda_rt.cudaFree(d_alm2_real)
                cuda_rt.cudaFree(d_alm2_imag)

        # Copy result back
        cl = np.zeros((n_fields, lp1), dtype=alm_real.dtype)
        cuda_rt.cudaMemcpy(cl.ctypes.data_as(c_void_p), d_cl, cl_size, 2)

    finally:
        cuda_rt.cudaFree(d_alm_real)
        cuda_rt.cudaFree(d_alm_imag)
        cuda_rt.cudaFree(d_cl)

    return cl


def alm2cl_cuda_all_pairs(l_max: int, alm_real: np.ndarray, alm_imag: np.ndarray) -> np.ndarray:
    """
    Compute all auto and cross power spectra for multiple fields.

    For n_fields, computes n_fields*(n_fields+1)/2 spectra.

    Args:
        l_max: Maximum multipole
        alm_real: Real part of alm, shape [n_fields, l_max+1, l_max+1]
        alm_imag: Imaginary part of alm, shape [n_fields, l_max+1, l_max+1]

    Returns:
        cl: Power spectra, shape [n_pairs, l_max+1]
            Pairs are ordered as: (0,0), (0,1), ..., (0,n-1), (1,1), (1,2), ..., (n-1,n-1)
    """
    lib = _get_lib()
    cuda_rt = _get_cuda_rt()

    use_f32 = (alm_real.dtype == np.float32)
    n_fields = alm_real.shape[0]
    n_pairs = n_fields * (n_fields + 1) // 2
    lp1 = l_max + 1

    # Ensure contiguous
    alm_real = np.ascontiguousarray(alm_real)
    alm_imag = np.ascontiguousarray(alm_imag)

    # Sizes
    alm_size = n_fields * lp1 * lp1 * alm_real.itemsize
    cl_size = n_pairs * lp1 * alm_real.itemsize

    # Allocate device memory
    d_alm_real = ctypes.c_void_p()
    d_alm_imag = ctypes.c_void_p()
    d_cl = ctypes.c_void_p()

    cuda_rt.cudaMalloc(ctypes.byref(d_alm_real), alm_size)
    cuda_rt.cudaMalloc(ctypes.byref(d_alm_imag), alm_size)
    cuda_rt.cudaMalloc(ctypes.byref(d_cl), cl_size)

    # Copy alm to device
    cuda_rt.cudaMemcpy(d_alm_real, alm_real.ctypes.data_as(c_void_p), alm_size, 1)
    cuda_rt.cudaMemcpy(d_alm_imag, alm_imag.ctypes.data_as(c_void_p), alm_size, 1)

    try:
        if use_f32:
            lib.alm2cl_cuda_all_pairs_f32(l_max, n_fields, d_alm_real, d_alm_imag, d_cl)
        else:
            lib.alm2cl_cuda_all_pairs_f64(l_max, n_fields, d_alm_real, d_alm_imag, d_cl)

        # Copy result back
        cl = np.zeros((n_pairs, lp1), dtype=alm_real.dtype)
        cuda_rt.cudaMemcpy(cl.ctypes.data_as(c_void_p), d_cl, cl_size, 2)

    finally:
        cuda_rt.cudaFree(d_alm_real)
        cuda_rt.cudaFree(d_alm_imag)
        cuda_rt.cudaFree(d_cl)

    return cl


def get_pair_indices(n_fields: int) -> list:
    """
    Get list of (field1, field2) tuples for pair indexing.

    Returns pairs in order: (0,0), (0,1), ..., (0,n-1), (1,1), (1,2), ..., (n-1,n-1)
    """
    pairs = []
    for f1 in range(n_fields):
        for f2 in range(f1, n_fields):
            pairs.append((f1, f2))
    return pairs


def _parse_map_input(maps, nside):
    """
    Parse mixed spin-0/spin-2 map input into separate batches.

    Input format examples:
        [T]                      -> spin-0: [T], spin-2: []
        [T, [Q, U]]              -> spin-0: [T], spin-2: [[Q, U]]
        [[Q1, U1], [Q2, U2]]     -> spin-0: [], spin-2: [[Q1, U1], [Q2, U2]]
        [T1, [Q1, U1], T2, [Q2, U2]] -> spin-0: [T1, T2], spin-2: [[Q1, U1], [Q2, U2]]

    Returns:
        spin0_maps: np.ndarray of shape [n_spin0, n_rings, 4*nside] or None
        spin2_maps: np.ndarray of shape [n_spin2, n_rings, 4*nside, 2] or None
        field_labels: List of ('T', idx) or ('E', idx) or ('B', idx) tuples
        field_indices: Dict mapping label to index in concatenated alm array
    """
    n_rings = 4 * nside - 1
    max_pix = 4 * nside

    spin0_list = []
    spin2_list = []
    field_labels = []

    spin0_idx = 0
    spin2_idx = 0

    for item in maps:
        item = np.asarray(item)

        # Check if this is a spin-2 map (has 2 components)
        if item.ndim == 3 and item.shape[0] == 2:
            # Spin-2: [2, n_rings, max_pix] -> Q, U
            Q = item[0]
            U = item[1]
            spin2_list.append(np.stack([Q, U], axis=-1))  # [n_rings, max_pix, 2]
            field_labels.append(('E', spin2_idx))
            field_labels.append(('B', spin2_idx))
            spin2_idx += 1
        elif item.ndim == 2:
            # Spin-0: [n_rings, max_pix]
            spin0_list.append(item)
            field_labels.append(('T', spin0_idx))
            spin0_idx += 1
        elif item.ndim == 1:
            # 1D HEALPix format [npix]
            if item.shape[0] == 12 * nside**2:
                # Need to reshape to ring format
                raise ValueError("1D HEALPix format not supported in map2cl. "
                               "Please use 2D ring format [n_rings, 4*nside].")
            else:
                raise ValueError(f"Unexpected 1D array shape: {item.shape}")
        else:
            raise ValueError(f"Unexpected array shape: {item.shape}. "
                           "Expected [n_rings, max_pix] for spin-0 or "
                           "[2, n_rings, max_pix] for spin-2.")

    spin0_maps = np.stack(spin0_list, axis=0) if spin0_list else None
    spin2_maps = np.stack(spin2_list, axis=0) if spin2_list else None

    # Build field indices
    field_indices = {}
    idx = 0
    for label in field_labels:
        field_indices[label] = idx
        idx += 1

    return spin0_maps, spin2_maps, field_labels, field_indices


class SPHTCudaMap2Cl:
    """
    High-level interface for computing power spectra directly from maps.

    This class handles mixed spin-0 and spin-2 inputs, automatically batching
    transforms and computing all auto and cross power spectra.
    """

    def __init__(self, nside: int, l_max: int = None, version: str = "v6",
                 storage_precision: str = None, recurrence_precision: str = None):
        """
        Initialize Map2Cl context.

        Args:
            nside: HEALPix nside parameter
            l_max: Maximum multipole (default: 3*nside)
            version: SPHT version to use (default: "v6")
            storage_precision: 'float64' or 'float32' (default: global config)
            recurrence_precision: 'float64' or 'float32' (default: global config)
        """
        self.nside = nside
        self.l_max = l_max if l_max is not None else 3 * nside
        self.n_rings = 4 * nside - 1
        self.version = version
        self.storage_precision = storage_precision or config.storage_precision
        self.recurrence_precision = recurrence_precision or config.recurrence_precision
        self._spht = SPHTCuda(nside, self.l_max, version,
                              storage_precision, recurrence_precision)
        self._lib = _get_lib()

    def map2cl(self, maps: list, return_dict: bool = True):
        """
        Compute all auto and cross power spectra from maps.

        Args:
            maps: List of maps in mixed format:
                  - Spin-0 (T): array of shape [n_rings, 4*nside]
                  - Spin-2 (Q,U): array of shape [2, n_rings, 4*nside]
                  Examples:
                    [T]  - single temperature map
                    [T, [Q, U]]  - temperature and polarization
                    [[Q1, U1], [Q2, U2]]  - two polarization maps
                    [T1, [Q1, U1], T2, [Q2, U2]]  - two full TQU sets
            return_dict: If True, return dict with spectrum labels as keys.
                        If False, return raw array and field labels.

        Returns:
            If return_dict=True:
                Dict with keys like 'TT', 'TE', 'TB', 'EE', 'EB', 'BB',
                'T1T2', 'E1E2', etc. Values are C_l arrays of shape [l_max+1]
            If return_dict=False:
                Tuple of (cl_array, field_labels, pair_indices)
                where cl_array has shape [n_pairs, l_max+1]
        """
        use_f32 = (self.storage_precision == "float32")
        dtype = np.float32 if use_f32 else np.float64

        # Parse input
        spin0_maps, spin2_maps, field_labels, field_indices = _parse_map_input(maps, self.nside)

        # Compute alm for all fields
        all_alm_real = []
        all_alm_imag = []

        # Process spin-0 maps
        if spin0_maps is not None:
            spin0_maps = spin0_maps.astype(dtype)
            n_spin0 = spin0_maps.shape[0]
            alm_result = self._spht.map2alm({0: spin0_maps}, spins=(0,), return_split=True)
            alm_real, alm_imag = alm_result[0]
            for i in range(n_spin0):
                all_alm_real.append(alm_real[i])
                all_alm_imag.append(alm_imag[i])

        # Process spin-2 maps
        if spin2_maps is not None:
            spin2_maps = spin2_maps.astype(dtype)
            n_spin2 = spin2_maps.shape[0]
            alm_result = self._spht.map2alm({2: spin2_maps}, spins=(2,), return_split=True)
            (E_real, E_imag), (B_real, B_imag) = alm_result[2]
            for i in range(n_spin2):
                all_alm_real.append(E_real[i])
                all_alm_imag.append(E_imag[i])
                all_alm_real.append(B_real[i])
                all_alm_imag.append(B_imag[i])

        # Stack all alm arrays
        n_fields = len(all_alm_real)
        all_alm_real = np.stack(all_alm_real, axis=0)
        all_alm_imag = np.stack(all_alm_imag, axis=0)

        # Compute all power spectra on GPU
        cl_array = alm2cl_cuda_all_pairs(self.l_max, all_alm_real, all_alm_imag)

        if not return_dict:
            pairs = get_pair_indices(n_fields)
            return cl_array, field_labels, pairs

        # Build labeled dictionary
        pairs = get_pair_indices(n_fields)
        cl_dict = {}

        for pair_idx, (f1, f2) in enumerate(pairs):
            label1 = field_labels[f1]
            label2 = field_labels[f2]

            # Create spectrum label
            type1, idx1 = label1
            type2, idx2 = label2

            if idx1 == idx2:
                # Same map set
                key = f"{type1}{type2}"
            else:
                # Cross between different map sets
                key = f"{type1}{idx1+1}{type2}{idx2+1}"

            # Handle duplicate keys (e.g., multiple T maps)
            if key in cl_dict:
                # Find a unique key
                base_key = key
                counter = 2
                while key in cl_dict:
                    key = f"{base_key}_{counter}"
                    counter += 1

            cl_dict[key] = cl_array[pair_idx]

        return cl_dict

    def map2alm(self, maps: list, return_split: bool = False):
        """
        Compute alm from maps (convenience wrapper).

        Args:
            maps: List of maps in mixed format (same as map2cl)
            return_split: If True, return split real/imag arrays

        Returns:
            Dict mapping field labels to alm arrays
        """
        use_f32 = (self.storage_precision == "float32")
        dtype = np.float32 if use_f32 else np.float64

        spin0_maps, spin2_maps, field_labels, field_indices = _parse_map_input(maps, self.nside)

        result = {}

        if spin0_maps is not None:
            spin0_maps = spin0_maps.astype(dtype)
            alm_result = self._spht.map2alm({0: spin0_maps}, spins=(0,), return_split=return_split)
            if return_split:
                alm_real, alm_imag = alm_result[0]
                for i, label in enumerate(field_labels):
                    if label[0] == 'T':
                        result[label] = (alm_real[i], alm_imag[i])
            else:
                for i, label in enumerate(field_labels):
                    if label[0] == 'T':
                        result[label] = alm_result[0][i]

        if spin2_maps is not None:
            spin2_maps = spin2_maps.astype(dtype)
            alm_result = self._spht.map2alm({2: spin2_maps}, spins=(2,), return_split=return_split)
            if return_split:
                (E_real, E_imag), (B_real, B_imag) = alm_result[2]
                eb_idx = 0
                for label in field_labels:
                    if label[0] == 'E':
                        idx = label[1]
                        result[('E', idx)] = (E_real[idx], E_imag[idx])
                    elif label[0] == 'B':
                        idx = label[1]
                        result[('B', idx)] = (B_real[idx], B_imag[idx])
            else:
                for label in field_labels:
                    if label[0] == 'E':
                        idx = label[1]
                        result[('E', idx)] = alm_result[2][idx, :, :, 0]
                    elif label[0] == 'B':
                        idx = label[1]
                        result[('B', idx)] = alm_result[2][idx, :, :, 1]

        return result


def map2cl_cuda(nside: int, l_max: int, maps: list, **kwargs) -> dict:
    """
    Convenience function to compute power spectra from maps.

    Args:
        nside: HEALPix nside parameter
        l_max: Maximum multipole
        maps: List of maps in mixed format:
              - Spin-0 (T): array of shape [n_rings, 4*nside]
              - Spin-2 (Q,U): array of shape [2, n_rings, 4*nside]
        **kwargs: Additional arguments passed to SPHTCudaMap2Cl

    Returns:
        Dict with spectrum labels as keys (e.g., 'TT', 'TE', 'EE', 'BB')
        and C_l arrays as values.

    Example:
        # Single temperature map
        cl = map2cl_cuda(nside, l_max, [T_map])
        # cl['TT'] is the temperature power spectrum

        # Temperature and polarization
        cl = map2cl_cuda(nside, l_max, [T_map, [Q_map, U_map]])
        # cl['TT'], cl['TE'], cl['TB'], cl['EE'], cl['EB'], cl['BB']

        # Two map sets for cross-correlation
        cl = map2cl_cuda(nside, l_max, [T1, [Q1, U1], T2, [Q2, U2]])
        # Includes all auto and cross spectra
    """
    m2cl = SPHTCudaMap2Cl(nside, l_max, **kwargs)
    return m2cl.map2cl(maps)

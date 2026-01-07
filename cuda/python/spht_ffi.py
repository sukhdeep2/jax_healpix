"""
SPHT CUDA FFI - Foreign Function Interface for CUDA library

This module handles loading the CUDA shared library, setting up function
signatures, and managing device/pinned memory buffers.
"""

import ctypes
from ctypes import c_int, c_float, c_double, c_void_p, POINTER, Structure
from pathlib import Path


# ============================================================================
# Configuration (JAX-style precision flags)
# ============================================================================

# Valid precision options
VALID_STORAGE_PRECISIONS = ("float64", "float32", "bfloat16")
VALID_RECURRENCE_PRECISIONS = ("float64", "float32", "bfloat16")
VALID_ACCUMULATION_MODES = ("linear", "log")


class _SPHTConfig:
    """Global configuration for SPHT CUDA precision settings."""

    def __init__(self):
        self._storage_precision = "float64"
        self._recurrence_precision = "float64"
        self._accumulation_mode = "linear"

    @property
    def storage_precision(self) -> str:
        """Precision for map/alm storage: 'float64', 'float32', or 'bfloat16'"""
        return self._storage_precision

    @property
    def recurrence_precision(self) -> str:
        """Precision for Ylm recurrence: 'float64', 'float32', or 'bfloat16'"""
        return self._recurrence_precision

    @property
    def accumulation_mode(self) -> str:
        """Accumulation mode: 'linear' (fast FMA) or 'log' (logsumexp)"""
        return self._accumulation_mode

    def update(self, key: str, value):
        """Update a configuration value."""
        if key == "spht_storage_precision":
            if value not in VALID_STORAGE_PRECISIONS:
                raise ValueError(f"storage_precision must be one of {VALID_STORAGE_PRECISIONS}, got {value}")
            self._storage_precision = value
            # bf16 requires log accumulation
            if value == "bfloat16" and self._accumulation_mode == "linear":
                self._accumulation_mode = "log"
        elif key == "spht_recurrence_precision":
            if value not in VALID_RECURRENCE_PRECISIONS:
                raise ValueError(f"recurrence_precision must be one of {VALID_RECURRENCE_PRECISIONS}, got {value}")
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

    def get_effective_accumulation_mode(self) -> str:
        """Get the effective accumulation mode (may be forced by precision)."""
        # bf16 always requires log mode
        if self._storage_precision == "bfloat16" or self._recurrence_precision == "bfloat16":
            return "log"
        return self._accumulation_mode


# Global config instance
config = _SPHTConfig()


def set_precision(storage: str = None, recurrence: str = None, accumulation: str = None):
    """Set precision for SPHT CUDA computations.

    Args:
        storage: Precision for map/alm data - 'float64', 'float32', or 'bfloat16'
        recurrence: Precision for Ylm recurrence - 'float64', 'float32', or 'bfloat16'
        accumulation: Accumulation mode - 'linear' (fast, default) or 'log' (numerically stable)

    Note:
        - bfloat16 requires accumulation='log' (set automatically)
        - 'linear' mode uses fast FMA-based accumulation
        - 'log' mode uses logsumexp (5-10x slower but handles extreme dynamic range)
    """
    if storage is not None:
        config.update("spht_storage_precision", storage)
    if recurrence is not None:
        config.update("spht_recurrence_precision", recurrence)
    if accumulation is not None:
        config.update("spht_accumulation_mode", accumulation)


# ============================================================================
# Type Definitions
# ============================================================================

class Complex64(Structure):
    """Complex number with double precision components."""
    _fields_ = [("x", c_double), ("y", c_double)]


class Complex32(Structure):
    """Complex number with single precision components."""
    _fields_ = [("x", c_float), ("y", c_float)]


# ============================================================================
# Library Loading
# ============================================================================

# Global library handles (lazy loaded)
_lib = None
_cuda_rt = None
_pinned_buffers = {}
_device_buffers = {}


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
        "Run: cd cuda && ./build.sh"
    )


def get_lib():
    """Get the SPHT CUDA library handle (loads if needed)."""
    global _lib
    if _lib is None:
        _lib = _load_library()
        _setup_functions(_lib)
    return _lib


def get_cuda_rt():
    """Get cached CUDA runtime library handle."""
    global _cuda_rt
    if _cuda_rt is None:
        _cuda_rt = ctypes.CDLL("libcudart.so")
    return _cuda_rt


# ============================================================================
# Buffer Management
# ============================================================================

def get_pinned_buffers(size, use_f32):
    """Get or allocate cached pinned memory buffers."""
    global _pinned_buffers
    key = (size, use_f32)
    if key not in _pinned_buffers:
        cuda_rt = get_cuda_rt()
        h_real = ctypes.c_void_p()
        h_imag = ctypes.c_void_p()
        cuda_rt.cudaHostAlloc(ctypes.byref(h_real), size, 0)
        cuda_rt.cudaHostAlloc(ctypes.byref(h_imag), size, 0)
        _pinned_buffers[key] = (h_real, h_imag)
    return _pinned_buffers[key]


def get_device_buffer(name: str, required_size: int, use_f32: bool):
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
    cuda_rt = get_cuda_rt()

    if key in _device_buffers:
        ptr, current_size = _device_buffers[key]
        if current_size >= required_size:
            return ptr
        else:
            cuda_rt.cudaFree(ptr)

    ptr = ctypes.c_void_p()
    cuda_rt.cudaMalloc(ctypes.byref(ptr), required_size)
    _device_buffers[key] = (ptr, required_size)
    return ptr


def clear_device_buffer_cache():
    """Free all cached device buffers. Call this to release GPU memory."""
    global _device_buffers
    cuda_rt = get_cuda_rt()
    for key, (ptr, size) in _device_buffers.items():
        cuda_rt.cudaFree(ptr)
    _device_buffers.clear()


def clear_pinned_buffer_cache():
    """Free all cached pinned memory buffers."""
    global _pinned_buffers
    cuda_rt = get_cuda_rt()
    for key, (h_real, h_imag) in _pinned_buffers.items():
        cuda_rt.cudaFreeHost(h_real)
        cuda_rt.cudaFreeHost(h_imag)
    _pinned_buffers.clear()


# ============================================================================
# Function Signature Setup
# ============================================================================

def _setup_functions(lib):
    """Set up function signatures for the C library."""
    # ========== alm2cl functions ==========
    lib.alm2cl_cuda_auto_f64.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2cl_cuda_auto_f64.restype = None
    lib.alm2cl_cuda_auto_f32.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2cl_cuda_auto_f32.restype = None

    lib.alm2cl_cuda_cross_f64.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.alm2cl_cuda_cross_f64.restype = None
    lib.alm2cl_cuda_cross_f32.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.alm2cl_cuda_cross_f32.restype = None

    # ========== map2alm v6 functions ==========
    lib.map2alm_cuda_v6_f64_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_f64_f64.restype = None
    lib.map2alm_cuda_v6_f64_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_f64_f32.restype = None
    lib.map2alm_cuda_v6_f32_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_f32_f64.restype = None
    lib.map2alm_cuda_v6_f32_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_f32_f32.restype = None

    # Spin-2 variants
    lib.map2alm_cuda_v6_spin2_f64_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_spin2_f64_f64.restype = None
    lib.map2alm_cuda_v6_spin2_f64_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_spin2_f64_f32.restype = None
    lib.map2alm_cuda_v6_spin2_f32_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_spin2_f32_f64.restype = None
    lib.map2alm_cuda_v6_spin2_f32_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_spin2_f32_f32.restype = None

    # ========== alm2map v6 functions ==========
    lib.alm2map_cuda_v6_f64_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_f64_f64.restype = None
    lib.alm2map_cuda_v6_f64_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_f64_f32.restype = None
    lib.alm2map_cuda_v6_f32_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_f32_f64.restype = None
    lib.alm2map_cuda_v6_f32_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_f32_f32.restype = None

    # Spin-2 variants
    lib.alm2map_cuda_v6_spin2_f64_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_spin2_f64_f64.restype = None
    lib.alm2map_cuda_v6_spin2_f64_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_spin2_f64_f32.restype = None
    lib.alm2map_cuda_v6_spin2_f32_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_spin2_f32_f64.restype = None
    lib.alm2map_cuda_v6_spin2_f32_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.alm2map_cuda_v6_spin2_f32_f32.restype = None

    # ========== map2alm LOG mode functions ==========
    lib.map2alm_cuda_v6_log_f64_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_log_f64_f64.restype = None
    lib.map2alm_cuda_v6_log_f64_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_log_f64_f32.restype = None
    lib.map2alm_cuda_v6_log_f32_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_log_f32_f64.restype = None
    lib.map2alm_cuda_v6_log_f32_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.map2alm_cuda_v6_log_f32_f32.restype = None

    # ========== Phase 1 method control ==========
    lib.set_phase1_method.argtypes = [c_int]
    lib.set_phase1_method.restype = None
    lib.get_phase1_method.argtypes = []
    lib.get_phase1_method.restype = c_int

    # ========== Ylm computation ==========
    lib.compute_ylm_cuda_f64.argtypes = [c_int, c_void_p, c_void_p, c_int]
    lib.compute_ylm_cuda_f64.restype = None

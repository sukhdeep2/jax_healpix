"""
HEALPix Operations - Additional functions for SPHT CUDA

Implements healpy-compatible functions:
- gauss_beam: Compute Gaussian beam window function
- smoothalm: Apply beam smoothing to alm
- almxfl: Multiply alm by arbitrary filter f(l)
- pixwin: Compute pixel window function
- resize_alm: Resize alm to different lmax
- synalm: Generate random alm from power spectrum
- synfast: Generate random map from power spectrum
- smoothing: Smooth a map with Gaussian beam
- anafast: Compute power spectrum from map
"""

import numpy as np
import ctypes
from ctypes import c_int, c_float, c_double, c_void_p, c_bool, c_ulonglong, POINTER
from pathlib import Path

# Import from main module - support both package and standalone import
try:
    from .spht_cuda import _get_lib, _get_cuda_rt, config, SPHTCuda, alm2cl_cuda
except ImportError:
    from spht_cuda import _get_lib, _get_cuda_rt, config, SPHTCuda, alm2cl_cuda


def _setup_healpix_ops(lib):
    """Set up function signatures for HEALPix operations."""

    # Gaussian beam
    lib.gauss_beam_cuda_f64.argtypes = [c_int, c_double, c_void_p, c_void_p]
    lib.gauss_beam_cuda_f64.restype = None
    lib.gauss_beam_cuda_f32.argtypes = [c_int, c_float, c_void_p, c_void_p]
    lib.gauss_beam_cuda_f32.restype = None

    # Smoothalm
    lib.smoothalm_cuda_f64.argtypes = [c_int, c_int, c_double, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.smoothalm_cuda_f64.restype = None
    lib.smoothalm_cuda_f32.argtypes = [c_int, c_int, c_float, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.smoothalm_cuda_f32.restype = None

    lib.smoothalm_inplace_cuda_f64.argtypes = [c_int, c_int, c_double, c_void_p, c_void_p]
    lib.smoothalm_inplace_cuda_f64.restype = None
    lib.smoothalm_inplace_cuda_f32.argtypes = [c_int, c_int, c_float, c_void_p, c_void_p]
    lib.smoothalm_inplace_cuda_f32.restype = None

    # ALMxFL
    lib.almxfl_inplace_cuda_f64.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.almxfl_inplace_cuda_f64.restype = None
    lib.almxfl_inplace_cuda_f32.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p]
    lib.almxfl_inplace_cuda_f32.restype = None

    # Pixwin
    lib.pixwin_cuda_f64.argtypes = [c_int, c_int, c_void_p, c_void_p]
    lib.pixwin_cuda_f64.restype = None
    lib.pixwin_cuda_f32.argtypes = [c_int, c_int, c_void_p, c_void_p]
    lib.pixwin_cuda_f32.restype = None

    # Resize ALM
    lib.resize_alm_cuda_f64.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.resize_alm_cuda_f64.restype = None
    lib.resize_alm_cuda_f32.argtypes = [c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.resize_alm_cuda_f32.restype = None

    # Synalm
    lib.synalm_cuda_f64.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_bool]
    lib.synalm_cuda_f64.restype = None
    lib.synalm_cuda_f32.argtypes = [c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_bool]
    lib.synalm_cuda_f32.restype = None

    lib.synalm_init_rng_cuda.argtypes = [POINTER(c_void_p), c_int, c_ulonglong]
    lib.synalm_init_rng_cuda.restype = None
    lib.synalm_free_rng_cuda.argtypes = [c_void_p]
    lib.synalm_free_rng_cuda.restype = None


# Global flag to track if functions are set up
_healpix_ops_setup = False


def _ensure_setup():
    """Ensure HEALPix ops functions are set up."""
    global _healpix_ops_setup
    if not _healpix_ops_setup:
        lib = _get_lib()
        try:
            _setup_healpix_ops(lib)
            _healpix_ops_setup = True
        except AttributeError as e:
            raise ImportError(
                f"HEALPix operations not available in this build: {e}\n"
                "Please rebuild with beam_alm_ops.cu and synalm.cu included."
            )


# ============================================================================
# Gaussian Beam
# ============================================================================

def gauss_beam(fwhm: float, lmax: int, pol: bool = False) -> np.ndarray:
    """
    Compute Gaussian beam window function.

    Args:
        fwhm: Full-width half-maximum in radians
        lmax: Maximum multipole
        pol: If True, return polarization beam (E/B modes).
             For Gaussian beam, all modes are identical.

    Returns:
        bl: Beam window function, shape [lmax+1] or [4, lmax+1] if pol=True
            For pol=True: [bl_T, bl_E, bl_B, bl_TE] (all identical for Gaussian)
    """
    _ensure_setup()
    lib = _get_lib()
    cuda_rt = _get_cuda_rt()

    use_f32 = (config.storage_precision == "float32")
    dtype = np.float32 if use_f32 else np.float64
    lp1 = lmax + 1

    # Allocate device memory
    bl_size = lp1 * np.dtype(dtype).itemsize
    d_bl = ctypes.c_void_p()
    d_log_bl = ctypes.c_void_p()
    cuda_rt.cudaMalloc(ctypes.byref(d_bl), bl_size)
    cuda_rt.cudaMalloc(ctypes.byref(d_log_bl), bl_size)

    try:
        # Run kernel
        if use_f32:
            lib.gauss_beam_cuda_f32(lmax, c_float(fwhm), d_bl, d_log_bl)
        else:
            lib.gauss_beam_cuda_f64(lmax, c_double(fwhm), d_bl, d_log_bl)

        # Synchronize to ensure kernel completion
        cuda_rt.cudaDeviceSynchronize()

        # Copy back
        bl = np.zeros(lp1, dtype=dtype)
        cuda_rt.cudaMemcpy(bl.ctypes.data_as(c_void_p), d_bl, bl_size, 2)

    finally:
        cuda_rt.cudaFree(d_bl)
        cuda_rt.cudaFree(d_log_bl)

    if pol:
        # For Gaussian beam, all polarization modes are identical
        return np.stack([bl, bl, bl, bl], axis=0)

    return bl


# ============================================================================
# Smooth ALM
# ============================================================================

def smoothalm(alms: np.ndarray, fwhm: float = 0.0, sigma: float = None,
              beam_window: np.ndarray = None, inplace: bool = False) -> np.ndarray:
    """
    Smooth alm with a Gaussian beam or custom beam window.

    Args:
        alms: Spherical harmonic coefficients, shape [n_fields, lmax+1, lmax+1]
              or complex array that will be split into real/imag
        fwhm: Full-width half-maximum of Gaussian beam in radians
        sigma: Beam sigma in radians (alternative to fwhm)
               sigma = fwhm / sqrt(8 * ln(2))
        beam_window: Custom beam window function [lmax+1]. If provided, fwhm/sigma
                     are ignored and this window is applied directly.
        inplace: If True, modify alms in place (only for split real/imag input)

    Returns:
        Smoothed alm coefficients, same shape as input
    """
    _ensure_setup()
    lib = _get_lib()
    cuda_rt = _get_cuda_rt()

    # Handle complex input
    if np.iscomplexobj(alms):
        alm_real = np.ascontiguousarray(alms.real)
        alm_imag = np.ascontiguousarray(alms.imag)
        was_complex = True
    else:
        raise ValueError("Expected complex alm array. Use split real/imag for more control.")

    use_f32 = (alm_real.dtype == np.float32)
    dtype = np.float32 if use_f32 else np.float64
    alm_real = alm_real.astype(dtype)
    alm_imag = alm_imag.astype(dtype)

    if alm_real.ndim == 2:
        alm_real = alm_real.reshape(1, alm_real.shape[0], alm_real.shape[1])
        alm_imag = alm_imag.reshape(1, alm_imag.shape[0], alm_imag.shape[1])

    n_fields = alm_real.shape[0]
    lmax = alm_real.shape[1] - 1
    lp1 = lmax + 1

    # Compute fwhm from sigma if needed
    if sigma is not None:
        fwhm = sigma * np.sqrt(8.0 * np.log(2.0))

    if beam_window is not None:
        # Use almxfl with custom window
        return almxfl(alms, beam_window)

    # Sizes
    alm_size = n_fields * lp1 * lp1 * dtype().itemsize

    # Allocate device memory
    d_alm_real_in = ctypes.c_void_p()
    d_alm_imag_in = ctypes.c_void_p()
    d_alm_real_out = ctypes.c_void_p()
    d_alm_imag_out = ctypes.c_void_p()

    cuda_rt.cudaMalloc(ctypes.byref(d_alm_real_in), alm_size)
    cuda_rt.cudaMalloc(ctypes.byref(d_alm_imag_in), alm_size)
    cuda_rt.cudaMalloc(ctypes.byref(d_alm_real_out), alm_size)
    cuda_rt.cudaMalloc(ctypes.byref(d_alm_imag_out), alm_size)

    # Copy to device
    cuda_rt.cudaMemcpy(d_alm_real_in, alm_real.ctypes.data_as(c_void_p), alm_size, 1)
    cuda_rt.cudaMemcpy(d_alm_imag_in, alm_imag.ctypes.data_as(c_void_p), alm_size, 1)

    try:
        # Run kernel
        if use_f32:
            lib.smoothalm_cuda_f32(lmax, n_fields, float(fwhm),
                                    d_alm_real_in, d_alm_imag_in,
                                    d_alm_real_out, d_alm_imag_out)
        else:
            lib.smoothalm_cuda_f64(lmax, n_fields, float(fwhm),
                                    d_alm_real_in, d_alm_imag_in,
                                    d_alm_real_out, d_alm_imag_out)

        # Copy back
        alm_real_out = np.zeros_like(alm_real)
        alm_imag_out = np.zeros_like(alm_imag)
        cuda_rt.cudaMemcpy(alm_real_out.ctypes.data_as(c_void_p), d_alm_real_out, alm_size, 2)
        cuda_rt.cudaMemcpy(alm_imag_out.ctypes.data_as(c_void_p), d_alm_imag_out, alm_size, 2)

    finally:
        cuda_rt.cudaFree(d_alm_real_in)
        cuda_rt.cudaFree(d_alm_imag_in)
        cuda_rt.cudaFree(d_alm_real_out)
        cuda_rt.cudaFree(d_alm_imag_out)

    # Combine back to complex
    complex_dtype = np.complex64 if use_f32 else np.complex128
    result = (alm_real_out + 1j * alm_imag_out).astype(complex_dtype)

    if result.shape[0] == 1:
        result = result[0]

    return result


# ============================================================================
# ALM x f(l) Filter
# ============================================================================

def almxfl(alms: np.ndarray, fl: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    Multiply alm by filter function f(l).

    Args:
        alms: Spherical harmonic coefficients (complex array)
        fl: Filter function, shape [lmax+1]
        inplace: If True, modify alms in place

    Returns:
        Filtered alm coefficients
    """
    _ensure_setup()
    lib = _get_lib()
    cuda_rt = _get_cuda_rt()

    # Handle complex input
    if not np.iscomplexobj(alms):
        raise ValueError("Expected complex alm array")

    alm_real = np.ascontiguousarray(alms.real)
    alm_imag = np.ascontiguousarray(alms.imag)

    use_f32 = (alm_real.dtype == np.float32)
    dtype = np.float32 if use_f32 else np.float64
    alm_real = alm_real.astype(dtype)
    alm_imag = alm_imag.astype(dtype)
    fl = np.ascontiguousarray(fl, dtype=dtype)

    original_shape = alm_real.shape
    if alm_real.ndim == 2:
        alm_real = alm_real.reshape(1, alm_real.shape[0], alm_real.shape[1])
        alm_imag = alm_imag.reshape(1, alm_imag.shape[0], alm_imag.shape[1])

    n_fields = alm_real.shape[0]
    lmax = alm_real.shape[1] - 1
    lp1 = lmax + 1

    if len(fl) < lp1:
        # Zero-pad fl if needed
        fl_padded = np.zeros(lp1, dtype=dtype)
        fl_padded[:len(fl)] = fl
        fl = fl_padded
    elif len(fl) > lp1:
        fl = fl[:lp1]

    # Sizes
    alm_size = n_fields * lp1 * lp1 * dtype().itemsize
    fl_size = lp1 * dtype().itemsize

    # Allocate device memory
    d_alm_real = ctypes.c_void_p()
    d_alm_imag = ctypes.c_void_p()
    d_fl = ctypes.c_void_p()

    cuda_rt.cudaMalloc(ctypes.byref(d_alm_real), alm_size)
    cuda_rt.cudaMalloc(ctypes.byref(d_alm_imag), alm_size)
    cuda_rt.cudaMalloc(ctypes.byref(d_fl), fl_size)

    # Copy to device
    cuda_rt.cudaMemcpy(d_alm_real, alm_real.ctypes.data_as(c_void_p), alm_size, 1)
    cuda_rt.cudaMemcpy(d_alm_imag, alm_imag.ctypes.data_as(c_void_p), alm_size, 1)
    cuda_rt.cudaMemcpy(d_fl, fl.ctypes.data_as(c_void_p), fl_size, 1)

    try:
        # Run in-place kernel
        if use_f32:
            lib.almxfl_inplace_cuda_f32(lmax, n_fields, d_fl, d_alm_real, d_alm_imag)
        else:
            lib.almxfl_inplace_cuda_f64(lmax, n_fields, d_fl, d_alm_real, d_alm_imag)

        # Copy back
        alm_real_out = np.zeros_like(alm_real)
        alm_imag_out = np.zeros_like(alm_imag)
        cuda_rt.cudaMemcpy(alm_real_out.ctypes.data_as(c_void_p), d_alm_real, alm_size, 2)
        cuda_rt.cudaMemcpy(alm_imag_out.ctypes.data_as(c_void_p), d_alm_imag, alm_size, 2)

    finally:
        cuda_rt.cudaFree(d_alm_real)
        cuda_rt.cudaFree(d_alm_imag)
        cuda_rt.cudaFree(d_fl)

    # Combine back to complex
    complex_dtype = np.complex64 if use_f32 else np.complex128
    result = (alm_real_out + 1j * alm_imag_out).astype(complex_dtype)

    if len(original_shape) == 2:
        result = result[0]

    return result


# ============================================================================
# Pixel Window
# ============================================================================

def pixwin(nside: int, lmax: int = None, pol: bool = False) -> np.ndarray:
    """
    Compute approximate pixel window function.

    Note: This is an approximation. For exact pixel windows, use healpy's
    precomputed tables.

    Args:
        nside: HEALPix nside parameter
        lmax: Maximum multipole (default: 3*nside)
        pol: If True, return polarization window (identical for approx)

    Returns:
        pixwin: Pixel window function, shape [lmax+1] or [2, lmax+1] if pol=True
    """
    _ensure_setup()
    lib = _get_lib()
    cuda_rt = _get_cuda_rt()

    if lmax is None:
        lmax = 3 * nside

    use_f32 = (config.storage_precision == "float32")
    dtype = np.float32 if use_f32 else np.float64
    lp1 = lmax + 1

    # Allocate device memory
    pw_size = lp1 * np.dtype(dtype).itemsize
    d_pixwin = ctypes.c_void_p()
    d_log_pixwin = ctypes.c_void_p()
    cuda_rt.cudaMalloc(ctypes.byref(d_pixwin), pw_size)
    cuda_rt.cudaMalloc(ctypes.byref(d_log_pixwin), pw_size)

    try:
        # Run kernel
        if use_f32:
            lib.pixwin_cuda_f32(lmax, nside, d_pixwin, d_log_pixwin)
        else:
            lib.pixwin_cuda_f64(lmax, nside, d_pixwin, d_log_pixwin)

        # Copy back
        pw = np.zeros(lp1, dtype=dtype)
        cuda_rt.cudaMemcpy(pw.ctypes.data_as(c_void_p), d_pixwin, pw_size, 2)

    finally:
        cuda_rt.cudaFree(d_pixwin)
        cuda_rt.cudaFree(d_log_pixwin)

    if pol:
        # For approximate pixel window, T and pol are similar
        return np.stack([pw, pw], axis=0)

    return pw


# ============================================================================
# Resize ALM
# ============================================================================

def resize_alm(alms: np.ndarray, lmax_new: int, mmax_new: int = None) -> np.ndarray:
    """
    Resize alm to different lmax (truncate or zero-pad).

    Args:
        alms: Input alm coefficients (complex array)
        lmax_new: New maximum multipole
        mmax_new: New maximum m (default: lmax_new)

    Returns:
        Resized alm coefficients
    """
    _ensure_setup()
    lib = _get_lib()
    cuda_rt = _get_cuda_rt()

    if mmax_new is None:
        mmax_new = lmax_new

    if not np.iscomplexobj(alms):
        raise ValueError("Expected complex alm array")

    alm_real = np.ascontiguousarray(alms.real)
    alm_imag = np.ascontiguousarray(alms.imag)

    use_f32 = (alm_real.dtype == np.float32)
    dtype = np.float32 if use_f32 else np.float64
    alm_real = alm_real.astype(dtype)
    alm_imag = alm_imag.astype(dtype)

    original_shape = alm_real.shape
    if alm_real.ndim == 2:
        alm_real = alm_real.reshape(1, alm_real.shape[0], alm_real.shape[1])
        alm_imag = alm_imag.reshape(1, alm_imag.shape[0], alm_imag.shape[1])

    n_fields = alm_real.shape[0]
    lmax_in = alm_real.shape[1] - 1
    lp1_in = lmax_in + 1
    lp1_out = lmax_new + 1

    # Sizes
    alm_size_in = n_fields * lp1_in * lp1_in * dtype().itemsize
    alm_size_out = n_fields * lp1_out * lp1_out * dtype().itemsize

    # Allocate device memory
    d_alm_real_in = ctypes.c_void_p()
    d_alm_imag_in = ctypes.c_void_p()
    d_alm_real_out = ctypes.c_void_p()
    d_alm_imag_out = ctypes.c_void_p()

    cuda_rt.cudaMalloc(ctypes.byref(d_alm_real_in), alm_size_in)
    cuda_rt.cudaMalloc(ctypes.byref(d_alm_imag_in), alm_size_in)
    cuda_rt.cudaMalloc(ctypes.byref(d_alm_real_out), alm_size_out)
    cuda_rt.cudaMalloc(ctypes.byref(d_alm_imag_out), alm_size_out)

    # Copy to device
    cuda_rt.cudaMemcpy(d_alm_real_in, alm_real.ctypes.data_as(c_void_p), alm_size_in, 1)
    cuda_rt.cudaMemcpy(d_alm_imag_in, alm_imag.ctypes.data_as(c_void_p), alm_size_in, 1)
    cuda_rt.cudaMemset(d_alm_real_out, 0, alm_size_out)
    cuda_rt.cudaMemset(d_alm_imag_out, 0, alm_size_out)

    try:
        # Run kernel
        if use_f32:
            lib.resize_alm_cuda_f32(lmax_in, lmax_new, n_fields,
                                     d_alm_real_in, d_alm_imag_in,
                                     d_alm_real_out, d_alm_imag_out)
        else:
            lib.resize_alm_cuda_f64(lmax_in, lmax_new, n_fields,
                                     d_alm_real_in, d_alm_imag_in,
                                     d_alm_real_out, d_alm_imag_out)

        # Copy back
        alm_real_out = np.zeros((n_fields, lp1_out, lp1_out), dtype=dtype)
        alm_imag_out = np.zeros((n_fields, lp1_out, lp1_out), dtype=dtype)
        cuda_rt.cudaMemcpy(alm_real_out.ctypes.data_as(c_void_p), d_alm_real_out, alm_size_out, 2)
        cuda_rt.cudaMemcpy(alm_imag_out.ctypes.data_as(c_void_p), d_alm_imag_out, alm_size_out, 2)

    finally:
        cuda_rt.cudaFree(d_alm_real_in)
        cuda_rt.cudaFree(d_alm_imag_in)
        cuda_rt.cudaFree(d_alm_real_out)
        cuda_rt.cudaFree(d_alm_imag_out)

    # Combine back to complex
    complex_dtype = np.complex64 if use_f32 else np.complex128
    result = (alm_real_out + 1j * alm_imag_out).astype(complex_dtype)

    if len(original_shape) == 2:
        result = result[0]

    return result


# ============================================================================
# Synalm: Generate Random ALM from Power Spectrum
# ============================================================================

# Global RNG state
_synalm_rng_state = None
_synalm_rng_size = 0


def _get_synalm_rng(size: int, seed: int = None):
    """Get or initialize RNG state for synalm."""
    global _synalm_rng_state, _synalm_rng_size

    lib = _get_lib()

    if _synalm_rng_state is None or size > _synalm_rng_size:
        # Free old state if exists
        if _synalm_rng_state is not None:
            lib.synalm_free_rng_cuda(_synalm_rng_state)

        # Initialize new state
        _synalm_rng_state = ctypes.c_void_p()
        if seed is None:
            seed = np.random.randint(0, 2**62)
        lib.synalm_init_rng_cuda(ctypes.byref(_synalm_rng_state), size, seed)
        _synalm_rng_size = size

    return _synalm_rng_state


def synalm(cls, lmax: int = None, mmax: int = None, new: bool = False,
           seed: int = None) -> np.ndarray:
    """
    Generate random alm from power spectrum.

    Args:
        cls: Power spectrum(s). Can be:
             - Single array [lmax+1] for one field
             - List of arrays for multiple independent fields
        lmax: Maximum l (default: len(cls)-1)
        mmax: Maximum m (default: lmax)
        new: Ignored (for healpy compatibility)
        seed: Random seed (default: random)

    Returns:
        alm: Generated spherical harmonic coefficients
             Shape [lmax+1, lmax+1] for single field
             Shape [n_fields, lmax+1, lmax+1] for multiple fields
    """
    _ensure_setup()
    lib = _get_lib()
    cuda_rt = _get_cuda_rt()

    # Handle input format
    if isinstance(cls, (list, tuple)):
        cls = [np.asarray(c) for c in cls]
        n_fields = len(cls)
    else:
        cls = [np.asarray(cls)]
        n_fields = 1

    # Determine lmax
    if lmax is None:
        lmax = len(cls[0]) - 1
    if mmax is None:
        mmax = lmax

    lp1 = lmax + 1

    use_f32 = (config.storage_precision == "float32")
    dtype = np.float32 if use_f32 else np.float64

    # Prepare Cl arrays (pad or truncate as needed)
    cl_array = np.zeros((n_fields, lp1), dtype=dtype)
    log_cl_array = np.full((n_fields, lp1), -700.0, dtype=dtype)  # LOG_MIN

    for i, cl in enumerate(cls):
        cl = np.asarray(cl, dtype=dtype)
        n = min(len(cl), lp1)
        cl_array[i, :n] = cl[:n]
        # Compute log(Cl) for positive values
        mask = cl_array[i] > 0
        log_cl_array[i, mask] = np.log(cl_array[i, mask])

    # Get RNG state
    n_elements = n_fields * lp1 * lp1
    rng_state = _get_synalm_rng(n_elements, seed)

    # Sizes
    cl_size = n_fields * lp1 * dtype().itemsize
    alm_size = n_fields * lp1 * lp1 * dtype().itemsize

    # Allocate device memory
    d_cl = ctypes.c_void_p()
    d_log_cl = ctypes.c_void_p()
    d_alm_real = ctypes.c_void_p()
    d_alm_imag = ctypes.c_void_p()

    cuda_rt.cudaMalloc(ctypes.byref(d_cl), cl_size)
    cuda_rt.cudaMalloc(ctypes.byref(d_log_cl), cl_size)
    cuda_rt.cudaMalloc(ctypes.byref(d_alm_real), alm_size)
    cuda_rt.cudaMalloc(ctypes.byref(d_alm_imag), alm_size)

    # Copy to device
    cuda_rt.cudaMemcpy(d_cl, cl_array.ctypes.data_as(c_void_p), cl_size, 1)
    cuda_rt.cudaMemcpy(d_log_cl, log_cl_array.ctypes.data_as(c_void_p), cl_size, 1)
    cuda_rt.cudaMemset(d_alm_real, 0, alm_size)
    cuda_rt.cudaMemset(d_alm_imag, 0, alm_size)

    try:
        # Run kernel (use log-space for safety with small Cl values)
        if use_f32:
            lib.synalm_cuda_f32(lmax, n_fields, d_cl, d_log_cl, rng_state,
                                 d_alm_real, d_alm_imag, True)
        else:
            lib.synalm_cuda_f64(lmax, n_fields, d_cl, d_log_cl, rng_state,
                                 d_alm_real, d_alm_imag, True)

        # Copy back
        alm_real = np.zeros((n_fields, lp1, lp1), dtype=dtype)
        alm_imag = np.zeros((n_fields, lp1, lp1), dtype=dtype)
        cuda_rt.cudaMemcpy(alm_real.ctypes.data_as(c_void_p), d_alm_real, alm_size, 2)
        cuda_rt.cudaMemcpy(alm_imag.ctypes.data_as(c_void_p), d_alm_imag, alm_size, 2)

    finally:
        cuda_rt.cudaFree(d_cl)
        cuda_rt.cudaFree(d_log_cl)
        cuda_rt.cudaFree(d_alm_real)
        cuda_rt.cudaFree(d_alm_imag)

    # Combine to complex
    complex_dtype = np.complex64 if use_f32 else np.complex128
    result = (alm_real + 1j * alm_imag).astype(complex_dtype)

    if n_fields == 1:
        result = result[0]

    return result


# ============================================================================
# Synfast: Generate Random Map from Power Spectrum
# ============================================================================

def synfast(cls, nside: int, lmax: int = None, mmax: int = None,
            alm: bool = False, pol: bool = True, pixwin: bool = False,
            fwhm: float = 0.0, sigma: float = None, new: bool = False,
            seed: int = None) -> np.ndarray:
    """
    Generate random map from power spectrum.

    Pipeline: synalm -> [smoothalm] -> [pixwin] -> alm2map

    Args:
        cls: Power spectrum(s)
        nside: HEALPix nside
        lmax: Maximum l (default: 3*nside)
        mmax: Maximum m (default: lmax)
        alm: If True, also return alm
        pol: If True, cls should contain polarization spectra
        pixwin: If True, apply pixel window
        fwhm: Gaussian beam FWHM in radians
        sigma: Beam sigma (alternative to fwhm)
        new: Ignored (for healpy compatibility)
        seed: Random seed

    Returns:
        map: Generated HEALPix map(s), shape [n_rings, 4*nside]
        (alm): If alm=True, also return alm coefficients
    """
    if lmax is None:
        lmax = 3 * nside

    # Generate alm
    alm_out = synalm(cls, lmax=lmax, mmax=mmax, seed=seed)

    # Apply beam smoothing if requested
    if fwhm > 0 or sigma is not None:
        alm_out = smoothalm(alm_out, fwhm=fwhm, sigma=sigma)

    # Apply pixel window if requested
    if pixwin:
        pw = pixwin(nside, lmax=lmax)
        alm_out = almxfl(alm_out, pw)

    # Synthesize map
    spht = SPHTCuda(nside, lmax)
    if alm_out.ndim == 2:
        alm_out = alm_out.reshape(1, alm_out.shape[0], alm_out.shape[1])
    maps = spht.alm2map({0: alm_out}, spins=(0,))
    map_out = maps[0]

    if map_out.shape[0] == 1:
        map_out = map_out[0]

    if alm:
        return map_out, alm_out
    return map_out


# ============================================================================
# Smoothing: Smooth Map with Gaussian Beam
# ============================================================================

def smoothing(map_in: np.ndarray, fwhm: float = 0.0, sigma: float = None,
              lmax: int = None, nside: int = None) -> np.ndarray:
    """
    Smooth a HEALPix map with a Gaussian beam.

    Pipeline: map2alm -> smoothalm -> alm2map

    Args:
        map_in: Input map, shape [n_rings, 4*nside] or [n_maps, n_rings, 4*nside]
        fwhm: Beam FWHM in radians
        sigma: Beam sigma (alternative to fwhm)
        lmax: Maximum l for transform (default: 3*nside)
        nside: HEALPix nside (inferred from map if not provided)

    Returns:
        Smoothed map with same shape as input
    """
    map_in = np.asarray(map_in)

    # Infer nside from map shape
    if nside is None:
        if map_in.ndim == 2:
            nside = map_in.shape[1] // 4
        elif map_in.ndim == 3:
            nside = map_in.shape[2] // 4
        else:
            raise ValueError(f"Cannot infer nside from map shape {map_in.shape}")

    if lmax is None:
        lmax = 3 * nside

    # Ensure proper shape
    original_shape = map_in.shape
    if map_in.ndim == 2:
        map_in = map_in.reshape(1, map_in.shape[0], map_in.shape[1])

    # Transform to alm
    spht = SPHTCuda(nside, lmax)
    alm_result = spht.map2alm({0: map_in}, spins=(0,))
    alm_data = alm_result[0]

    # Smooth
    alm_smooth = smoothalm(alm_data, fwhm=fwhm, sigma=sigma)

    # Transform back to map
    if alm_smooth.ndim == 2:
        alm_smooth = alm_smooth.reshape(1, alm_smooth.shape[0], alm_smooth.shape[1])
    maps_out = spht.alm2map({0: alm_smooth}, spins=(0,))
    map_out = maps_out[0]

    # Restore original shape
    if len(original_shape) == 2:
        map_out = map_out[0]

    return map_out


# ============================================================================
# Anafast: Compute Power Spectrum from Map
# ============================================================================

def anafast(map1: np.ndarray, map2: np.ndarray = None, nside: int = None,
            lmax: int = None, mmax: int = None, iter: int = 0,
            alm: bool = False, pol: bool = True, use_weights: bool = False,
            regression: bool = True) -> np.ndarray:
    """
    Compute power spectrum from map(s).

    Pipeline: map2alm [x iter] -> alm2cl

    Args:
        map1: First input map
        map2: Second map for cross-spectrum (optional)
        nside: HEALPix nside (inferred from map if not provided)
        lmax: Maximum l (default: 3*nside)
        mmax: Maximum m (default: lmax)
        iter: Number of iterative refinement steps (0 for single pass)
        alm: If True, also return alm
        pol: If True, map contains polarization
        use_weights: Ignored (no pixel weights in CUDA implementation)
        regression: Ignored (for healpy compatibility)

    Returns:
        cl: Power spectrum, shape [lmax+1]
        (alm): If alm=True, also return alm coefficients
    """
    map1 = np.asarray(map1)

    # Infer nside
    if nside is None:
        if map1.ndim == 2:
            nside = map1.shape[1] // 4
        elif map1.ndim == 3:
            nside = map1.shape[2] // 4
        elif map1.ndim == 1:
            npix = map1.shape[0]
            nside = int(np.sqrt(npix / 12))
        else:
            raise ValueError(f"Cannot infer nside from map shape {map1.shape}")

    if lmax is None:
        lmax = 3 * nside

    # Ensure proper shape
    if map1.ndim == 2:
        map1 = map1.reshape(1, map1.shape[0], map1.shape[1])
    elif map1.ndim == 1:
        # 1D format - need to convert
        raise ValueError("1D HEALPix format not supported. Please use 2D ring format.")

    spht = SPHTCuda(nside, lmax)

    # Transform to alm (with optional iteration)
    alm_result = spht.map2alm({0: map1}, spins=(0,), return_split=True)
    alm_real, alm_imag = alm_result[0]

    if iter > 0:
        # Iterative refinement
        for _ in range(iter):
            # Synthesize map
            alm_complex = (alm_real + 1j * alm_imag)
            if alm_complex.ndim == 2:
                alm_complex = alm_complex.reshape(1, alm_complex.shape[0], alm_complex.shape[1])
            maps_synth = spht.alm2map({0: alm_complex}, spins=(0,))

            # Compute residual
            residual = map1 - maps_synth[0]

            # Add correction
            correction = spht.map2alm({0: residual}, spins=(0,), return_split=True)
            corr_real, corr_imag = correction[0]
            alm_real = alm_real + corr_real
            alm_imag = alm_imag + corr_imag

    # Compute power spectrum
    cl = alm2cl_cuda(lmax, alm_real, alm_imag)

    if cl.shape[0] == 1:
        cl = cl[0]

    if alm:
        use_f32 = (alm_real.dtype == np.float32)
        complex_dtype = np.complex64 if use_f32 else np.complex128
        alm_out = (alm_real + 1j * alm_imag).astype(complex_dtype)
        if alm_out.shape[0] == 1:
            alm_out = alm_out[0]
        return cl, alm_out

    return cl

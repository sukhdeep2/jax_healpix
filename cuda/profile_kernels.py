#!/usr/bin/env python3
"""
CUDA SPHT Kernel Profiler

Profiles map2alm and alm2map functions with detailed timing breakdown:
- Python overhead
- Phase 1: Gm computation (ring DFT)
- Phase 2: alm reduction (Ylm recurrence)
- Memory transfers
- Kernel statistics (registers, shared memory, occupancy)

Usage:
    python profile_kernels.py [--nside N] [--dtype f32|f64] [--iterations N]
"""

import argparse
import subprocess
import sys
import os
import time
import numpy as np

# Add paths for imports
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')


def profile_python_timing(nside, dtype, iterations=5):
    """Profile with Python timing and CUDA events"""
    print("\n" + "="*70)
    print(f"TIMING PROFILE: nside={nside}, dtype={dtype}, iterations={iterations}")
    print("="*70)

    from spht_cuda import SPHTCuda
    import torch

    l_max = nside
    n_maps = 1
    np_dtype = np.float32 if dtype == "f32" else np.float64
    precision = "float32" if dtype == "f32" else "float64"

    # Initialize
    spht = SPHTCuda(nside, l_max, storage_precision=precision, recurrence_precision=precision)

    # Create test data - complex alm
    np.random.seed(42)
    alm_re = np.random.randn(n_maps, l_max+1, l_max+1).astype(np_dtype)
    alm_im = np.random.randn(n_maps, l_max+1, l_max+1).astype(np_dtype)
    for l in range(l_max+1):
        alm_re[:, l, l+1:] = 0
        alm_im[:, l, l+1:] = 0
        alm_im[:, l, 0] = 0

    # Create complex alm array
    complex_dtype = np.complex64 if dtype == "f32" else np.complex128
    alm_complex = (alm_re + 1j * alm_im).astype(complex_dtype)
    alm_in = {0: alm_complex}

    # Warmup
    print("\nWarming up...")
    for _ in range(2):
        map_out = spht.alm2map(alm_in, spins=(0,))
        alm_out = spht.map2alm(map_out, spins=(0,))
    torch.cuda.synchronize()

    # Profile alm2map
    print("\n--- alm2map ---")
    alm2map_times = []
    for i in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        map_out = spht.alm2map(alm_in, spins=(0,))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        alm2map_times.append((t1 - t0) * 1000)
        print(f"  Iter {i+1}: {alm2map_times[-1]:.2f} ms")

    print(f"  Mean: {np.mean(alm2map_times):.2f} +/- {np.std(alm2map_times):.2f} ms")

    # Profile map2alm
    print("\n--- map2alm ---")
    map2alm_times = []
    for i in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        alm_out = spht.map2alm(map_out, spins=(0,))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        map2alm_times.append((t1 - t0) * 1000)
        print(f"  Iter {i+1}: {map2alm_times[-1]:.2f} ms")

    print(f"  Mean: {np.mean(map2alm_times):.2f} +/- {np.std(map2alm_times):.2f} ms")

    # Summary
    print("\n--- Summary ---")
    print(f"  alm2map: {np.mean(alm2map_times):.2f} ms")
    print(f"  map2alm: {np.mean(map2alm_times):.2f} ms")
    print(f"  Roundtrip: {np.mean(alm2map_times) + np.mean(map2alm_times):.2f} ms")

    return map_out, spht


def profile_cuda_events(nside, dtype, iterations=5, map_out=None, spht=None):
    """Profile using CUDA events for precise GPU timing"""
    print("\n" + "="*70)
    print(f"CUDA EVENT TIMING")
    print("="*70)

    from spht_cuda import SPHTCuda
    import torch

    l_max = nside
    n_maps = 1
    np_dtype = np.float32 if dtype == "f32" else np.float64
    precision = "float32" if dtype == "f32" else "float64"

    if spht is None:
        spht = SPHTCuda(nside, l_max, storage_precision=precision, recurrence_precision=precision)

    np.random.seed(42)
    alm_re = np.random.randn(n_maps, l_max+1, l_max+1).astype(np_dtype)
    alm_im = np.random.randn(n_maps, l_max+1, l_max+1).astype(np_dtype)
    for l in range(l_max+1):
        alm_re[:, l, l+1:] = 0
        alm_im[:, l, l+1:] = 0
        alm_im[:, l, 0] = 0

    complex_dtype = np.complex64 if dtype == "f32" else np.complex128
    alm_complex = (alm_re + 1j * alm_im).astype(complex_dtype)
    alm_in = {0: alm_complex}

    # Warmup
    if map_out is None:
        for _ in range(2):
            map_out = spht.alm2map(alm_in, spins=(0,))
            _ = spht.map2alm(map_out, spins=(0,))
    torch.cuda.synchronize()

    # Create CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Profile alm2map
    alm2map_gpu_times = []
    for _ in range(iterations):
        start.record()
        map_out = spht.alm2map(alm_in, spins=(0,))
        end.record()
        torch.cuda.synchronize()
        alm2map_gpu_times.append(start.elapsed_time(end))

    # Profile map2alm
    map2alm_gpu_times = []
    for _ in range(iterations):
        start.record()
        _ = spht.map2alm(map_out, spins=(0,))
        end.record()
        torch.cuda.synchronize()
        map2alm_gpu_times.append(start.elapsed_time(end))

    print(f"\nalm2map GPU time: {np.mean(alm2map_gpu_times):.2f} +/- {np.std(alm2map_gpu_times):.2f} ms")
    print(f"map2alm GPU time: {np.mean(map2alm_gpu_times):.2f} +/- {np.std(map2alm_gpu_times):.2f} ms")


def profile_with_env_timing(nside, dtype, iterations=3):
    """Profile using SPHT_TIMING environment variable for internal timing"""
    print("\n" + "="*70)
    print(f"INTERNAL PHASE TIMING (SPHT_TIMING=1)")
    print("="*70)

    env = os.environ.copy()
    env["SPHT_TIMING"] = "1"

    np_dtype_str = "np.float32" if dtype == "f32" else "np.float64"

    test_script = f'''
import sys
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
sys.path.insert(0, '/home/deep/repos/SPHT/jax_healpix')
import numpy as np
from spht_cuda import SPHTCuda
import torch

nside = {nside}
l_max = nside
n_maps = 1
np_dtype = {np_dtype_str}

spht = SPHTCuda(nside, l_max, storage_dtype=np_dtype, recurrence_dtype=np_dtype)

np.random.seed(42)
alm_re = np.random.randn(n_maps, l_max+1, l_max+1).astype(np_dtype)
alm_im = np.random.randn(n_maps, l_max+1, l_max+1).astype(np_dtype)
for l in range(l_max+1):
    alm_re[:, l, l+1:] = 0
    alm_im[:, l, l+1:] = 0
    alm_im[:, l, 0] = 0

# Warmup
for _ in range(2):
    map_out = spht.alm2map(alm_re, alm_im)
    alm_out_re, alm_out_im = spht.map2alm(map_out)
torch.cuda.synchronize()

print("\\n=== Profiled runs ===")
for i in range({iterations}):
    print(f"\\n--- Iteration {{i+1}} ---")
    torch.cuda.synchronize()
    map_out = spht.alm2map(alm_re, alm_im)
    torch.cuda.synchronize()
    print("  [alm2map complete]")
    alm_out_re, alm_out_im = spht.map2alm(map_out)
    torch.cuda.synchronize()
    print("  [map2alm complete]")
'''

    result = subprocess.run(
        ["python3", "-c", test_script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120
    )
    print(result.stdout)
    if result.stderr:
        # Filter out common warnings
        for line in result.stderr.split('\n'):
            if line and 'warning' not in line.lower() and 'XLA' not in line:
                print(line)


def check_compilation_stats():
    """Check kernel compilation statistics using cuobjdump"""
    print("\n" + "="*70)
    print("COMPILATION STATISTICS (registers, shared memory per kernel)")
    print("="*70)

    so_file = "/home/deep/repos/SPHT/cuda/build/libspht_cuda.so"

    if not os.path.exists(so_file):
        print(f"Library not found: {so_file}")
        return

    print(f"\nAnalyzing: {so_file}")

    # Use cuobjdump to get kernel info
    try:
        result = subprocess.run(
            ["cuobjdump", "-res-usage", so_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            # Parse output for key metrics
            lines = result.stdout.split('\n')
            current_kernel = None
            for line in lines:
                if 'Function' in line:
                    # Extract kernel name
                    current_kernel = line.strip()
                    print(f"\n{current_kernel}")
                elif current_kernel:
                    if 'REG:' in line or 'STACK:' in line or 'registers' in line.lower():
                        print(f"  {line.strip()}")
                    elif 'SM' in line or 'smem' in line.lower() or 'shared' in line.lower():
                        print(f"  {line.strip()}")
        else:
            print(f"cuobjdump error: {result.stderr}")
    except FileNotFoundError:
        print("cuobjdump not found - install CUDA toolkit")
    except subprocess.TimeoutExpired:
        print("cuobjdump timed out")


def run_nsight_trace(nside, dtype):
    """Run Nsight Systems trace for timeline view"""
    print("\n" + "="*70)
    print("NSIGHT SYSTEMS TRACE")
    print("="*70)

    np_dtype_str = "np.float32" if dtype == "f32" else "np.float64"

    test_script = f'''
import sys
sys.path.insert(0, '/home/deep/repos/SPHT/cuda/python')
import numpy as np
from spht_cuda import SPHTCuda
import torch

nside = {nside}
l_max = nside
np_dtype = {np_dtype_str}

spht = SPHTCuda(nside, l_max, storage_dtype=np_dtype, recurrence_dtype=np_dtype)

np.random.seed(42)
alm_re = np.random.randn(1, l_max+1, l_max+1).astype(np_dtype)
alm_im = np.random.randn(1, l_max+1, l_max+1).astype(np_dtype)
for l in range(l_max+1):
    alm_re[:, l, l+1:] = 0
    alm_im[:, l, l+1:] = 0
    alm_im[:, l, 0] = 0

# Warmup
for _ in range(2):
    map_out = spht.alm2map(alm_re, alm_im)
    _ = spht.map2alm(map_out)
torch.cuda.synchronize()

# Profile run
torch.cuda.nvtx.range_push("alm2map")
map_out = spht.alm2map(alm_re, alm_im)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("map2alm")
_ = spht.map2alm(map_out)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
'''

    test_file = "/tmp/spht_nsight_test.py"
    with open(test_file, "w") as f:
        f.write(test_script)

    try:
        result = subprocess.run(["which", "nsys"], capture_output=True)
        if result.returncode == 0:
            output_file = f"/tmp/spht_profile_nside{nside}_{dtype}"
            print(f"\nRunning nsys profile -> {output_file}.nsys-rep")
            cmd = [
                "nsys", "profile",
                "--output", output_file,
                "--force-overwrite", "true",
                "--trace", "cuda,nvtx",
                "python3", test_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"Profile saved. View with: nsys-ui {output_file}.nsys-rep")
            else:
                print(f"nsys failed: {result.stderr}")
        else:
            print("nsys not found - install Nsight Systems")
    except subprocess.TimeoutExpired:
        print("nsys timed out")


def main():
    parser = argparse.ArgumentParser(description="Profile CUDA SPHT kernels")
    parser.add_argument("--nside", type=int, default=256, help="HEALPix nside")
    parser.add_argument("--dtype", choices=["f32", "f64"], default="f32", help="Data type")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--skip-compilation", action="store_true", help="Skip compilation stats")
    parser.add_argument("--nsight", action="store_true", help="Run Nsight Systems trace")
    args = parser.parse_args()

    print("="*70)
    print(f"SPHT CUDA PROFILER")
    print(f"nside={args.nside}, dtype={args.dtype}, iterations={args.iterations}")
    print("="*70)

    # GPU info
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                                "--format=csv,noheader"], capture_output=True, text=True)
        print(f"\nGPU: {result.stdout.strip()}")
    except:
        pass

    # Run profiling
    map_out, spht = profile_python_timing(args.nside, args.dtype, args.iterations)
    profile_cuda_events(args.nside, args.dtype, args.iterations, map_out, spht)
    profile_with_env_timing(args.nside, args.dtype, min(args.iterations, 3))

    if not args.skip_compilation:
        check_compilation_stats()

    if args.nsight:
        run_nsight_trace(args.nside, args.dtype)


if __name__ == "__main__":
    main()

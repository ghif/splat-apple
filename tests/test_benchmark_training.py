"""
Performance Benchmark for Gaussian Splatting Training.
Compares PyTorch (MPS) and MLX (Metal) across Python and C++ implementations.
"""
import torch
import torch.optim as optim
import mlx.core as mx
import mlx.optimizers as optim_mlx
import time
import os
import pytest
import numpy as np

# Torch GS
from torch_gs.core.gaussians import init_gaussians_from_pcd as init_torch
from torch_gs.io.colmap import load_colmap_dataset as load_torch
from torch_gs.training.trainer import train_step as step_torch

# MLX GS
from mlx_gs.core.gaussians import init_gaussians_from_pcd as init_mlx
from mlx_gs.io.colmap import load_colmap_dataset as load_mlx
from mlx_gs.training.trainer import train_step as step_mlx

# Dataset Path
DATA_DIR = "/Users/mghifary/Work/Code/AI/data/gsplat"

def run_benchmark_torch(gaussians, optimizer, cameras, targets, device, rasterizer_type):
    """Runs a benchmark for PyTorch."""
    num_runs = 10
    # Warm-up
    step_torch(gaussians, optimizer, targets[0], cameras[0], device=device, rasterizer_type=rasterizer_type)
    torch.mps.synchronize()
    
    times = []
    for i in range(num_runs):
        cam_idx = i % len(cameras)
        curr_cam, curr_target = cameras[cam_idx], targets[cam_idx]
        start = time.perf_counter()
        step_torch(gaussians, optimizer, curr_target, curr_cam, device=device, rasterizer_type=rasterizer_type)
        torch.mps.synchronize()
        times.append(time.perf_counter() - start)
    
    # Skip first 2 for steady-state (JIT/Metal overhead)
    avg = sum(times[2:]) / (num_runs - 2) if num_runs > 2 else sum(times) / num_runs
    return avg

def run_benchmark_mlx(params, optimizer, cameras, targets, rasterizer_type):
    """Runs a benchmark for MLX."""
    num_runs = 10
    # Warm-up
    step_mlx(params, optimizer, targets[0], cameras[0], rasterizer_type=rasterizer_type)
    mx.eval(params["means"])
    
    times = []
    for i in range(num_runs):
        idx = i % len(cameras)
        start = time.perf_counter()
        # Returns: loss, rendered_image, psnr, grad_norms
        loss, _, _, _ = step_mlx(params, optimizer, targets[idx], cameras[idx], rasterizer_type=rasterizer_type) 
        mx.eval(loss, params["means"]) # Force sync
        times.append(time.perf_counter() - start)
        
    # Skip first 2 for steady-state
    avg = sum(times[2:]) / (num_runs - 2) if num_runs > 2 else sum(times) / num_runs
    return avg

def test_benchmark_training_fern():
    """
    Main benchmark entry point for the 'Fern' scene.
    Compares all training modes.
    """
    device = "mps"
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available.")
        
    path = os.path.join(DATA_DIR, "nerf_example_data/nerf_llff_data/fern")
    if not os.path.exists(path):
        pytest.skip(f"Fern dataset not found at {path}.")
        
    # 1. PyTorch Setup
    xyz_t, rgb_t, cams_t, tars_t = load_torch(path, "images_8", device=device)
    gaussians_t = init_torch(torch.tensor(xyz_t, dtype=torch.float32), torch.tensor(rgb_t, dtype=torch.float32), device=device)
    gaussians_t.means.requires_grad = True
    gaussians_t.scales.requires_grad = True
    gaussians_t.quaternions.requires_grad = True
    gaussians_t.opacities.requires_grad = True
    gaussians_t.sh_coeffs.requires_grad = True
    opt_t = optim.Adam([gaussians_t.means, gaussians_t.scales, gaussians_t.quaternions, gaussians_t.opacities, gaussians_t.sh_coeffs], lr=0.001)

    # 2. MLX Setup
    xyz_m, rgb_m, cams_m, tars_m = load_mlx(path, "images_8")
    gaussians_m = init_mlx(xyz_m, rgb_m)
    params_m = {
        "means": gaussians_m.means, "scales": gaussians_m.scales,
        "quaternions": gaussians_m.quaternions, "opacities": gaussians_m.opacities,
        "sh_coeffs": gaussians_m.sh_coeffs
    }
    opt_m = optim_mlx.Adam(learning_rate=0.001)

    print(f"\nBenchmarking {xyz_t.shape[0]} Gaussians...")
    
    # 3. Execution
    py_torch = run_benchmark_torch(gaussians_t, opt_t, cams_t, tars_t, device, "python")
    cpp_torch = run_benchmark_torch(gaussians_t, opt_t, cams_t, tars_t, device, "cpp")
    py_mlx = run_benchmark_mlx(params_m, opt_m, cams_m, tars_m, "python")
    cpp_mlx = run_benchmark_mlx(params_m, opt_m, cams_m, tars_m, "cpp")
    
    # 4. Results
    print(f"\nTraining Benchmark (Fern):")
    print(f"  {'Framework':<15} | {'Mode':<10} | {'Avg Time':<10} | {'Speed':<10}")
    print(f"  {'-'*55}")
    print(f"  {'PyTorch':<15} | {'Python':<10} | {py_torch:8.4f}s | {1/py_torch:6.2f} it/s")
    print(f"  {'PyTorch':<15} | {'C++':<10} | {cpp_torch:8.4f}s | {1/cpp_torch:6.2f} it/s")
    print(f"  {'MLX':<15} | {'Python':<10} | {py_mlx:8.4f}s | {1/py_mlx:6.2f} it/s")
    print(f"  {'MLX':<15} | {'C++/Metal':<10} | {cpp_mlx:8.4f}s | {1/cpp_mlx:6.2f} it/s")
    
    assert cpp_mlx < 1.0 # Sanity check

if __name__ == "__main__":
    test_benchmark_training_fern()

import mlx.core as mx
import mlx.optimizers as optim_mlx
import torch
import torch.optim as optim_torch
import time
import os
import numpy as np
import argparse

# MLX GS
from mlx_gs.core.gaussians import init_gaussians_from_pcd as init_mlx
from mlx_gs.io.colmap import load_colmap_dataset as load_mlx
from mlx_gs.training.trainer import train_step as step_mlx

# Torch GS
from torch_gs.core.gaussians import init_gaussians_from_pcd as init_torch
from torch_gs.io.colmap import load_colmap_dataset as load_torch
from torch_gs.training.trainer import train_step as step_torch

DATA_DIR = "/Users/mghifary/Work/Code/AI/data/gsplat"

def benchmark_mlx(num_runs=10, rasterizer_type="python"):
    path = os.path.join(DATA_DIR, "nerf_example_data/nerf_llff_data/fern")
    xyz, rgb, cameras, targets = load_mlx(path, "images_8")
    gaussians = init_mlx(xyz, rgb)
    optimizer = optim_mlx.Adam(learning_rate=0.001)
    
    # Convert Gaussians dataclass to dictionary for MLX transform
    # This is necessary because mx.value_and_grad expects a tree of arrays
    params = {
        "means": gaussians.means,
        "scales": gaussians.scales,
        "quaternions": gaussians.quaternions,
        "opacities": gaussians.opacities,
        "sh_coeffs": gaussians.sh_coeffs
    }
    
    # Warm-up
    print(f"MLX ({rasterizer_type}): Warming up...")
    step_mlx(params, optimizer, targets[0], cameras[0], rasterizer_type=rasterizer_type)
    mx.eval(params["means"])
    
    times = []
    print(f"MLX ({rasterizer_type}): Running {num_runs} iterations...")
    for i in range(num_runs):
        idx = i % len(cameras)
        start = time.perf_counter()
        # Returns: loss, rendered_image, psnr, grad_norms
        loss, _, _, _ = step_mlx(params, optimizer, targets[idx], cameras[idx], rasterizer_type=rasterizer_type) 
        mx.eval(loss, params["means"]) # Force sync
        end = time.perf_counter()
        times.append(end - start)
        print(f"  MLX ({rasterizer_type}) Iter {i+1}: {end-start:.4f}s")
        
    # Skip first 2 iterations for average to account for JIT/Metal overhead
    avg = sum(times[2:]) / (num_runs - 2) if num_runs > 2 else sum(times) / num_runs
    return avg

def benchmark_torch(num_runs=10, rasterizer_type="python"):
    device = "mps"
    path = os.path.join(DATA_DIR, "nerf_example_data/nerf_llff_data/fern")
    xyz, rgb, cameras, targets = load_torch(path, "images_8", device=device)
    gaussians = init_torch(torch.tensor(xyz, dtype=torch.float32), torch.tensor(rgb, dtype=torch.float32), device=device)
    
    gaussians.means.requires_grad = True
    gaussians.scales.requires_grad = True
    gaussians.quaternions.requires_grad = True
    gaussians.opacities.requires_grad = True
    gaussians.sh_coeffs.requires_grad = True
    
    optimizer = optim_torch.Adam([gaussians.means, gaussians.scales, gaussians.quaternions, gaussians.opacities, gaussians.sh_coeffs], lr=0.001)
    
    # Warm-up
    print(f"Torch ({rasterizer_type}): Warming up...")
    step_torch(gaussians, optimizer, targets[0], cameras[0], device=device, rasterizer_type=rasterizer_type)
    torch.mps.synchronize()
    
    times = []
    print(f"Torch ({rasterizer_type}): Running {num_runs} iterations...")
    for i in range(num_runs):
        idx = i % len(cameras)
        start = time.perf_counter()
        loss, _, _ = step_torch(gaussians, optimizer, targets[idx], cameras[idx], device=device, rasterizer_type=rasterizer_type)
        torch.mps.synchronize()
        end = time.perf_counter()
        times.append(end - start)
        print(f"  Torch ({rasterizer_type}) Iter {i+1}: {end-start:.4f}s")
        
    # Skip first 2 iterations for average
    avg = sum(times[2:]) / (num_runs - 2) if num_runs > 2 else sum(times) / num_runs
    return avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=10)
    args = parser.parse_args()
    
    print("=== Gaussian Splatting Framework Benchmark ===")
    
    torch_py_avg = benchmark_torch(args.num_runs, rasterizer_type="python")
    try:
        torch_cpp_avg = benchmark_torch(args.num_runs, rasterizer_type="cpp")
    except Exception as e:
        print(f"Torch CPP Benchmark Failed: {e}")
        torch_cpp_avg = None
        
    mlx_py_avg = benchmark_mlx(args.num_runs, rasterizer_type="python")
    try:
        mlx_cpp_avg = benchmark_mlx(args.num_runs, rasterizer_type="cpp")
    except Exception as e:
        print(f"MLX CPP Benchmark Failed: {e}")
        mlx_cpp_avg = None
    
    print("\nBenchmark Summary:")
    print(f"  PyTorch (Python): {torch_py_avg:.4f}s/it ({1/torch_py_avg:.2f} it/s)")
    if torch_cpp_avg:
        print(f"  PyTorch (C++):    {torch_cpp_avg:.4f}s/it ({1/torch_cpp_avg:.2f} it/s)")
    print(f"  MLX (Python):     {mlx_py_avg:.4f}s/it ({1/mlx_py_avg:.2f} it/s)")
    if mlx_cpp_avg:
        print(f"  MLX (C++/Metal):  {mlx_cpp_avg:.4f}s/it ({1/mlx_cpp_avg:.2f} it/s)")
    
    print("\nComparisons:")
    if mlx_cpp_avg:
        speedup_mlx = mlx_py_avg / mlx_cpp_avg
        print(f"  MLX C++ is {speedup_mlx:.2f}x faster than MLX Python")
        
        speedup_vs_torch_py = torch_py_avg / mlx_cpp_avg
        print(f"  MLX C++ is {speedup_vs_torch_py:.2f}x faster than PyTorch (Python)")
        
        if torch_cpp_avg:
            speedup_vs_torch_cpp = torch_cpp_avg / mlx_cpp_avg
            if speedup_vs_torch_cpp > 1.0:
                print(f"  MLX C++ is {speedup_vs_torch_cpp:.2f}x faster than PyTorch (C++)")
            else:
                print(f"  PyTorch (C++) is {1/speedup_vs_torch_cpp:.2f}x faster than MLX C++")

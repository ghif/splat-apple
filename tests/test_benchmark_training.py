"""
Performance Benchmark for Gaussian Splatting Training.
This script measures the execution time of full training iterations
including projection, rasterization, and backpropagation.

It is designed to compare performance between different implementations
and hardware backends (MPS, CUDA, CPU).
"""
import torch
import torch.optim as optim
import time
import os
import pytest
import numpy as np
from torch_gs.core.gaussians import init_gaussians_from_pcd
from torch_gs.io.colmap import load_colmap_dataset
from torch_gs.training.trainer import train_step, Camera

# Dataset Path
DATA_DIR = "/Users/mghifary/Work/Code/AI/data/gsplat"

def run_benchmark(gaussians, optimizer, cameras, targets, device, rasterizer_type):
    """Runs a benchmark for a specific rasterizer type."""
    num_runs = 10
    print(f"\nBenchmarking {rasterizer_type.upper()} rasterizer ({num_runs} iterations)...")
    
    # Warm-up
    camera = cameras[0]
    target = targets[0]
    train_step(gaussians, optimizer, target, camera, device=device, rasterizer_type=rasterizer_type)
    torch.mps.synchronize()
    
    times = []
    for i in range(num_runs):
        cam_idx = i % len(cameras)
        curr_cam, curr_target = cameras[cam_idx], targets[cam_idx]
        
        start = time.perf_counter()
        train_step(gaussians, optimizer, curr_target, curr_cam, device=device, rasterizer_type=rasterizer_type)
        torch.mps.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
        print(f"  Iteration {i+1}: {end - start:.4f}s")
    
    avg_time = sum(times) / num_runs
    it_per_sec = 1.0 / avg_time
    return avg_time, it_per_sec

def test_benchmark_training_fern():
    """
    Main benchmark entry point for the 'Fern' scene.
    Compares Python vs C++ rasterizer on MPS.
    """
    device = "mps"
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available.")
        
    print(f"\nRunning benchmark on device: {device}")
    
    path = os.path.join(DATA_DIR, "nerf_example_data/nerf_llff_data/fern")
    if not os.path.exists(path):
        pytest.skip(f"Fern dataset not found at {path}.")
        
    # 1. Pipeline Setup
    print(f"\nLoading Fern dataset for training benchmark from {path}...")
    try:
        xyz, rgb, cameras, targets = load_colmap_dataset(path, "images_8", device=device)
    except Exception as e:
        pytest.skip(f"Failed to load dataset: {e}")
    
    if len(cameras) == 0:
        pytest.skip("No cameras found in dataset.")

    num_points = xyz.shape[0]
    W, H = cameras[0].W, cameras[0].H
    
    # Initialize optimization targets
    gaussians = init_gaussians_from_pcd(torch.tensor(xyz, dtype=torch.float32), torch.tensor(rgb, dtype=torch.float32), device=device)
    
    # Enable gradients
    gaussians.means.requires_grad = True
    gaussians.scales.requires_grad = True
    gaussians.quaternions.requires_grad = True
    gaussians.opacities.requires_grad = True
    gaussians.sh_coeffs.requires_grad = True
    
    optimizer = optim.Adam([
        {'params': [gaussians.means], 'lr': 0.00016},
        {'params': [gaussians.scales], 'lr': 0.005},
        {'params': [gaussians.quaternions], 'lr': 0.001},
        {'params': [gaussians.opacities], 'lr': 0.05},
        {'params': [gaussians.sh_coeffs], 'lr': 0.0025}
    ], lr=0.001, eps=1e-15)
    
    print(f"Benchmarking training step with {num_points} Gaussians at {W}x{H} resolution...")
    
    # Run Python Benchmark
    py_avg, py_its = run_benchmark(gaussians, optimizer, cameras, targets, device, "python")
    
    # Run C++ Benchmark
    cpp_avg, cpp_its = run_benchmark(gaussians, optimizer, cameras, targets, device, "cpp")
    
    # 4. Result Summarization
    print(f"\nTraining Benchmark Comparison (Fern):")
    print(f"  {'Rasterizer':<12} | {'Avg Time':<10} | {'Speed':<10}")
    print(f"  {'-'*35}")
    print(f"  {'Python':<12} | {py_avg:8.4f}s | {py_its:6.2f} it/s")
    print(f"  {'C++':<12} | {cpp_avg:8.4f}s | {cpp_its:6.2f} it/s")
    
    speedup = py_avg / cpp_avg
    print(f"\n  C++ Speedup: {speedup:.2f}x")
    
    assert cpp_avg < 10.0 # Sanity check
    
if __name__ == "__main__":
    test_benchmark_training_fern()

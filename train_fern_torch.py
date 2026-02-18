"""
Gaussian Splatting Training Script for the 'Fern' Dataset.
This script orchestrates the full pipeline:
1. Data loading (COLMAP sparse reconstruction).
2. Gaussian initialization from point cloud.
3. Adam optimization loop with combined L1 + SSIM loss.
4. Progress monitoring (PSNR logging and periodic rendering).
5. Final model export to a PLY file compatible with the bundled viewer.
"""
import torch
import torch.optim as optim
import numpy as np
import os
import datetime
import random
import argparse
from tqdm import tqdm
from PIL import Image

# Import package modules
from torch_gs.core.gaussians import init_gaussians_from_pcd, Gaussians
from torch_gs.renderer.renderer import render
from torch_gs.io.colmap import load_colmap_dataset
from torch_gs.training.trainer import train_step, Camera

# Configuration
DATA_DIR = "/Users/mghifary/Work/Code/AI/data/gsplat"

def save_ply(path, gaussians):
    """
    Export current Gaussians to a PLY file.
    The format follows the standard 3D Gaussian Splatting convention,
    including positions, normals (unused), colors (SH DC), opacities, 
    scales, and orientation quaternions.
    
    Args:
        path (str): Output file path (.ply).
        gaussians (Gaussians): The Gaussian collection to save.
    """
    from plyfile import PlyData, PlyElement
    
    # Extract data to CPU/Numpy for file writing
    xyz = gaussians.means.cpu().detach().numpy()
    normals = np.zeros_like(xyz)
    # Use only DC component for color export
    f_dc = gaussians.sh_coeffs[:, 0, :].cpu().detach().numpy().flatten().reshape(-1, 3)
    opacities = gaussians.opacities.cpu().detach().numpy()
    scales = gaussians.scales.cpu().detach().numpy()
    rot = gaussians.quaternions.cpu().detach().numpy()
    
    # Define PLY property structure
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]
    
    # Fill elements
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements['nx'], elements['ny'], elements['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
    elements['f_dc_0'], elements['f_dc_1'], elements['f_dc_2'] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    elements['opacity'] = opacities[:, 0]
    elements['scale_0'], elements['scale_1'], elements['scale_2'] = scales[:, 0], scales[:, 1], scales[:, 2]
    elements['rot_0'], elements['rot_1'], elements['rot_2'], elements['rot_3'] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    
    # Write file
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def run_training(num_iterations: int = 2000, device="mps", rasterizer_type="python"):
    """
    Orchestrates the training process on a specific dataset.
    
    Args:
        num_iterations (int): Total training steps.
        device (str): Device to use for optimization.
        rasterizer_type (str): "python" or "cpp".
    """
    # 1. Load Data: Retrieve images and calibrated cameras
    path = os.path.join(DATA_DIR, "nerf_example_data/nerf_llff_data/fern")
    print(f"Loading data from {path}...")
    try:
        xyz, rgb, cameras, targets = load_colmap_dataset(path, "images_8", device=device)
    except Exception as e:
        print(f"Error loading colmap data: {e}.")
        xyz, rgb, cameras, targets = np.array([]), np.array([]), [], []
        
    # Validation / Mock fallback for environment testing
    if len(xyz) == 0:
        print("Data not found or empty. Using random initialization for verification.")
        xyz = (np.random.rand(1000, 3) - 0.5) * 5.0
        rgb = np.random.rand(1000, 3)
        cameras = []
        targets = []
        for i in range(10):
            w2c = torch.eye(4, device=device)
            w2c[2, 3] = 5.0 
            cam = Camera(W=800, H=800, fx=800.0, fy=800.0, cx=400.0, cy=400.0, W2C=w2c)
            cameras.append(cam)
            targets.append(torch.zeros((800, 800, 3), device=device))
            
    print(f"Loaded {len(xyz)} points")
    print(f"Prepared {len(cameras)} cameras for training")
    
    # 2. Initialize Gaussians from the sparse point cloud
    gaussians_src = init_gaussians_from_pcd(torch.tensor(xyz, dtype=torch.float32), torch.tensor(rgb, dtype=torch.float32), device=device)
    
    # Make all parameters differentiable
    gaussians_src.means.requires_grad = True
    gaussians_src.scales.requires_grad = True
    gaussians_src.quaternions.requires_grad = True
    gaussians_src.opacities.requires_grad = True
    gaussians_src.sh_coeffs.requires_grad = True
    
    # 3. Setup Optimizer
    # We use Adam with specific learning rates for different Gaussian attributes.
    # Means usually have the smallest LR to maintain structural stability.
    optimizer = optim.Adam([
        {'params': [gaussians_src.means], 'lr': 0.00016, "name": "means"},
        {'params': [gaussians_src.scales], 'lr': 0.005, "name": "scales"},
        {'params': [gaussians_src.quaternions], 'lr': 0.001, "name": "quats"},
        {'params': [gaussians_src.opacities], 'lr': 0.05, "name": "opacities"},
        {'params': [gaussians_src.sh_coeffs], 'lr': 0.0025, "name": "sh_coeffs"}
    ], lr=0.001, eps=1e-15)
    
    # 4. Preparation for Logging
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"fern_torch_{timestamp}")
    if rasterizer_type == "cpp":
        output_dir = os.path.join("results", f"fern_torch_cpp_{timestamp}")
        
    progress_dir = os.path.join(output_dir, "progress")
    ply_dir = os.path.join(output_dir, "ply")
    os.makedirs(progress_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    pbar = tqdm(range(num_iterations))
    
    # 5. Optimization Loop
    for i in pbar:
        # Pick a random camera/target pair for this iteration
        idx = random.randint(0, len(cameras)-1)
        cam, target = cameras[idx], targets[idx]
        
        # Perform single differentiable training step
        loss, psnr, rendered_image = train_step(
            gaussians_src, optimizer, target, cam, 
            device=device, rasterizer_type=rasterizer_type
        )
        
        if i % 10 == 0:
            pbar.set_description(f"Loss: {loss:.4f} PSNR: {psnr:.2f}")
            
            # Periodic visualization: Save a preview image of the current state
            if i % 100 == 0:
                img_np = rendered_image.detach().cpu().numpy()
                Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).save(
                    os.path.join(progress_dir, f"progress_{i:04d}.png")
                )
    
    # Final cleanup and model export
    print("Training done. Saving final model...")
    try:
        save_ply(os.path.join(ply_dir, "fern_final.ply"), gaussians_src)
    except ImportError:
        print("plyfile not installed, skipping PLY save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gaussian Splatting on MPS")
    parser.add_argument("--num_iterations", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cuda/cpu)")
    parser.add_argument("--rasterizer", type=str, default="python", choices=["python", "cpp"], help="Rasterizer version")
    args = parser.parse_args()
    
    run_training(num_iterations=args.num_iterations, device=args.device, rasterizer_type=args.rasterizer)

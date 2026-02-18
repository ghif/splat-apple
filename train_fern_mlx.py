import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import os
import datetime
import random
import argparse
from tqdm import tqdm
from PIL import Image

# Import MLX GS modules
from mlx_gs.core.gaussians import init_gaussians_from_pcd, Gaussians
from mlx_gs.renderer.renderer import render
from mlx_gs.io.colmap import load_colmap_dataset
from mlx_gs.training.trainer import train_step, Camera

# Configuration
DATA_DIR = "/Users/mghifary/Work/Code/AI/data/gsplat"

def save_ply(path, gaussians):
    """
    Export MLX Gaussians to PLY.
    """
    from plyfile import PlyData, PlyElement
    
    # Convert MLX to Numpy
    xyz = np.array(gaussians.means)
    normals = np.zeros_like(xyz)
    f_dc = np.array(gaussians.sh_coeffs[:, 0, :]).flatten().reshape(-1, 3)
    opacities = np.array(gaussians.opacities)
    scales = np.array(gaussians.scales)
    rot = np.array(gaussians.quaternions)
    
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]
    
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements['nx'], elements['ny'], elements['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
    elements['f_dc_0'], elements['f_dc_1'], elements['f_dc_2'] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    elements['opacity'] = opacities[:, 0]
    elements['scale_0'], elements['scale_1'], elements['scale_2'] = scales[:, 0], scales[:, 1], scales[:, 2]
    elements['rot_0'], elements['rot_1'], elements['rot_2'], elements['rot_3'] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def run_training(num_iterations: int = 2000, rasterizer_type="python"):
    """
    MLX training loop.
    """
    # 1. Load Data
    path = os.path.join(DATA_DIR, "intro_to_deep_learning_with_mlx_data/ferns")
    if not os.path.exists(path):
        # Fallback to standard path if my specific user path doesn't exist
        path = os.path.join(DATA_DIR, "nerf_example_data/nerf_llff_data/fern")
    
    print(f"Loading data from {path}...")
    xyz, rgb, cameras, targets = load_colmap_dataset(path, "images_8")
    
    # Optional: Slice for faster debugging if needed, but enabling full data for now
    # indices = np.random.choice(xyz.shape[0], 1000, replace=False)
    # xyz = xyz[indices]
    # rgb = rgb[indices]

    print(f"Loaded {len(xyz)} points")
    print(f"Prepared {len(cameras)} cameras")
    
    # 2. Initialize Gaussians
    gaussians = init_gaussians_from_pcd(xyz, rgb)
    
    # Convert to parameter dict for MLX training
    params = {
        "means": gaussians.means,
        "scales": gaussians.scales,
        "quaternions": gaussians.quaternions,
        "opacities": gaussians.opacities,
        "sh_coeffs": gaussians.sh_coeffs
    }
    
    # 3. Setup Optimizers
    optimizers = {
        "means": optim.Adam(learning_rate=0.00016),
        "scales": optim.Adam(learning_rate=0.005),
        "quaternions": optim.Adam(learning_rate=0.001),
        "opacities": optim.Adam(learning_rate=0.05),
        "sh_coeffs": optim.Adam(learning_rate=0.0025)
    }
    
    # 4. Preparation for Logging
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"fern_mlx_{timestamp}")
    if rasterizer_type == "cpp":
        output_dir = os.path.join("results", f"fern_mlx_cpp_{timestamp}")
    progress_dir = os.path.join(output_dir, "progress")
    ply_dir = os.path.join(output_dir, "ply")
    os.makedirs(progress_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    pbar = tqdm(range(num_iterations))
    
    # 5. Optimization Loop
    for i in pbar:
        idx = random.randint(0, len(cameras)-1)
        cam, target = cameras[idx], targets[idx]
        
        loss, rendered_image, psnr, grad_norms = train_step(params, optimizers, target, cam, lambda_ssim=0.2, rasterizer_type=rasterizer_type)
        
        mx.eval(params, loss, psnr)
        
        if i % 10 == 0:
            if mx.isnan(loss):
                print(f"\nIteration {i}: NaN detected in loss!")
                break
            
            pbar.set_description(f"Loss: {loss.item():.4f} PSNR: {psnr.item():.2f}")
            
            if i % 100 == 0:
                img_np = np.array(rendered_image)
                Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).save(
                    os.path.join(progress_dir, f"progress_{i:04d}.png")
                )
    
    # Export final
    gaussians_final = Gaussians(**params)
    print("Training done. Saving final model...")
    save_ply(os.path.join(ply_dir, "fern_final.ply"), gaussians_final)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gaussian Splatting on MLX")
    parser.add_argument("--num_iterations", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--rasterizer", type=str, default="python", choices=["python", "cpp"], help="Rasterizer version")
    args = parser.parse_args()
    
    run_training(num_iterations=args.num_iterations, rasterizer_type=args.rasterizer)

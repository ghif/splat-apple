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

def run_training(
    data_dir: str, 
    img_folder: str = "images_8",
    num_iterations: int = 2000, 
    rasterizer_type: str = "python",
    normalize_scene: bool = False
):
    """
    Generic MLX training loop.
    """
    # 1. Load Data
    print(f"Loading data from {data_dir}...")
    xyz, rgb, cameras, targets = load_colmap_dataset(data_dir, img_folder)
    
    # Optional Scene Normalization (useful for 360 scenes)
    if normalize_scene:
        cam_centers = []
        for cam in cameras:
            w2c = np.array(cam.W2C)
            R = w2c[:3, :3]
            T = w2c[:3, 3]
            C = -R.T @ T
            cam_centers.append(C)
        cam_centers = np.array(cam_centers)
        
        centroid = np.mean(cam_centers, axis=0)
        avg_dist = np.mean(np.linalg.norm(cam_centers - centroid, axis=1))
        scale = 1.0 / (avg_dist + 1e-6)
        
        print(f"Normalizing scene: centroid={centroid}, scale={scale}")
        xyz = (xyz - centroid) * scale
        
        for cam in cameras:
            w2c = np.array(cam.W2C)
            R = w2c[:3, :3]
            T = w2c[:3, 3]
            new_T = (R @ centroid + T) * scale
            new_w2c = np.eye(4)
            new_w2c[:3, :3] = R
            new_w2c[:3, 3] = new_T
            cam.W2C = mx.array(new_w2c, dtype=mx.float32)

    # Handle zero colors: if all are zero, initialize with random colors
    if np.all(rgb == 0):
        print("Detected zero colors in point cloud. Initializing with random colors.")
        rgb = np.random.uniform(0.4, 0.6, size=rgb.shape)

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
    dataset_name = os.path.basename(data_dir.rstrip("/"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"{dataset_name}_mlx_{timestamp}")
    if rasterizer_type == "cpp":
        output_dir = os.path.join("results", f"{dataset_name}_mlx_cpp_{timestamp}")
    
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
    save_ply(os.path.join(ply_dir, f"{dataset_name}_final.ply"), gaussians_final)
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gaussian Splatting on MLX (Generic)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--img_folder", type=str, default="images_8", help="Name of the image folder (e.g., images, images_4, images_8)")
    parser.add_argument("--num_iterations", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--rasterizer", type=str, default="python", choices=["python", "cpp"], help="Rasterizer version")
    parser.add_argument("--normalize", action="store_true", help="Apply scene normalization (recommended for 360 scenes)")
    
    args = parser.parse_args()
    
    run_training(
        data_dir=args.data_dir,
        img_folder=args.img_folder,
        num_iterations=args.num_iterations,
        rasterizer_type=args.rasterizer,
        normalize_scene=args.normalize
    )

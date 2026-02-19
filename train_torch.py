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

def save_ply(path, gaussians):
    """
    Export current Gaussians to a PLY file.
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("plyfile not installed, skipping PLY save.")
        return
    
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

def run_training(
    data_dir: str,
    img_folder: str = "images_8",
    num_iterations: int = 2000, 
    device="mps", 
    rasterizer_type="python",
    normalize_scene: bool = False
):
    """
    Generic PyTorch training loop.
    """
    # 1. Load Data
    print(f"Loading data from {data_dir}...")
    try:
        xyz, rgb, cameras, targets = load_colmap_dataset(data_dir, img_folder, device=device)
    except Exception as e:
        print(f"Error loading colmap data: {e}.")
        return

    # Optional Scene Normalization (useful for 360 scenes)
    if normalize_scene:
        cam_centers = []
        for cam in cameras:
            w2c = cam.W2C.detach().cpu().numpy()
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
            w2c = cam.W2C.detach().cpu().numpy()
            R = w2c[:3, :3]
            T = w2c[:3, 3]
            new_T = (R @ centroid + T) * scale
            new_w2c = np.eye(4)
            new_w2c[:3, :3] = R
            new_w2c[:3, 3] = new_T
            cam.W2C = torch.tensor(new_w2c, dtype=torch.float32, device=device)

    # Handle zero colors: if all are zero, initialize with random colors
    if np.all(rgb == 0):
        print("Detected zero colors in point cloud. Initializing with random colors.")
        rgb = np.random.uniform(0.4, 0.6, size=rgb.shape)

    print(f"Loaded {len(xyz)} points")
    print(f"Prepared {len(cameras)} cameras")
    
    # 2. Initialize Gaussians
    gaussians = init_gaussians_from_pcd(
        torch.tensor(xyz, dtype=torch.float32), 
        torch.tensor(rgb, dtype=torch.float32), 
        device=device
    )
    
    # Make all parameters differentiable
    gaussians.means.requires_grad = True
    gaussians.scales.requires_grad = True
    gaussians.quaternions.requires_grad = True
    gaussians.opacities.requires_grad = True
    gaussians.sh_coeffs.requires_grad = True
    
    # 3. Setup Optimizer
    optimizer = optim.Adam([
        {'params': [gaussians.means], 'lr': 0.00016, "name": "means"},
        {'params': [gaussians.scales], 'lr': 0.005, "name": "scales"},
        {'params': [gaussians.quaternions], 'lr': 0.001, "name": "quats"},
        {'params': [gaussians.opacities], 'lr': 0.05, "name": "opacities"},
        {'params': [gaussians.sh_coeffs], 'lr': 0.0025, "name": "sh_coeffs"}
    ], lr=0.001, eps=1e-15)
    
    # 4. Preparation for Logging
    os.makedirs("results", exist_ok=True)
    dataset_name = os.path.basename(data_dir.rstrip("/"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"{dataset_name}_torch_{timestamp}")
    if rasterizer_type == "cpp":
        output_dir = os.path.join("results", f"{dataset_name}_torch_cpp_{timestamp}")
        
    progress_dir = os.path.join(output_dir, "progress")
    ply_dir = os.path.join(output_dir, "ply")
    os.makedirs(progress_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    pbar = tqdm(range(num_iterations))
    
    # 5. Optimization Loop
    for i in pbar:
        # Pick a random camera/target pair
        idx = random.randint(0, len(cameras)-1)
        cam, target = cameras[idx], targets[idx]
        
        # Single training step
        loss, psnr, rendered_image = train_step(
            gaussians, optimizer, target, cam, 
            device=device, rasterizer_type=rasterizer_type
        )
        
        if i % 10 == 0:
            if np.isnan(loss):
                print(f"\nIteration {i}: NaN detected in loss!")
                break
                
            pbar.set_description(f"Loss: {loss:.4f} PSNR: {psnr:.2f}")
            
            if i % 100 == 0:
                img_np = rendered_image.detach().cpu().numpy()
                Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).save(
                    os.path.join(progress_dir, f"progress_{i:04d}.png")
                )
    
    # Final cleanup and model export
    print("Training done. Saving final model...")
    save_ply(os.path.join(ply_dir, f"{dataset_name}_final.ply"), gaussians)
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gaussian Splatting on PyTorch (Generic)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--img_folder", type=str, default="images_8", help="Name of the image folder (e.g., images, images_4, images_8)")
    parser.add_argument("--num_iterations", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cuda/cpu)")
    parser.add_argument("--rasterizer", type=str, default="python", choices=["python", "cpp"], help="Rasterizer version")
    parser.add_argument("--normalize", action="store_true", help="Apply scene normalization (recommended for 360 scenes)")
    
    args = parser.parse_args()
    
    run_training(
        data_dir=args.data_dir,
        img_folder=args.img_folder,
        num_iterations=args.num_iterations,
        device=args.device,
        rasterizer_type=args.rasterizer,
        normalize_scene=args.normalize
    )

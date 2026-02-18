"""
COLMAP Data Loader for Gaussian Splatting.
This module parses COLMAP binary output files (sparse reconstruction)
and prepares the data for training, including point clouds and camera parameters.
"""
import struct
import numpy as np
import torch
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
from torch_gs.training.trainer import Camera
from PIL import Image

@dataclass
class CameraInfo:
    """Intermediate data structure for raw COLMAP camera data."""
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image: np.ndarray
    image_path: str
    image_name: str
    width: int
    height: int

def read_cameras_binary(path_to_model_file):
    """Parses cameras.bin from COLMAP."""
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("<iiQQ", fid.read(24))
            camera_id, model_id, width, height = camera_properties
            
            # Model mapping: COLMAP uses specific IDs for camera models
            if model_id == 0: # SIMPLE_PINHOLE
                params = struct.unpack("<3d", fid.read(24))
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model_id == 1: # PINHOLE
                params = struct.unpack("<4d", fid.read(32))
                fx, fy, cx, cy = params
            elif model_id == 2: # SIMPLE_RADIAL
                params = struct.unpack("<4d", fid.read(32))
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                raise NotImplementedError(f"Camera model {model_id} not implemented")

            cameras[camera_id] = {
                "id": camera_id,
                "model": "PINHOLE", # We treat all as pinhole for simplicity
                "width": width,
                "height": height,
                "params": (fx, fy, cx, cy)
            }
    return cameras

def read_images_binary(path_to_model_file):
    """Parses images.bin from COLMAP (extrinsics and registrations)."""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_properties = struct.unpack("<idddddddi", fid.read(64))
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = struct.unpack("<c", fid.read(1))[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = struct.unpack("<c", fid.read(1))[0]
                
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            fid.read(num_points2D * 24) # Skip 2D features
            
            images[image_id] = {
                "id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name
            }
    return images

def read_points3D_binary(path_to_model_file):
    """Parses points3D.bin from COLMAP (the sparse point cloud)."""
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            binary_point_properties = struct.unpack("<QdddBBBd", fid.read(43))
            point3D_id = binary_point_properties[0]
            xyz = np.array(binary_point_properties[1:4])
            rgb = np.array(binary_point_properties[4:7])
            error = binary_point_properties[7]
            track_length = struct.unpack("<Q", fid.read(8))[0]
            fid.read(track_length * 8) # Skip track info
            
            points3D[point3D_id] = {
                "id": point3D_id,
                "xyz": xyz,
                "rgb": rgb,
                "error": error
            }
    return points3D

def qvec2rotmat(qvec):
    """Converts COLMAP quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def load_colmap_data_raw(source_path, images_dir_name="images_8"):
    """
    Internal loader for COLMAP binary data.
    Parses intrinsics, extrinsics, and the initial point cloud.
    
    Args:
        source_path (str): Root directory of the dataset.
        images_dir_name (str): Subdirectory containing undistorted images.
        
    Returns:
        tuple: (xyz_points, rgb_colors, list_of_CameraInfo)
    """
    cameras_extrinsic_file = os.path.join(source_path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(source_path, "sparse/0", "cameras.bin")
    points3D_file = os.path.join(source_path, "sparse/0", "points3D.bin")
    
    try:
        cam_extrinsics = read_images_binary(cameras_extrinsic_file)
        cam_intrinsics = read_cameras_binary(cameras_intrinsic_file)
        points3D_data = read_points3D_binary(points3D_file)
    except FileNotFoundError:
        print(f"COLMAP binary files not found in {source_path}/sparse/0/")
        return np.array([]), np.array([]), []
    
    xyz = np.array([p["xyz"] for p in points3D_data.values()])
    rgb = np.array([p["rgb"] for p in points3D_data.values()]) / 255.0
    
    train_cameras = []
    sorted_image_ids = sorted(cam_extrinsics.keys())
    images_dir = os.path.join(source_path, images_dir_name)
    
    # Path resolution fallback
    if not os.path.exists(images_dir):
        for fallback in ["images_4", "images"]:
            if os.path.exists(os.path.join(source_path, fallback)):
                images_dir = os.path.join(source_path, fallback)
                break
             
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Check if COLMAP recorded relative paths or if we need index-based mapping
    first_name = cam_extrinsics[sorted_image_ids[0]]["name"]
    use_index_mapping = not os.path.exists(os.path.join(images_dir, first_name))
    
    for i, image_id in enumerate(sorted_image_ids):
        extr = cam_extrinsics[image_id]
        intr = cam_intrinsics[extr["camera_id"]]
        
        R = qvec2rotmat(extr["qvec"])
        T = extr["tvec"]
        
        focal_length_x, focal_length_y = intr["params"][0], intr["params"][1]
        width, height = intr["width"], intr["height"]

        image_name = image_files[i] if use_index_mapping and i < len(image_files) else extr["name"]
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path): continue
            
        image_pil = Image.open(image_path)
        image = np.array(image_pil)
        
        # Scaling correction: If image size on disk differs from COLMAP metadata
        if image.shape[1] != width or image.shape[0] != height:
             scale_x, scale_y = image.shape[1] / width, image.shape[0] / height
             focal_length_x *= scale_x
             focal_length_y *= scale_y
             width, height = image.shape[1], image.shape[0]
        
        # Compute FOV values for internal representation
        fovx = 2 * math.atan(width / (2 * focal_length_x))
        fovy = 2 * math.atan(height / (2 * focal_length_y))
        
        train_cameras.append(CameraInfo(
            uid=image_id, R=R, T=T, FovY=fovy, FovX=fovx,
            image=image / 255.0, image_path=image_path, image_name=image_name,
            width=width, height=height
        ))
        
    return xyz, rgb, train_cameras

def load_colmap_dataset(path: str, images_subdir: str = "images_8", device="mps"):
    """
    Load COLMAP data and convert to package-standard Camera entities.
    This is the main entry point for data loading.
    
    Args:
        path (str): Root dataset path.
        images_subdir (str): Folder name for the images.
        device (str): Destination device.
        
    Returns:
        tuple: (xyz_initial, rgb_initial, list_of_Cameras, list_of_target_tensors)
    """
    xyz, rgb, train_cam_infos = load_colmap_data_raw(path, images_subdir)
    
    cameras = []
    targets = []
    
    for info in train_cam_infos:
        # Convert FOV back to focal lengths if needed, or use directly
        fx = info.width / (2 * math.tan(info.FovX / 2))
        fy = info.height / (2 * math.tan(info.FovY / 2))
        cx, cy = info.width / 2.0, info.height / 2.0
        
        # Construct World-to-Camera 4x4 homogenous matrix
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = info.R, info.T
        
        camera = Camera(
            W=info.width, H=info.height,
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            W2C=torch.tensor(w2c, dtype=torch.float32, device=device), 
            full_proj=torch.eye(4, device=device) # Identity for now
        )
        cameras.append(camera)
        targets.append(torch.tensor(info.image, dtype=torch.float32, device=device))
        
    return xyz, rgb, cameras, targets

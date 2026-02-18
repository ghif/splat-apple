import struct
import numpy as np
import jax.numpy as jnp
from PIL import Image
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
from jax_gs.core.camera import Camera

@dataclass
class CameraInfo:
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
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("<iiQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            if model_id == 0: # SIMPLE_PINHOLE
                params = struct.unpack("<3d", fid.read(24))
                f = params[0]
                cx = params[1]
                cy = params[2]
                fx = f
                fy = f
            elif model_id == 1: # PINHOLE
                params = struct.unpack("<4d", fid.read(32))
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
            elif model_id == 2: # SIMPLE_RADIAL
                params = struct.unpack("<4d", fid.read(32))
                f = params[0]
                cx = params[1]
                cy = params[2]
                k = params[3]
                fx = f
                fy = f
            else:
                raise NotImplementedError(f"Camera model {model_id} not implemented")

            cameras[camera_id] = {
                "id": camera_id,
                "model": "PINHOLE",
                "width": width,
                "height": height,
                "params": (fx, fy, cx, cy)
            }
    return cameras

def read_images_binary(path_to_model_file):
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
            fid.read(num_points2D * 24) # Skip points2D data
            
            images[image_id] = {
                "id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name
            }
    return images

def read_points3D_binary(path_to_model_file):
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

def load_colmap_data_raw(source_path, images_dir_name="images_4"):
    """
    Internal loader for COLMAP binary data.
    """
    cameras_extrinsic_file = os.path.join(source_path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(source_path, "sparse/0", "cameras.bin")
    points3D_file = os.path.join(source_path, "sparse/0", "points3D.bin")
    
    cam_extrinsics = read_images_binary(cameras_extrinsic_file)
    cam_intrinsics = read_cameras_binary(cameras_intrinsic_file)
    points3D_data = read_points3D_binary(points3D_file)
    
    xyz = np.array([p["xyz"] for p in points3D_data.values()])
    rgb = np.array([p["rgb"] for p in points3D_data.values()]) / 255.0
    
    train_cameras = []
    sorted_image_ids = sorted(cam_extrinsics.keys())
    images_dir = os.path.join(source_path, images_dir_name)
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    first_name = cam_extrinsics[sorted_image_ids[0]]["name"]
    use_index_mapping = not os.path.exists(os.path.join(images_dir, first_name))
    
    for i, image_id in enumerate(sorted_image_ids):
        extr = cam_extrinsics[image_id]
        intr = cam_intrinsics[extr["camera_id"]]
        
        R = qvec2rotmat(extr["qvec"])
        T = extr["tvec"]
        
        focal_length_x = intr["params"][0]
        focal_length_y = intr["params"][1]
        width, height = intr["width"], intr["height"]

        image_name = image_files[i] if use_index_mapping and i < len(image_files) else extr["name"]
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path): continue
            
        image_pil = Image.open(image_path)
        image = np.array(image_pil)
        
        if image.shape[1] != width or image.shape[0] != height:
             scale_x, scale_y = image.shape[1] / width, image.shape[0] / height
             focal_length_x *= scale_x
             focal_length_y *= scale_y
             width, height = image.shape[1], image.shape[0]
        
        fovx = 2 * math.atan(width / (2 * focal_length_x))
        fovy = 2 * math.atan(height / (2 * focal_length_y))
        
        train_cameras.append(CameraInfo(
            uid=image_id, R=R, T=T, FovY=fovy, FovX=fovx,
            image=image / 255.0, image_path=image_path, image_name=image_name,
            width=width, height=height
        ))
        
    return xyz, rgb, train_cameras

def load_colmap_dataset(path: str, images_subdir: str = "images_8"):
    """
    Load COLMAP data and convert to package-standard Camera entities.
    """
    xyz, rgb, train_cam_infos = load_colmap_data_raw(path, images_subdir)
    
    jax_cameras = []
    jax_targets = []
    
    for info in train_cam_infos:
        fx = info.width / (2 * math.tan(info.FovX / 2))
        fy = info.height / (2 * math.tan(info.FovY / 2))
        cx, cy = info.width / 2.0, info.height / 2.0
        
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = info.R, info.T
        
        camera = Camera(
            W=info.width, H=info.height,
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            W2C=jnp.array(w2c), full_proj=jnp.eye(4)
        )
        jax_cameras.append(camera)
        jax_targets.append(jnp.array(info.image))
        
    return xyz, rgb, jax_cameras, jax_targets

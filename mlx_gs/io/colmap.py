import struct
import numpy as np
import mlx.core as mx
import os
import math
from dataclasses import dataclass
from mlx_gs.training.trainer import Camera
from torch_gs.io.colmap import load_colmap_data_raw

def load_colmap_dataset(path: str, images_subdir: str = "images_8"):
    """
    Load COLMAP data and convert to MLX Camera entities.
    Reuses the raw loader from torch_gs (which is numpy-based).
    """
    xyz, rgb, train_cam_infos = load_colmap_data_raw(path, images_subdir)
    
    cameras = []
    targets = []
    
    for info in train_cam_infos:
        fx = info.width / (2 * math.tan(info.FovX / 2))
        fy = info.height / (2 * math.tan(info.FovY / 2))
        cx, cy = info.width / 2.0, info.height / 2.0
        
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = info.R, info.T
        
        camera = Camera(
            W=info.width, H=info.height,
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            W2C=mx.array(w2c, dtype=mx.float32), 
            full_proj=mx.eye(4)
        )
        cameras.append(camera)
        targets.append(mx.array(info.image, dtype=mx.float32))
        
    return xyz, rgb, cameras, targets

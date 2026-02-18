import mlx.core as mx
from dataclasses import dataclass
import numpy as np

@dataclass
class Gaussians:
    """
    Main data structure representing a collection of 3D Gaussians in MLX.
    
    Attributes:
        means: (N, 3) array of central positions.
        scales: (N, 3) array of log-scales.
        quaternions: (N, 4) array of orientations (w, x, y, z).
        opacities: (N, 1) array of raw opacities.
        sh_coeffs: (N, K, 3) array of SH coefficients.
    """
    means: mx.array
    scales: mx.array
    quaternions: mx.array
    opacities: mx.array
    sh_coeffs: mx.array

def init_gaussians_from_pcd(points: np.ndarray, colors: np.ndarray) -> Gaussians:
    """
    Initializes MLX-based Gaussians from a point cloud.
    """
    means = mx.array(points, dtype=mx.float32)
    
    # Initialize scales (log space)
    scales = mx.full((points.shape[0], 3), -5.0, dtype=mx.float32)
    
    # Initialize quaternions (identity)
    quaternions = mx.zeros((points.shape[0], 4), dtype=mx.float32)
    quaternions[:, 0] = 1.0
    
    # Initialize opacities (raw)
    opacities = mx.zeros((points.shape[0], 1), dtype=mx.float32)
    
    # SH Coefficients (DC term only)
    # SH DC = (color - 0.5) / 0.28209479177387814
    sh_coeffs = mx.zeros((points.shape[0], 1, 3), dtype=mx.float32)
    sh_coeffs[:, 0, :] = (mx.array(colors, dtype=mx.float32) - 0.5) / 0.28209479177387814
    
    return Gaussians(
        means=means,
        scales=scales,
        quaternions=quaternions,
        opacities=opacities,
        sh_coeffs=sh_coeffs
    )

def get_covariance_3d(scales: mx.array, quaternions: mx.array) -> mx.array:
    """
    Computes 3D covariance matrix Sigma = R S S^T R^T.
    """
    # 1. Scaling matrix S
    s = mx.exp(scales)
    
    # Stable normalization
    norm = mx.sqrt(mx.sum(mx.square(quaternions), axis=-1, keepdims=True) + 1e-12)
    q = quaternions / norm
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Build rotation matrix rows (MLX style broadcasting)
    # R = mx.zeros((q.shape[0], 3, 3))
    # R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    # ...
    
    r00 = 1 - 2 * (y*y + z*z)
    r01 = 2 * (x*y - r*z)
    r02 = 2 * (x*z + r*y)
    
    r10 = 2 * (x*y + r*z)
    r11 = 1 - 2 * (x*x + z*z)
    r12 = 2 * (y*z - r*x)
    
    r20 = 2 * (x*z - r*y)
    r21 = 2 * (y*z + r*x)
    r22 = 1 - 2 * (x*x + y*y)
    
    row0 = mx.stack([r00, r01, r02], axis=-1)
    row1 = mx.stack([r10, r11, r12], axis=-1)
    row2 = mx.stack([r20, r21, r22], axis=-1)
    
    R = mx.stack([row0, row1, row2], axis=-2)
    
    # 3. M = R * S
    # S is diagonal, so we can just scale columns of R
    M = R * s[:, None, :]
    
    # 4. Sigma = M * M^T
    Sigma = mx.matmul(M, M.transpose(0, 2, 1))
    
    return Sigma

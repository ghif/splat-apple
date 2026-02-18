"""
Gaussian projection from 3D world space to 2D image coordinates.
This module handles camera transformations, perspective projection of Gaussians,
and computation of 2D covariances and screen-space radii.
"""
import torch
from torch_gs.core.gaussians import Gaussians, get_covariance_3d

def project_gaussians(gaussians: Gaussians, camera, device="mps"):
    """
    Project 3D Gaussians into the 2D image plane of a given camera.
    
    The projection process involves:
    1. Transforming 3D means to camera coordinates.
    2. Filtering out Gaussians that are behind the camera.
    3. Projecting 3D covariances to 2D using the perspective Jacobian (EWA Splatting).
    4. Calculating the screen-space means and influence radii.

    Args:
        gaussians (Gaussians): The collection of 3D Gaussians to project.
        camera: Camera object with properties: W2C, fx, fy, cx, cy, H, W.
        device (str): Device for computation ("mps", "cuda", or "cpu").
        
    Returns:
        tuple: (means2D, cov2D, radii, valid_mask, depths)
            - means2D: (N, 2) Projected means in pixel coordinates.
            - cov2D: (N, 2, 2) 2D covariance matrices for splatting.
            - radii: (N,) Max screen-space radius of influence for each splat.
            - valid_mask: (N,) Boolean mask for Gaussians in the frustum.
            - depths: (N,) Z-depth of each Gaussian in camera space.
    """
    means3D = gaussians.means
    scales = gaussians.scales
    quats = gaussians.quaternions
    
    # 1. Coordinate Transformation (World to Camera)
    # R: Rotation matrix from W2C (3x3)
    # T: Translation vector from W2C (3,)
    R = camera.W2C[:3, :3]
    T = camera.W2C[:3, 3]
    
    # Transform 3D means: means_cam = means3D @ R.T + T
    means_cam = torch.matmul(means3D, R.T) + T
    
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    
    # 2. Frustum Culling
    # Keep only Gaussians that are in front of the camera (z > threshold)
    valid_mask = z > 0.01
    
    # 3. Compute 3D Covariances
    cov3D = get_covariance_3d(scales, quats)
    
    # 4. Perspective Projection to 2D (EWA Splatting)
    # We use the Jacobian of the perspective projection evaluated at Gaussian's center.
    # J = [ [fx/z,   0,   -fx*x/z^2],
    #       [  0,   fy/z, -fy*y/z^2] ]
    inv_z = 1.0 / (z + 1e-6)
    inv_z2 = inv_z**2
    
    zeros = torch.zeros_like(z)
    row0 = torch.stack([camera.fx * inv_z, zeros, -camera.fx * x * inv_z2], dim=-1)
    row1 = torch.stack([zeros, camera.fy * inv_z, -camera.fy * y * inv_z2], dim=-1)
    J = torch.stack([row0, row1], dim=-2) # (N, 2, 3)
    
    # Project 3D covariance to 2D: cov2D = J @ R @ cov3D @ R.T @ J.T
    # We first pre-multiply M = J @ R
    W_rot = R # Rotation part of World-to-Camera
    M = torch.matmul(J, W_rot.unsqueeze(0).expand(means3D.shape[0], -1, -1))
    
    # Compute cov2D = M @ cov3D @ M.T
    cov2D = torch.matmul(M, torch.matmul(cov3D, M.transpose(1, 2)))

    # 5. Low-Pass Filter
    # Add a small bias for numerical stability and to ensure anti-aliasing
    # This prevents splats from becoming infinitely thin when viewed edge-on.
    eye2D = torch.eye(2, device=device).unsqueeze(0)
    cov2D = cov2D + 0.3 * eye2D
    
    # 6. Screen-Space Means
    # standard projection: u = fx * (x/z) + cx, v = fy * (y/z) + cy
    means2D = torch.stack([
        camera.fx * x * inv_z + camera.cx,
        camera.fy * y * inv_z + camera.cy
    ], dim=-1)
    
    # 7. Influence Radii for Sorting and Tiling
    # Based on the eigenvalues of the 2D covariance matrix.
    # Radius is approx 3.0 * sqrt(max_eigenvalue).
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    trace = cov2D[:, 0, 0] + cov2D[:, 1, 1]
    mid = trace / 2.0
    term = torch.sqrt(torch.clamp(mid**2 - det, min=0.0))
    lambda1 = mid + term # Largest eigenvalue
    radii = torch.ceil(3.0 * torch.sqrt(lambda1))
    
    return means2D, cov2D, radii, valid_mask, z

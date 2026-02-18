"""
Core data structures and initialization for 3D Gaussian Splatting.
This module defines the Gaussian representation and basic geometric computations
like 3D covariance matrix construction.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Gaussians:
    """
    Main data structure representing a collection of 3D Gaussians.
    Each Gaussian is defined by its position (means), scale, orientation (quaternions),
    opacity, and color/appearance (SH coefficients).
    
    Attributes:
        means: (N, 3) tensor of central positions (x, y, z).
        scales: (N, 3) tensor of log-scales for each axis.
        quaternions: (N, 4) tensor representing orientation in (w, x, y, z) format.
        opacities: (N, 1) tensor of raw opacity values (to be passed through sigmoid).
        sh_coeffs: (N, K, 3) tensor of Spherical Harmonics coefficients for color modeling.
    """
    means: torch.Tensor  # (N, 3)
    scales: torch.Tensor  # (N, 3)
    quaternions: torch.Tensor  # (N, 4)
    opacities: torch.Tensor  # (N, 1)
    sh_coeffs: torch.Tensor  # (N, K, 3)

def init_gaussians_from_pcd(points: torch.Tensor, colors: torch.Tensor, device="mps") -> Gaussians:
    """
    Initialize a set of Gaussians from a starting point cloud.
    
    This function converts raw point positions and RGB colors into the internal
    Gaussian representation, setting up initial scales, orientations, and opacities.
    
    Args:
        points (torch.Tensor): Tensor of shape (N, 3) containing point positions.
        colors (torch.Tensor): Tensor of shape (N, 3) containing RGB colors in [0, 1].
        device (str): Destination device ("mps", "cuda", or "cpu").
        
    Returns:
        Gaussians: Initialized Gaussians object.
    """
    num_points = points.shape[0]
    
    # 1. Position: Initialize means directly from the point cloud positions
    means = points.to(device)
    
    # 2. Scales: Initialized in log-space.
    # We use log(0.05) ~ -3.0 as a conservative small starting scale for all axes.
    # In more advanced versions, this could be tuned to the mean distance between points.
    scales = torch.full((num_points, 3), -3.0, device=device)
    
    # 3. Rotations: Initialize as identity quaternions [1, 0, 0, 0] (no rotation)
    quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_points, 1)
    
    # 4. Opacities: Initialize to 0.0 (inverse sigmoid of 0.5)
    # This allows Gaussians to start semi-transparent and evolve during training.
    opacities = torch.full((num_points, 1), 0.0, device=device)
    
    # 5. SH Coefficients (Appearance)
    # We use only the DC term (0th degree) initially, which represents basic color.
    # The transformation: SH_DC = (RGB - 0.5) / 0.28209...
    sh_dc = (colors.to(device) - 0.5) / 0.28209479177387814
    
    # Degree 3 SH has 16 coefficients (4*4). We allocate space for all 16.
    sh_coeffs = torch.zeros((num_points, 16, 3), device=device) 
    sh_coeffs[:, 0, :] = sh_dc
    
    return Gaussians(
        means=means,
        scales=scales,
        quaternions=quaternions,
        opacities=opacities,
        sh_coeffs=sh_coeffs
    )

def get_covariance_3d(scales: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    """
    Computes the 3D covariance matrix from scales and orientations.
    Using the formula: Σ = R S S^T R^T
    where R is the rotation matrix derived from quaternions and S is the scaling matrix.
    
    Args:
        scales (torch.Tensor): (N, 3) log-scales of the Gaussians.
        quaternions (torch.Tensor): (N, 4) orientation quaternions.
        
    Returns:
        torch.Tensor: (N, 3, 3) 3D covariance matrices.
    """
    # 1. Normalize quaternions to ensure they represent valid rotations
    q = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)
    
    # 2. Extract components
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # 3. Construct the rotation matrix R manually for speed and clarity
    # R = [ [1-2y^2-2z^2, 2xy-2rz,     2xz+2ry],
    #       [2xy+2rz,     1-2x^2-2z^2, 2yz-2rx],
    #       [2xz-2ry,     2yz+2rx,     1-2x^2-2y^2] ]
    
    # Column 1
    R00 = 1 - 2*y**2 - 2*z**2
    R10 = 2*x*y + 2*r*z
    R20 = 2*x*z - 2*r*y
    
    # Column 2
    R01 = 2*x*y - 2*r*z
    R11 = 1 - 2*x**2 - 2*z**2
    R21 = 2*y*z + 2*r*x
    
    # Column 3
    R02 = 2*x*z + 2*r*y
    R12 = 2*y*z - 2*r*x
    R22 = 1 - 2*x**2 - 2*y**2
    
    # Stack into (N, 3, 3)
    R = torch.stack([
        torch.stack([R00, R01, R02], dim=-1),
        torch.stack([R10, R11, R12], dim=-1),
        torch.stack([R20, R21, R22], dim=-1)
    ], dim=-2)
    
    # 4. Construct scaling matrix S from log-scales
    s = torch.exp(scales)
    
    # 5. Compute M = R @ S
    # Since S is a diagonal matrix, this is equivalent to scaling columns of R.
    M = R * s.unsqueeze(1)
    
    # 6. Compute Sigma = M @ M^T (which is R S S^T R^T)
    Sigma = torch.bmm(M, M.transpose(1, 2))
    
    return Sigma

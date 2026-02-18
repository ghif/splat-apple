"""
Main Renderer Orchestration for 3D Gaussian Splatting.
This module provides the high-level 'render' function that connects
projection, sorting, and rasterization into a single pipeline.
"""
import torch
import torch.nn.functional as F
from torch_gs.core.gaussians import Gaussians
from torch_gs.renderer.projection import project_gaussians
from torch_gs.renderer.rasterizer import get_tile_interactions, render_tiles, TILE_SIZE
from torch_gs.renderer.rasterizer_cpp import get_tile_interactions_cpp, render_tiles_cpp
from torch_gs.renderer.rasterizer_mps import render_tiles_mps

def render(gaussians: Gaussians, camera, background=None, device="mps", rasterizer_type="python"):
    """
    Renders a 3D Gaussian scene from a specific camera viewpoint.
    """
    if background is None:
        background = torch.zeros(3, device=device)
        
    # 1. Project Gaussians from 3D space to 2D image coordinates
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(gaussians, camera, device=device)
    
    # 2. Appearance: Compute colors from Spherical Harmonics (SH)
    colors = gaussians.sh_coeffs[:, 0, :] * 0.28209479177387814 + 0.5
    colors = torch.clamp(colors, 0.0, 1.0)
    
    # 3. Geometry & Visibility: Determine tile interactions
    if rasterizer_type == "cpp":
        # Phase 4: Use GPU-resident interactions + Metal rasterizer
        sorted_tile_ids, sorted_gaussian_ids = get_tile_interactions(
            means2D, radii, valid_mask, depths, camera.H, camera.W, TILE_SIZE, device=device
        )
        
        image = render_tiles_mps(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, background, device=device
        )
    elif rasterizer_type == "cpu_legacy":
        # Old CPU-based C++ rasterizer
        sorted_tile_ids, sorted_gaussian_ids = get_tile_interactions_cpp(
            means2D, radii, valid_mask, depths, camera.H, camera.W, TILE_SIZE, device=device
        )
        
        image = render_tiles_cpp(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, background, device=device
        )
    else:
        # Pure Python vectorized path
        sorted_tile_ids, sorted_gaussian_ids = get_tile_interactions(
            means2D, radii, valid_mask, depths, camera.H, camera.W, TILE_SIZE, device=device
        )
        
        image = render_tiles(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, background, device=device
        )
    
    return image

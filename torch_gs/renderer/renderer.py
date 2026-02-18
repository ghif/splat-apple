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

def render(gaussians: Gaussians, camera, background=None, device="mps", rasterizer_type="python"):
    """
    Renders a 3D Gaussian scene from a specific camera viewpoint.
    
    The pipeline consists of:
    1. 3D to 2D Projection: Projecting 3D Gaussians to the image plane.
    2. SH Color Computation: Evaluating spherical harmonics for viewing-dependent colors.
    3. Tile-Based Sorting: Mapping projected Gaussians to screen tiles and sorting by depth.
    4. Rasterization: Determining final pixel colors via alpha-blending of sorted splats.

    Args:
        gaussians (Gaussians): The collection of 3D Gaussians to render.
        camera: Camera object with W2C, intrinsics, and image dimensions (H, W).
        background (torch.Tensor, optional): (3,) RGB background color. Defaults to black.
        device (str, optional): Target device for rendering. Defaults to "mps".
        rasterizer_type (str, optional): "python" or "cpp". Defaults to "python".
        
    Returns:
        torch.Tensor: (H, W, 3) The rendered image.
    """
    if background is None:
        background = torch.zeros(3, device=device)
        
    # 1. Project Gaussians from 3D space to 2D image coordinates
    # This returns screen positions, 2D covariances, and influence radii.
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(gaussians, camera, device=device)
    
    # 2. Appearance: Compute colors from Spherical Harmonics (SH)
    # For now, we use only the DC term (0th degree SH) which is view-independent.
    # The color is derived from sh_coeffs[:, 0] and scaled back to [0, 1].
    colors = gaussians.sh_coeffs[:, 0, :] * 0.28209479177387814 + 0.5
    colors = torch.clamp(colors, 0.0, 1.0)
    
    # 3. Geometry & Visibility: Determine tile interactions
    # Efficiently find which tiles each Gaussian overlaps and sort them for alpha-blending.
    if rasterizer_type == "cpp":
        sorted_tile_ids, sorted_gaussian_ids = get_tile_interactions_cpp(
            means2D, radii, valid_mask, depths, camera.H, camera.W, TILE_SIZE, device=device
        )
        
        # 4. Rendering: Splatting and Blending
        image = render_tiles_cpp(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, background, device=device
        )
    else:
        sorted_tile_ids, sorted_gaussian_ids = get_tile_interactions(
            means2D, radii, valid_mask, depths, camera.H, camera.W, TILE_SIZE, device=device
        )
        
        # 4. Rendering: Splatting and Blending
        image = render_tiles(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, background, device=device
        )
    
    return image

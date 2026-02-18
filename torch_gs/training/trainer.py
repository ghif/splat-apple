"""
Training utilities for Gaussian Splatting.
This module defines the camera representation and a single differentiable training step.
"""
import torch
from torch_gs.renderer.renderer import render
from torch_gs.training.losses import l1_loss, d_ssim_loss
from dataclasses import dataclass

@dataclass
class Camera:
    """
    Representation of a pinhole camera for rendering.
    
    Attributes:
        W, H (int): Image resolution.
        fx, fy (float): Focal lengths in pixel units.
        cx, cy (float): Principal point coordinates.
        W2C (torch.Tensor): 4x4 World-to-Camera transformation matrix.
        full_proj (torch.Tensor, optional): 4x4 projection matrix.
    """
    W: int
    H: int
    fx: float
    fy: float
    cx: float
    cy: float
    W2C: torch.Tensor
    full_proj: torch.Tensor = None

def train_step(gaussians, optimizer, target_image, camera, lambda_ssim=0.2, device="mps", rasterizer_type="python"):
    """
    Performs a single optimization step for the Gaussians.
    
    Operations:
    1. Render the scene from the given camera viewpoint.
    2. Compute the weighted loss between rendered and target images.
    3. Backpropagate gradients to the Gaussian parameters.
    4. Update parameters using the optimizer.
    5. Calculate PSNR for monitoring training quality.

    Args:
        gaussians (Gaussians): The collection of Gaussians being optimized.
        optimizer (torch.optim.Optimizer): Optimizer managing Gaussian leaf tensors.
        target_image (torch.Tensor): (H, W, 3) Ground truth RGB image.
        camera (Camera): Camera used for rendering the current view.
        lambda_ssim (float): Weight for SSIM loss (balance between L1 and structural loss).
        device (str): Computation device.
        rasterizer_type (str): "python" or "cpp".
        
    Returns:
        tuple: (loss_value, psnr_value, rendered_image)
    """
    optimizer.zero_grad()
    
    # 1. Render the image from current camera parameters
    image = render(gaussians, camera, device=device, rasterizer_type=rasterizer_type)
    
    # 2. Compute combined Loss (L1 + SSIM)
    l1 = l1_loss(image, target_image)
    d_ssim = d_ssim_loss(image, target_image)
    loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * d_ssim
    
    # 3. Optimization
    loss.backward()
    optimizer.step()
    
    # 4. Diagnostics (no gradient needed)
    with torch.no_grad():
        mse = ((image - target_image) ** 2).mean()
        psnr = -10.0 * torch.log10(mse)
        
    return loss.item(), psnr.item(), image

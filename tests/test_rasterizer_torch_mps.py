import torch
import numpy as np
import pytest
import os
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch_gs.renderer import rasterizer, rasterizer_mps

def test_torch_mps_consistency():
    """
    Check if PyTorch MPS rasterizer produces same output as Python version.
    """
    device = "mps"
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    torch.manual_seed(42)
    H, W = 64, 64
    tile_size = 16
    num_points = 100
    
    means2D = torch.rand(num_points, 2, device=device) * W
    cov2D = torch.eye(2, device=device).unsqueeze(0).repeat(num_points, 1, 1) * 10.0
    opacities = torch.randn(num_points, 1, device=device)
    colors = torch.rand(num_points, 3, device=device)
    
    radii = torch.full((num_points,), 10.0, device=device)
    valid_mask = torch.ones(num_points, dtype=torch.bool, device=device)
    depths = torch.rand(num_points, device=device) * 10.0
    
    # Interactions
    sti, sgi = rasterizer.get_tile_interactions(
        means2D, radii, valid_mask, depths, H, W, tile_size, device=device
    )
    
    # Render Python
    img_py = rasterizer.render_tiles(
        means2D, cov2D, opacities, colors, 
        sti, sgi, H, W, tile_size, device=device
    )
    
    # Render MPS
    img_mps = rasterizer_mps.render_tiles_mps(
        means2D, cov2D, opacities, colors, 
        sti, sgi, H, W, tile_size, device=device
    )
    
    img_py_cpu = img_py.cpu()
    img_mps_cpu = img_mps.cpu()
    
    diff = torch.abs(img_py_cpu - img_mps_cpu)
    max_diff = diff.max().item()
    print(f"Max difference: {max_diff:.6f}")
    
    assert max_diff < 2e-2

def test_torch_mps_gradients():
    """
    Check if gradients are computed and non-zero.
    """
    device = "mps"
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    H, W = 32, 32
    tile_size = 16
    num_points = 10
    
    means2D = (torch.rand(num_points, 2, device=device) * W).requires_grad_(True)
    cov2D = (torch.eye(2, device=device).unsqueeze(0).repeat(num_points, 1, 1) * 10.0).requires_grad_(True)
    opacities = torch.randn(num_points, 1, device=device).requires_grad_(True)
    colors = torch.rand(num_points, 3, device=device).requires_grad_(True)
    
    radii = torch.full((num_points,), 10.0, device=device)
    valid_mask = torch.ones(num_points, dtype=torch.bool, device=device)
    depths = torch.rand(num_points, device=device) * 10.0
    
    sti, sgi = rasterizer.get_tile_interactions(
        means2D, radii, valid_mask, depths, H, W, tile_size, device=device
    )
    
    img = rasterizer_mps.render_tiles_mps(
        means2D, cov2D, opacities, colors, 
        sti, sgi, H, W, tile_size, device=device
    )
    
    loss = img.sum()
    loss.backward()
    
    assert means2D.grad is not None
    assert cov2D.grad is not None
    assert opacities.grad is not None
    assert colors.grad is not None
    
    assert means2D.grad.abs().max() > 0
    assert cov2D.grad.abs().max() > 0
    assert opacities.grad.abs().max() > 0
    assert colors.grad.abs().max() > 0
    
    print("Gradients successful")

if __name__ == "__main__":
    print("Testing PyTorch MPS rasterizer...")
    test_torch_mps_consistency()
    test_torch_mps_gradients()
    print("All tests passed!")

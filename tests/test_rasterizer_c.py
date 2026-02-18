import mlx.core as mx
import numpy as np
import pytest
import os
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlx_gs.renderer import rasterizer, rasterizer_c

def test_rasterizer_consistency():
    """
    Check if C++ rasterizer produces the same output as Python rasterizer.
    """
    mx.random.seed(42)
    H, W = 64, 64
    tile_size = 16
    num_points = 100
    
    means2D = mx.random.uniform(0, W, (num_points, 2))
    cov2D = mx.array([[[10.0, 2.0], [2.0, 10.0]]] * num_points)
    opacities = mx.random.uniform(-1, 1, (num_points, 1))
    colors = mx.random.uniform(0, 1, (num_points, 3))
    
    # Interactions
    radii = mx.array([10.0] * num_points)
    valid_mask = mx.array([True] * num_points)
    depths = mx.random.uniform(1, 10, (num_points,))
    
    sorted_tile_ids_py, sorted_gaussian_ids_py = rasterizer._get_tile_interactions_impl(
        means2D, radii, valid_mask, depths, H, W, tile_size
    )
    
    sorted_tile_ids_c, sorted_gaussian_ids_c = rasterizer_c.get_tile_interactions(
        means2D, radii, valid_mask, depths, H, W, tile_size
    )
    
    # Render
    img_py = rasterizer.render_tiles(
        means2D, cov2D, opacities, colors, 
        sorted_tile_ids_py, sorted_gaussian_ids_py, 
        H, W, tile_size, rasterizer_type="python"
    )
    
    img_c = rasterizer_c.render_tiles(
        means2D, cov2D, opacities, colors, 
        sorted_tile_ids_c, sorted_gaussian_ids_c, 
        H, W, tile_size
    )
    
    mx.eval(img_py, img_c)
    
    diff = mx.abs(img_py - img_c)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    # Small differences expected due to float precision and different accumulation order/parallelism
    assert max_diff < 2e-3

def test_rasterizer_gradients():
    """
    Check if gradients are computed and non-zero.
    """
    H, W = 32, 32
    tile_size = 16
    num_points = 10
    
    params = {
        "means2D": mx.random.uniform(0, W, (num_points, 2)),
        "cov2D": mx.array([[[10.0, 0.0], [0.0, 10.0]]] * num_points),
        "opacities": mx.random.uniform(-1, 1, (num_points, 1)),
        "colors": mx.random.uniform(0, 1, (num_points, 3))
    }
    
    radii = mx.array([10.0] * num_points)
    valid_mask = mx.array([True] * num_points)
    depths = mx.random.uniform(1, 10, (num_points,))
    
    sti, sgi = rasterizer_c.get_tile_interactions(
        params["means2D"], radii, valid_mask, depths, H, W, tile_size
    )
    
    def loss_fn(p):
        img = rasterizer_c.render_tiles(
            p["means2D"], p["cov2D"], p["opacities"], p["colors"],
            sti, sgi, H, W, tile_size
        )
        return mx.mean(mx.square(img))

    grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = grad_fn(params)
    
    mx.eval(loss, grads)
    
    assert loss > 0
    for k, g in grads.items():
        assert mx.max(mx.abs(g)) > 0
        print(f"Grad {k} max: {mx.max(mx.abs(g)).item():.6f}")

if __name__ == "__main__":
    print("Testing rasterizer consistency...")
    test_rasterizer_consistency()
    print("Testing rasterizer gradients...")
    test_rasterizer_gradients()
    print("All tests passed!")

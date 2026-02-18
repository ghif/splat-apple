import mlx.core as mx
import numpy as np
import pytest
import os
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlx_gs.renderer import rasterizer_c, rasterizer_metal

def test_metal_consistency():
    if not hasattr(rasterizer_metal, 'rasterizer_metal') or rasterizer_metal.rasterizer_metal is None:
        pytest.skip("Metal rasterizer not available")

    mx.random.seed(42)
    H, W = 64, 64
    tile_size = 16
    num_points = 100
    
    means2D = mx.random.uniform(0, W, (num_points, 2))
    cov2D = mx.array([[[10.0, 2.0], [2.0, 10.0]]] * num_points)
    opacities = mx.random.uniform(-1, 1, (num_points, 1))
    colors = mx.random.uniform(0, 1, (num_points, 3))
    
    radii = mx.array([10.0] * num_points)
    valid_mask = mx.array([True] * num_points)
    depths = mx.random.uniform(1, 10, (num_points,))
    
    sorted_tile_ids, sorted_gaussian_ids = rasterizer_c.get_tile_interactions(
        means2D, radii, valid_mask, depths, H, W, tile_size
    )
    
    img_c = rasterizer_c.render_tiles(
        means2D, cov2D, opacities, colors, 
        sorted_tile_ids, sorted_gaussian_ids, 
        H, W, tile_size
    )
    
    img_m = rasterizer_metal.render_tiles(
        means2D, cov2D, opacities, colors, 
        sorted_tile_ids, sorted_gaussian_ids, 
        H, W, tile_size
    )
    
    mx.eval(img_c, img_m)
    
    diff = mx.abs(img_c - img_m)
    max_diff = mx.max(diff).item()
    print(f"Max difference: {max_diff:.6f}")
    # Metal uses fast math, C++ uses std::exp. Some diff expected.
    # Also order of accumulation might differ (parallelism).
    assert max_diff < 2e-2

def test_metal_gradients():
    if not hasattr(rasterizer_metal, 'rasterizer_metal') or rasterizer_metal.rasterizer_metal is None:
        pytest.skip("Metal rasterizer not available")

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
        img = rasterizer_metal.render_tiles(
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
    print("Testing Metal rasterizer consistency...")
    test_metal_consistency()
    print("Testing Metal rasterizer gradients...")
    test_metal_gradients()
    print("All tests passed!")

import jax.numpy as jnp
from jax_gs.core.gaussians import Gaussians
from jax_gs.core.camera import Camera
from jax_gs.renderer.projection import project_gaussians
from jax_gs.renderer.rasterizer import get_tile_interactions, render_tiles, TILE_SIZE

try:
    import mlx.core as mx
    from mlx_gs.renderer.renderer import render_mlx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch
    from jax_gs.renderer import rasterizer_torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def render(gaussians: Gaussians, camera: Camera, background=jnp.array([0.0, 0.0, 0.0]), use_mlx: bool = False, use_torch: bool = False):
    """
    Main entry point for rendering.

    Args:
        gaussians: Gaussians dataclass
        camera: Camera dataclass
        background: Background color
    Returns:
        image: Rendered image
    """
    # 1. Project Gaussians to 2D
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(gaussians, camera)
    
    colors = gaussians.sh_coeffs[:, 0, :] * 0.28209479177387814 + 0.5
    colors = jnp.clip(colors, 0.0, 1.0)
    
    # 3. Sort and Rasterize
    if use_mlx and HAS_MLX:
        # Use centralized MLX-native rendering pipeline
        import numpy as np
        
        # Camera dict for MLX
        mlx_camera = {
            "W": int(camera.W),
            "H": int(camera.H),
            "fx": float(camera.fx),
            "fy": float(camera.fy),
            "cx": float(camera.cx),
            "cy": float(camera.cy),
            "W2C": mx.array(np.array(camera.W2C))
        }
        
        # Params dict for MLX
        params = {
            "means": mx.array(np.array(gaussians.means)),
            "scales": mx.array(np.array(gaussians.scales)),
            "quaternions": mx.array(np.array(gaussians.quaternions)),
            "opacities": mx.array(np.array(gaussians.opacities)),
            "sh_coeffs": mx.array(np.array(gaussians.sh_coeffs))
        }
        
        m_background = mx.array(np.array(background))
        
        # Render using the decoupled rendering entry point
        image_mlx = render_mlx(params, mlx_camera, m_background)
        
        # Convert back to JAX
        image = jnp.array(np.array(image_mlx))
    elif use_torch and HAS_TORCH:
        # Convert JAX to Torch
        import numpy as np
        t_means2D = torch.from_numpy(np.array(means2D)).to("mps")
        t_cov2D = torch.from_numpy(np.array(cov2D)).to("mps")
        t_radii = torch.from_numpy(np.array(radii)).to("mps")
        t_valid_mask = torch.from_numpy(np.array(valid_mask)).to("mps")
        t_depths = torch.from_numpy(np.array(depths)).to("mps")
        t_opacities = torch.from_numpy(np.array(gaussians.opacities)).to("mps")
        t_colors = torch.from_numpy(np.array(colors)).to("mps")
        t_background = torch.from_numpy(np.array(background)).to("mps")
        
        sorted_tile_ids, sorted_gaussian_ids, _ = rasterizer_torch.get_tile_interactions(
            t_means2D, t_radii, t_valid_mask, t_depths, camera.H, camera.W, TILE_SIZE, device="mps"
        )
        
        image_torch = rasterizer_torch.render_tiles(
            t_means2D, t_cov2D, t_opacities, t_colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, t_background, device="mps"
        )
        
        # Convert back to JAX
        image = jnp.array(image_torch.cpu().numpy())
    else:
        # 3. Sort interactions
        sorted_tile_ids, sorted_gaussian_ids, n_interactions = get_tile_interactions(
            means2D, radii, valid_mask, depths, camera.H, camera.W, TILE_SIZE
        )
        
        
        # 4. Rasterize tiles
        image = render_tiles(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, background
        )
    
    return image

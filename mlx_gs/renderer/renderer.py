import mlx.core as mx
from mlx_gs.renderer.projection import project_gaussians
from mlx_gs.renderer import rasterizer
try:
    from mlx_gs.renderer import rasterizer_metal
except ImportError:
    rasterizer_metal = None


def _render_stage1(params, camera_dict):
    # Standard projection
    class Obj:
        def __init__(self, d): self.__dict__.update(d)
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(Obj(params), Obj(camera_dict))
    
    # SH DC to RGB
    colors = mx.clip(params["sh_coeffs"][:, 0, :] * 0.28209479177387814 + 0.5, 0.0, 1.0)
    
    return means2D, cov2D, radii, valid_mask, depths, colors

def render(gaussians, camera, background=None, rasterizer_type="python"):
    if isinstance(gaussians, dict):
        params = gaussians
    else:
        params = {
            "means": gaussians.means, "scales": gaussians.scales,
            "quaternions": gaussians.quaternions, "opacities": gaussians.opacities,
            "sh_coeffs": gaussians.sh_coeffs
        }
    
    cam_dict = {
        "H": camera.H, "W": camera.W, "fx": camera.fx, "fy": camera.fy,
        "cx": camera.cx, "cy": camera.cy, "W2C": camera.W2C
    }
    
    # Stage 1: Projection (Compiled independent of interactions)
    means2D, cov2D, radii, valid_mask, depths, colors = _render_stage1(params, cam_dict)
    
    if rasterizer_type == "cpp":
        if rasterizer_metal is None or rasterizer_metal.rasterizer_metal is None:
            raise ImportError("Metal rasterizer not available. Please build the extension.")
            
        # Phase 4: Use GPU-resident interactions
        sorted_tile_ids, sorted_gaussian_ids = rasterizer_metal.get_tile_interactions(
            means2D, radii, valid_mask, depths, camera.H, camera.W, rasterizer.TILE_SIZE
        )
        
        return rasterizer_metal.render_tiles(
            means2D, cov2D, params["opacities"], colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, rasterizer.TILE_SIZE, background
        )
    else:
        # Python Interaction Stage
        sorted_tile_ids, sorted_gaussian_ids = rasterizer._get_tile_interactions_impl(
            means2D, radii, valid_mask, depths, camera.H, camera.W, rasterizer.TILE_SIZE
        )
        return rasterizer.render_tiles(
            means2D, cov2D, params["opacities"], colors, 
            sorted_tile_ids, sorted_gaussian_ids, 
            camera.H, camera.W, rasterizer.TILE_SIZE, background
        )

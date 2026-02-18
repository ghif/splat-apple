import torch
import torch_gs._C as cpp_gs

def render_tiles_cpp(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, 
                     H, W, tile_size, background=None, device="mps"):
    if background is None:
        background = torch.zeros(3, device=device)
        
    return cpp_gs.render_tiles(
        means2D, cov2D, opacities, colors, 
        sorted_tile_ids, sorted_gaussian_ids, 
        H, W, tile_size, background
    )

def get_tile_interactions_cpp(means2D, radii, valid_mask, depths, H, W, tile_size, device="mps"):
    return cpp_gs.get_tile_interactions(
        means2D, radii, valid_mask, depths, H, W, tile_size
    )

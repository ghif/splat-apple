import torch
import numpy as np
from typing import Tuple

# Standard Constants
TILE_SIZE = 16
BLOCK_SIZE = 256  

def get_tile_interactions(means2D, radii, valid_mask, depths, H, W, tile_size: int = TILE_SIZE, device="mps"):
    """
    Generate tile interactions using PyTorch.
    """
    num_points = means2D.shape[0]
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    
    min_x = torch.clamp((means2D[:, 0] - radii), 0, W - 1)
    max_x = torch.clamp((means2D[:, 0] + radii), 0, W - 1)
    min_y = torch.clamp((means2D[:, 1] - radii), 0, H - 1)
    max_y = torch.clamp((means2D[:, 1] + radii), 0, H - 1)
    
    tile_min_x = (min_x // tile_size).to(torch.int32)
    tile_max_x = (max_x // tile_size).to(torch.int32)
    tile_min_y = (min_y // tile_size).to(torch.int32)
    tile_max_y = (max_y // tile_size).to(torch.int32)
    
    # Filter points completely outside image
    on_screen = (means2D[:, 0] + radii > 0) & (means2D[:, 0] - radii < W) & \
                (means2D[:, 1] + radii > 0) & (means2D[:, 1] - radii < H)
    
    valid_mask = valid_mask & on_screen & (tile_max_x >= tile_min_x) & (tile_max_y >= tile_min_y)
    
    def get_gaussian_tiles(idx, t_min_x, t_max_x, t_min_y, t_max_y, is_valid):
        # We need a fixed grid size for vmap (like 8x8)
        xs = torch.arange(8, device=device) 
        ys = torch.arange(8, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        
        abs_x = t_min_x + grid_x
        abs_y = t_min_y + grid_y
        
        in_range = (abs_x <= t_max_x) & (abs_y <= t_max_y) & is_valid
        
        tile_ids = abs_y * num_tiles_x + abs_x
        tile_ids = torch.where(in_range, tile_ids, torch.tensor(-1, device=device, dtype=torch.int32))
        
        return tile_ids.flatten()

    all_tile_ids = torch.vmap(get_gaussian_tiles)(
        torch.arange(num_points, device=device), 
        tile_min_x, tile_max_x, 
        tile_min_y, tile_max_y, 
        valid_mask
    )
    
    all_gaussian_ids = torch.arange(num_points, device=device)[:, None].expand(-1, 64)
    
    flat_tile_ids = all_tile_ids.reshape(-1)
    flat_gaussian_ids = all_gaussian_ids.reshape(-1)
    flat_depths = depths[:, None].expand(-1, 64).reshape(-1)
    
    valid_interactions = flat_tile_ids != -1
    
    # Pack-Sort for Torch
    DEPTH_BITS = 13
    num_tiles_total = num_tiles_x * num_tiles_y
    
    sort_tile_ids = torch.where(valid_interactions, flat_tile_ids, torch.tensor(num_tiles_total, device=device, dtype=torch.int32))
    
    # Quantize depths
    depth_min = flat_depths.min()
    depth_max = flat_depths.max()
    depth_quant = ((flat_depths - depth_min) / (depth_max - depth_min + 1e-6) * (2**DEPTH_BITS - 1)).to(torch.int32)
    
    key = (sort_tile_ids.to(torch.int64) << DEPTH_BITS) | depth_quant.to(torch.int64)
    sort_indices = torch.argsort(key)
    
    sorted_tile_ids = sort_tile_ids[sort_indices]
    sorted_gaussian_ids = flat_gaussian_ids[sort_indices]
    
    total_interactions = sorted_tile_ids.shape[0]
    padded_size = max(total_interactions, BLOCK_SIZE)
    
    pad_tile_ids = torch.full((padded_size,), num_tiles_total, dtype=torch.int32, device=device)
    pad_gaussian_ids = torch.zeros((padded_size,), dtype=torch.int32, device=device)
    
    pad_tile_ids[:total_interactions] = sorted_tile_ids
    pad_gaussian_ids[:total_interactions] = sorted_gaussian_ids
    
    return pad_tile_ids, pad_gaussian_ids, valid_interactions.sum()

def render_tiles(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, 
                 H, W, tile_size: int = TILE_SIZE, background=None, device="mps"):
    """
    Render tiles using PyTorch.
    """
    if background is None:
        background = torch.zeros(3, device=device)
        
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    num_tiles = num_tiles_x * num_tiles_y
    
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    det = torch.clamp(det, min=1e-6)
    
    # Inverse covariance
    # [[a, b], [c, d]] -> 1/det * [[d, -b], [-c, a]]
    inv_cov2D = torch.stack([
        torch.stack([cov2D[:, 1, 1] / det, -cov2D[:, 0, 1] / det], dim=-1),
        torch.stack([-cov2D[:, 1, 0] / det, cov2D[:, 0, 0] / det], dim=-1)
    ], dim=-2)
    
    sig_opacities = torch.sigmoid(opacities)

    # Calculate tile boundaries
    sorted_tile_ids_cpu = sorted_tile_ids.cpu().numpy()
    tile_indices = np.arange(num_tiles + 1)
    tile_boundaries = np.searchsorted(sorted_tile_ids_cpu, tile_indices)
    tile_boundaries = torch.from_numpy(tile_boundaries).to(device)

    # Calculate tile-to-pixel offsets
    tile_indices_all = torch.arange(num_tiles, device=device)
    ty_all = tile_indices_all // num_tiles_x
    tx_all = tile_indices_all % num_tiles_x
    pix_min_x_all = tx_all * tile_size
    pix_min_y_all = ty_all * tile_size
    
    # Pre-calculate pixel meshgrid (one tile)
    py = torch.arange(tile_size, device=device)
    px = torch.arange(tile_size, device=device)
    grid_y, grid_x = torch.meshgrid(py, px, indexing='ij')
    local_pixel_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2) # (256, 2)

    def render_tile_batch(batch_indices):
        B = batch_indices.shape[0]
        
        # 1. Gather tile-specific data
        s_indices = tile_boundaries[batch_indices]
        e_indices = tile_boundaries[batch_indices + 1]
        counts = e_indices - s_indices
        
        # Pixel coordinates for each tile in batch: (B, 256, 2)
        batch_pix_min_x = pix_min_x_all[batch_indices]
        batch_pix_min_y = pix_min_y_all[batch_indices]
        # pixel_coords: (B, 256, 2)
        pixel_coords = local_pixel_coords.unsqueeze(0) + torch.stack([batch_pix_min_x, batch_pix_min_y], dim=-1).unsqueeze(1)
        pixel_valid = (pixel_coords[:, :, 0] < W) & (pixel_coords[:, :, 1] < H)
        
        # 2. Gather Gaussians for each tile in batch: (B, BLOCK_SIZE)
        window = torch.arange(BLOCK_SIZE, device=device)
        gather_indices = torch.clamp(s_indices.unsqueeze(-1) + window.unsqueeze(0), 0, sorted_gaussian_ids.shape[0] - 1)
        indices = sorted_gaussian_ids[gather_indices]
        local_mask = window.unsqueeze(0) < counts.unsqueeze(-1)
        
        # Fetch Gaussian parameters: (B, BLOCK_SIZE, ...)
        t_means = means2D[indices]
        t_inv_cov = inv_cov2D[indices]
        t_ops = sig_opacities[indices]
        t_cols = colors[indices]
        
        # 3. Vectorized Alpha-Blending over (Batch, Pixels)
        # Process Gaussians one by one to save memory O(B * Pixels)
        accum_color = torch.zeros((B, pixel_coords.shape[1], 3), device=device)
        T = torch.ones((B, pixel_coords.shape[1]), device=device)
        
        for i in range(BLOCK_SIZE):
            mu = t_means[:, i] # (B, 2)
            icov = t_inv_cov[:, i] # (B, 2, 2)
            op = t_ops[:, i, 0] # (B,)
            col = t_cols[:, i] # (B, 3)
            l_mask = local_mask[:, i] # (B,)
            
            # d: (B, 256, 2)
            d = pixel_coords - mu.unsqueeze(1)
            
            # power = -0.5 * d^T * Sigma^-1 * d
            # d_icov: (B, 256, 2) @ (B, 2, 2) -> (B, 256, 2)
            d_icov = torch.matmul(d, icov)
            power = -0.5 * torch.sum(d_icov * d, dim=-1) # (B, 256)
            
            alphas = torch.exp(power) * op.unsqueeze(1)
            alphas = torch.clamp(alphas, max=0.99)
            
            # Combined mask: Gaussian validity + T threshold + power range
            m = (power > -10.0) & l_mask.unsqueeze(1) & (T > 1e-4)
            alphas = torch.where(m, alphas, torch.zeros_like(alphas))
            
            weight = alphas * T
            accum_color = accum_color + weight.unsqueeze(-1) * col.unsqueeze(1)
            T = T * (1.0 - alphas)
            
        # Final Background Blending: (B, 256, 3)
        final_color = accum_color + T.unsqueeze(-1) * background.unsqueeze(0).unsqueeze(0)
        
        # Apply pixel valid mask
        final_color = torch.where(pixel_valid.unsqueeze(-1), final_color, torch.zeros_like(final_color))
        
        return final_color.reshape(B, tile_size, tile_size, 3)

    # Process all tiles in batches
    chunk_size = 128
    all_tiles_list = []
    for i in range(0, num_tiles, chunk_size):
        end = min(i + chunk_size, num_tiles)
        batch = torch.arange(i, end, device=device)
        all_tiles_list.append(render_tile_batch(batch))
        
    all_tiles = torch.cat(all_tiles_list, dim=0)
    
    output_grid = all_tiles.reshape(num_tiles_y, num_tiles_x, tile_size, tile_size, 3)
    output_image = output_grid.permute(0, 2, 1, 3, 4).reshape(num_tiles_y * tile_size, num_tiles_x * tile_size, 3)
    
    return output_image[:H, :W, :]

"""
Vectorized Tile-Based Rasterizer for Gaussian Splatting.
This module implements the core rendering logic, including:
1. Tile interaction generation: Mapping Gaussians to screen tiles.
2. Vectorized alpha-blending: Computing pixel colors using high-performance tensor operations.
3. Memory-efficient batching: Processing tiles in chunks to balance speed and VRAM usage.
"""
import torch
import numpy as np
from typing import Tuple

# Standard Constants
TILE_SIZE = 16 # Pixels per tile side (16x16 = 256 pixels)
BLOCK_SIZE = 256  

def get_tile_interactions(means2D, radii, valid_mask, depths, H, W, tile_size: int = TILE_SIZE, device="mps"):
    """
    Determine which Gaussians overlap which screen tiles and sort them by depth.
    
    Args:
        means2D (torch.Tensor): (N, 2) Screen-space means.
        radii (torch.Tensor): (N,) Influence radii.
        valid_mask (torch.Tensor): (N,) Initial validity mask from projection.
        depths (torch.Tensor): (N,) Z-depth for sorting.
        H, W (int): Image dimensions.
        tile_size (int): Size of each tile.
        device (str): Computation device.
        
    Returns:
        tuple: (sorted_tile_ids, sorted_gaussian_ids)
            - sorted_tile_ids: Tile ID for each valid interaction, sorted.
            - sorted_gaussian_ids: Gaussian ID for each interaction.
    """
    num_points = means2D.shape[0]
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    
    # Calculate bounding boxes in tile coordinates
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
    
    # We need a fixed grid size for vmap (like 8x8) or just broadcasting
    # torch.vmap is available in newer torch versions, but we can standard broadcasting for compatibility
    
    # Pre-calculate relative tile offsets for broadcasting (8x8 grid assumption for max coverage)
    OFFSET_SIZE = 8
    ys = torch.arange(OFFSET_SIZE, device=device)
    xs = torch.arange(OFFSET_SIZE, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    off_x = grid_x.flatten()
    off_y = grid_y.flatten()

    # Broadcase to find all tiles each Gaussian touches
    # tile_min_x: (N,) -> (N, 1)
    # off_x: (64,) -> (1, 64)
    abs_x = tile_min_x.unsqueeze(1) + off_x.unsqueeze(0)
    abs_y = tile_min_y.unsqueeze(1) + off_y.unsqueeze(0)
    
    in_range = (abs_x <= tile_max_x.unsqueeze(1)) & (abs_y <= tile_max_y.unsqueeze(1)) & valid_mask.unsqueeze(1)
    
    all_tile_ids = abs_y * num_tiles_x + abs_x
    all_tile_ids = torch.where(in_range, all_tile_ids, torch.tensor(-1, device=device, dtype=torch.int32))
    
    all_gaussian_ids = torch.arange(num_points, device=device).unsqueeze(1).expand(-1, 64)
    
    flat_tile_ids = all_tile_ids.reshape(-1)
    flat_gaussian_ids = all_gaussian_ids.reshape(-1)
    flat_depths = depths.unsqueeze(1).expand(-1, 64).reshape(-1)
    
    valid_interactions = flat_tile_ids != -1
    
    # Pack-Sort: Create a key combining TileID and Depth for efficient sorting
    # TileID in high bits, Depth in low bits -> sorting by key = primary sort by tile, secondary by depth
    DEPTH_BITS = 13
    num_tiles_total = num_tiles_x * num_tiles_y
    
    sort_tile_ids = torch.where(valid_interactions, flat_tile_ids, torch.tensor(num_tiles_total, device=device, dtype=torch.int32))
    
    # Quantize depths to fit in remaining bits
    depth_min = flat_depths.min()
    depth_max = flat_depths.max()
    depth_quant = ((flat_depths - depth_min) / (depth_max - depth_min + 1e-6) * (2**DEPTH_BITS - 1)).to(torch.int32)
    
    # MPS doesn't support 64-bit integers well in all ops, but let's try
    # If 64-bit sort fails, we might need to sort by tile then depth separately (stable sort)
    # key = (sort_tile_ids.to(torch.int64) << DEPTH_BITS) | depth_quant.to(torch.int64)
    # sort_indices = torch.argsort(key)
    
    # Alternative stable sort:
    # 1. Sort by depth
    # 2. Sort by tile (stable)
    # current_indices = torch.argsort(depth_quant)
    # sorted_tile_ids_temp = sort_tile_ids[current_indices]
    # sort_indices_stable = torch.argsort(sorted_tile_ids_temp, stable=True)
    # final_indices = current_indices[sort_indices_stable]
    
    # But let's stick to the packed key if possible, it's faster.
    # Fallback for MPS if int64 is an issue:
    # Key construction (64-bit integer)
    key = (sort_tile_ids.to(torch.int64) << DEPTH_BITS) | depth_quant.to(torch.int64)
    sort_indices = torch.argsort(key)
    
    sorted_tile_ids = sort_tile_ids[sort_indices]
    sorted_gaussian_ids = flat_gaussian_ids[sort_indices]
    
    # Filter out invalid interactions (sentinels pushed to end by sort)
    valid_count = valid_interactions.sum()
    sorted_tile_ids = sorted_tile_ids[:valid_count]
    sorted_gaussian_ids = sorted_gaussian_ids[:valid_count]
    
    return sorted_tile_ids, sorted_gaussian_ids

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

def render_tile_batch(
    batch_indices, tile_boundaries, pix_min_x_all, pix_min_y_all, 
    local_pixel_coords, sorted_gaussian_ids, means2D, inv_cov2D, 
    sig_opacities, colors, background, H, W, tile_size, device
):
    """
    Renders a batch of tiles using a fully vectorized implementation.
    This avoids Python loops and maximizes GPU utilization on MPS.
    
    Optimization: Uses torch.cumprod for transmittance calculation, allowing 
    simultaneous blending of all Gaussians in a tile.
    
    Args:
        batch_indices (torch.Tensor): Indices of tiles in this batch.
        tile_boundaries (torch.Tensor): Start/end indices of each tile in the sorted arrays.
        pix_min_x_all, pix_min_y_all (torch.Tensor): Top-left pixel coords for all tiles.
        local_pixel_coords (torch.Tensor): (256, 2) Relative pixel coords within a tile.
        sorted_gaussian_ids (torch.Tensor): Sorted IDs from get_tile_interactions.
        means2D, inv_cov2D, ... : Projected Gaussian parameters.
        H, W (int): Target image resolution.
        
    Returns:
        torch.Tensor: (B, tile_size, tile_size, 3) Rendered colors for the batch.
    """
    B = batch_indices.shape[0]
    
    # 1. Gather tile-specific data (start/end in sorted list)
    s_indices = tile_boundaries[batch_indices]
    e_indices = tile_boundaries[batch_indices + 1]
    counts = e_indices - s_indices
    
    # Pixel coordinates: (B, 256, 2)
    # Absolute pixel coordinates for all pixels in the batch
    batch_pix_min_x = pix_min_x_all[batch_indices]
    batch_pix_min_y = pix_min_y_all[batch_indices]
    pixel_coords = local_pixel_coords.unsqueeze(0) + torch.stack([batch_pix_min_x, batch_pix_min_y], dim=-1).unsqueeze(1)
    pixel_valid = (pixel_coords[:, :, 0] < W) & (pixel_coords[:, :, 1] < H)
    
    # 2. Gather Gaussians for the tiles in the batch
    # 256 is usually enough for Fern images_8. 
    # Vectorizing more than this might hit memory, but 256 is safe.
    # We use a fixed LIMIT to allow stable broadcasting and compilation.
    LIMIT = 256
    window = torch.arange(LIMIT, device=device)
    
    # Gather indices (clamped to valid range)
    gather_indices = torch.clamp(s_indices.unsqueeze(-1) + window.unsqueeze(0), 0, sorted_gaussian_ids.shape[0] - 1)
    indices = sorted_gaussian_ids[gather_indices] # (B, LIMIT)
    local_mask = window.unsqueeze(0) < counts.unsqueeze(-1) # (B, LIMIT) mask for actual Gaussians in tile
    
    t_means = means2D[indices] # (B, LIMIT, 2)
    t_inv_cov = inv_cov2D[indices] # (B, LIMIT, 2, 2)
    t_ops = sig_opacities[indices] # (B, LIMIT, 1)
    t_cols = colors[indices] # (B, LIMIT, 3)
    
    # 3. FULLY Vectorized Alpha-Blending
    # pixel_coords: (B, 256, 1, 2)
    # mu: (B, 1, LIMIT, 2)
    # dx: (B, 256, LIMIT, 2)
    # Compute relative offsets (dx) for every pixel vs every Gaussian in the batch
    dx = pixel_coords.unsqueeze(2) - t_means.unsqueeze(1)
    
    # d_icov calculation
    # icov: (B, 1, LIMIT, 2, 2)
    # dx: (B, 256, LIMIT, 1, 2)
    # Extract inverse covariance components for manual expansion (faster than bmm)
    a = t_inv_cov[:, :, 0, 0].unsqueeze(1)
    b = t_inv_cov[:, :, 0, 1].unsqueeze(1)
    c = t_inv_cov[:, :, 1, 0].unsqueeze(1)
    d = t_inv_cov[:, :, 1, 1].unsqueeze(1)
    
    dx_val = dx[:, :, :, 0]
    dy_val = dx[:, :, :, 1]
    
    # power = -0.5 * (dx^T * inv_cov * dx)
    # results in (B, 256, LIMIT)
    # Compute mahalanobis distance: power = -0.5 * (dx^T * Σ^-1 * dx)
    power = -0.5 * (dx_val**2 * a + dx_val * dy_val * (b + c) + dy_val**2 * d)
    
    # alphas: (B, 256, LIMIT)
    # Calculate per-splat alpha contribution
    alphas = torch.exp(torch.clamp(power, min=-10.0)) * t_ops.reshape(B, 1, LIMIT)
    
    # Apply local_mask: (B, 1, LIMIT)
    # Mask out padding splats or those with negligible contribution
    alphas = torch.where(local_mask.unsqueeze(1), alphas, torch.zeros_like(alphas))
    alphas = torch.clamp(alphas, max=0.99)
    
    # Calculate Transmittance T_i = prod_{j<i} (1 - alpha_j)
    # T: (B, 256, LIMIT)
    # TRANSMITTANCE: T_i is the probability light isn't absorbed by previous splats.
    # Formula: T_i = prod_{j<i} (1 - alpha_j)
    one_minus_alpha = 1.0 - alphas
    # Use log-space cumsum for better stability if needed, 
    # but cumprod is usually faster on GPU.
    # T_i is product UP TO i-1. 
    # We can pad with 1s at the beginning.
    prefix_prod = torch.cumprod(one_minus_alpha, dim=2)
    # Shift to get T_i
    # Shift prod to get T_i (starts at 1.0 for first splat)
    T = torch.cat([torch.ones((B, 256, 1), device=device), prefix_prod[:, :, :-1]], dim=2)
    
    # weights: (B, 256, LIMIT)
    # WEIGHT = Transmittance * Alpha (SMC contribution)
    weights = alphas * T
    
    # accum_color: (B, 256, 3)
    # t_cols: (B, 1, LIMIT, 3)
    # ACCUMULATE: Sum contributions across LIMIT dimension
    accum_color = torch.sum(weights.unsqueeze(-1) * t_cols.unsqueeze(1), dim=2)
    
    # final_T: (B, 256, 1)
    # Final Transmittance (to blend with background)
    final_T = prefix_prod[:, :, -1:]
    
    # Final Background Blending
    # BLEND: result = sum(weights * color) + FinalTransmittance * Background
    final_color = accum_color + final_T * background.reshape(1, 1, 3)
    # Apply image bound mask
    final_color = torch.where(pixel_valid.unsqueeze(-1), final_color, torch.zeros_like(final_color))
    
    return final_color.reshape(B, tile_size, tile_size, 3)

# NO compile needed for fully vectorized code as it runs at 3.3 it/s in Eager mode on MPS.
# torch.compile actually slows it down on current MPS backends due to unrolling overhead.

def render_tiles(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, 
                 H, W, tile_size: int = TILE_SIZE, background=None, device="mps"):
    """
    Orchestrates the tile-based rendering process.
    Divides the screen into tiles, batches them, and calls the vectorized renderer.
    
    Args:
        means2D (torch.Tensor): (N, 2) Projected means.
        cov2D (torch.Tensor): (N, 2, 2) 2D Covariances.
        opacities (torch.Tensor): (N, 1) Gaussian opacities.
        colors (torch.Tensor): (N, 3) Per-Gaussian colors.
        sorted_tile_ids/gaussian_ids: Output from get_tile_interactions.
        H, W (int): Resolution.
        tile_size (int): Tile dimensions.
        background (torch.Tensor): (3,) Default color.
        
    Returns:
        torch.Tensor: (H, W, 3) Final rendered image.
    """
    if background is None:
        background = torch.zeros(3, device=device)
        
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    num_tiles = num_tiles_x * num_tiles_y
    
    # 1. Pre-compute derived splat parameters
    # Determinant for inverse covariance
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    det = torch.clamp(det, min=1e-6)
    
    # Inverse covariance matrices (N, 2, 2)
    inv_cov2D = torch.stack([
        torch.stack([cov2D[:, 1, 1] / det, -cov2D[:, 0, 1] / det], dim=-1),
        torch.stack([-cov2D[:, 1, 0] / det, cov2D[:, 0, 0] / det], dim=-1)
    ], dim=-2)
    
    sig_opacities = torch.sigmoid(opacities)

    # 2. Setup tile data structures
    # Searchsorted is fine on CPU/Numpy for this scale
    # Determine the boundaries of each tile in the sorted interaction list
    sorted_tile_ids_cpu = sorted_tile_ids.cpu().numpy()
    tile_indices = np.arange(num_tiles + 1)
    tile_boundaries = np.searchsorted(sorted_tile_ids_cpu, tile_indices)
    tile_boundaries = torch.from_numpy(tile_boundaries).to(device)

    # Pixel offsets for each tile
    tile_indices_all = torch.arange(num_tiles, device=device)
    ty_all = tile_indices_all // num_tiles_x
    tx_all = tile_indices_all % num_tiles_x
    pix_min_x_all = tx_all * tile_size
    pix_min_y_all = ty_all * tile_size
    
    # Tile meshgrid (static 16x16 relative grid)
    py = torch.arange(tile_size, device=device)
    px = torch.arange(tile_size, device=device)
    grid_y, grid_x = torch.meshgrid(py, px, indexing='ij')
    local_pixel_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2) 

    # 3. Process Tiles in Batches (Chunk processing to avoid OOM)
    chunk_size = 16 # Optimized chunk size for Apple Silicon memory bandwidth
    all_tiles_list = []
    for i in range(0, num_tiles, chunk_size):
        end = min(i + chunk_size, num_tiles)
        batch = torch.arange(i, end, device=device)
        
        # Use vectorized batch renderer
        # Call vectorized batch renderer
        all_tiles_list.append(render_tile_batch(
            batch, tile_boundaries, pix_min_x_all, pix_min_y_all,
            local_pixel_coords, sorted_gaussian_ids, means2D, inv_cov2D,
            sig_opacities, colors, background, H, W, tile_size, device
        ))
        
    # 4. Final Reconstruction
    # Concatenate tile chunks and reshape to image dimensions
    all_tiles = torch.cat(all_tiles_list, dim=0)
    output_grid = all_tiles.reshape(num_tiles_y, num_tiles_x, tile_size, tile_size, 3)
    
    # Swap axes to rearrange tiles into a single HxWx3 image
    # (H_tiles, W_tiles, TileH, TileW, 3) -> (H_tiles, TileH, W_tiles, TileW, 3) -> (H, W, 3)
    output_image = output_grid.permute(0, 2, 1, 3, 4).reshape(num_tiles_y * tile_size, num_tiles_x * tile_size, 3)
    
    return output_image[:H, :W, :]

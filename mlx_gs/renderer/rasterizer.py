import mlx.core as mx
import numpy as np
import math

TILE_SIZE = 16

def _get_tile_interactions_impl(means2D, radii, valid_mask, depths, H, W, tile_size):
    num_tiles_x = math.ceil(W / tile_size)
    num_tiles_y = math.ceil(H / tile_size)
    
    # 1. Compute bounds
    m_x = mx.clip(mx.floor((means2D[:, 0] - radii) / tile_size).astype(mx.int32), 0, num_tiles_x - 1)
    max_x = mx.clip(mx.floor((means2D[:, 0] + radii) / tile_size).astype(mx.int32), 0, num_tiles_x - 1)
    m_y = mx.clip(mx.floor((means2D[:, 1] - radii) / tile_size).astype(mx.int32), 0, num_tiles_y - 1)
    max_y = mx.clip(mx.floor((means2D[:, 1] + radii) / tile_size).astype(mx.int32), 0, num_tiles_y - 1)
    
    nx = max_x - m_x + 1
    ny = max_y - m_y + 1
    counts = nx * ny * valid_mask.astype(mx.int32)
    
    # 2. Find offsets and total count
    offsets = mx.cumsum(counts)
    total = int(offsets[-1].item())
    
    if total == 0:
        return mx.array([], dtype=mx.int32), mx.array([], dtype=mx.int32)
    
    # 3. Expansion Stage (Pure MLX GPU Trick)
    active_mask = (counts > 0).astype(mx.int32)
    num_active = int(mx.sum(active_mask).item())
    active_indices = mx.sort(mx.argsort(active_mask)[-num_active:])
    
    active_counts = counts[active_indices]
    active_offsets = mx.cumsum(active_counts)
    active_starts = mx.concatenate([mx.array([0]), active_offsets[:-1]])
    
    mark = mx.zeros((total,), dtype=mx.int32)
    mark[active_starts] = 1
    map_idx = mx.cumsum(mark) - 1
    gaussian_ids = active_indices[map_idx]
    
    # 4. Compute Tile IDs
    local_idx = mx.arange(total, dtype=mx.int32) - active_starts[map_idx].astype(mx.int32)
    cur_nx = nx[gaussian_ids]
    tile_ids = (m_y[gaussian_ids] + (local_idx // cur_nx)) * num_tiles_x + (m_x[gaussian_ids] + (local_idx % cur_nx))
    
    # 5. Sort by TileID and Depth
    d_vals = depths[gaussian_ids]
    d_min, d_max = mx.min(depths), mx.max(depths)
    depth_quant = ((d_vals - d_min) / (d_max - d_min + 1e-6) * 0xFFFFFFFF).astype(mx.uint64)
    keys = (tile_ids.astype(mx.uint64) << 32) | depth_quant
    
    sort_indices = mx.argsort(keys)
    return mx.stop_gradient(tile_ids[sort_indices]), mx.stop_gradient(gaussian_ids[sort_indices])

def render_tile_batch(
    batch_indices, tile_boundaries, pix_min_x_all, pix_min_y_all, 
    local_pixel_coords, sorted_gaussian_ids, means2D, inv_cov2D, 
    sig_opacities, colors, background, H, W, tile_size
):
    B = batch_indices.shape[0]
    pix_min_x, pix_min_y = pix_min_x_all[batch_indices], pix_min_y_all[batch_indices]
    
    s_indices = tile_boundaries[batch_indices]
    e_indices = tile_boundaries[batch_indices + 1]
    counts = e_indices - s_indices
    
    LIMIT = 1024
    gather_indices = mx.clip(s_indices[:, None] + mx.arange(LIMIT)[None, :], 0, sorted_gaussian_ids.shape[0] - 1)
    gaussian_id_batch = sorted_gaussian_ids[gather_indices]
    
    local_mask = mx.arange(LIMIT)[None, :] < mx.minimum(counts[:, None], LIMIT)
    
    # Pixels
    global_pixel_coords = local_pixel_coords[None, :, :] + mx.stack([pix_min_x, pix_min_y], axis=2)
    
    # Gaussians
    means_batch = means2D[gaussian_id_batch]
    inv_cov_batch = inv_cov2D[gaussian_id_batch]
    opac_batch = sig_opacities[gaussian_id_batch]
    colors_batch = colors[gaussian_id_batch]
    
    d = global_pixel_coords.reshape(B, 1, -1, 2) - means_batch.reshape(B, LIMIT, 1, 2)
    
    # Quadratic form
    a, b, c, d_val = inv_cov_batch[:,:,0,0], inv_cov_batch[:,:,0,1], inv_cov_batch[:,:,1,0], inv_cov_batch[:,:,1,1]
    dx, dy = d[..., 0], d[..., 1]
    mahalanobis = dx**2 * a[:,:,None] + dx * dy * (b[:,:,None] + c[:,:,None]) + dy**2 * d_val[:,:,None]
    
    # Safe exponential with robust masking
    # Masking inputs to avoid NaN in exp branch if mahalanobis is invalid
    mahalanobis_safe = mx.where(local_mask[:, :, None], mahalanobis, mx.zeros_like(mahalanobis))
    gauss_values = mx.exp(-0.5 * mahalanobis_safe) * local_mask[:, :, None]
    
    alphas = 1.0 - mx.exp(-gauss_values * opac_batch.reshape(B, LIMIT, 1))
    
    # Blending
    trans = 1.0 - alphas
    T = mx.cumprod(trans, axis=1)
    T = mx.concatenate([mx.ones((B, 1, T.shape[2])), T[:, :-1, :]], axis=1)
    
    weights = alphas * T
    final_color = mx.sum(weights[:, :, :, None] * colors_batch[:, :, None, :], axis=1)
    
    T_final = mx.prod(trans, axis=1, keepdims=True)
    final_color = final_color + T_final.transpose(0, 2, 1) * background
    
    return final_color.reshape(B, tile_size, tile_size, 3)

@mx.compile
def render_tiles_jit(means2D, cov2D, sig_opacities, colors, sorted_gaussian_ids, tile_boundaries, H, W, tile_size, background):
    num_tiles_x, num_tiles_y = math.ceil(W/tile_size), math.ceil(H/tile_size)
    num_tiles = num_tiles_x * num_tiles_y
    
    # Inverse Cov
    det = cov2D[:,0,0]*cov2D[:,1,1] - cov2D[:,0,1]*cov2D[:,1,0]
    inv_det = 1.0 / mx.maximum(det, 1e-6)
    inv_cov2D = mx.zeros_like(cov2D)
    inv_cov2D[:,0,0], inv_cov2D[:,0,1] = cov2D[:,1,1]*inv_det, -cov2D[:,0,1]*inv_det
    inv_cov2D[:,1,0], inv_cov2D[:,1,1] = -cov2D[:,1,0]*inv_det, cov2D[:,0,0]*inv_det
    
    tile_indices = mx.arange(num_tiles)
    pix_min_x, pix_min_y = (tile_indices % num_tiles_x) * tile_size, (tile_indices // num_tiles_x) * tile_size
    
    tx = mx.arange(tile_size)
    grid_y, grid_x = mx.meshgrid(tx, tx, indexing='ij')
    local_pixel_coords = mx.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    
    chunk_size = 128
    all_tile_colors = []
    for i in range(0, num_tiles, chunk_size):
        batch_indices = mx.arange(i, min(i + chunk_size, num_tiles))
        all_tile_colors.append(render_tile_batch(batch_indices, tile_boundaries, pix_min_x[:, None], pix_min_y[:, None], local_pixel_coords, sorted_gaussian_ids, means2D, inv_cov2D, sig_opacities, colors, background, H, W, tile_size))
        
    full_batch = mx.concatenate(all_tile_colors, axis=0)
    canvas = full_batch.reshape(num_tiles_y, num_tiles_x, tile_size, tile_size, 3)
    return canvas.transpose(0, 2, 1, 3, 4).reshape(num_tiles_y * tile_size, num_tiles_x * tile_size, 3)[:H, :W, :]

def render_tiles(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, tile_size: int = TILE_SIZE, background=None):
    if background is None: background = mx.zeros((3,))
    
    # Sync for NumPy searchsorted
    num_tiles = math.ceil(W/tile_size) * math.ceil(H/tile_size)
    tile_boundaries = np.searchsorted(np.array(sorted_tile_ids), np.arange(num_tiles + 1))
    return render_tiles_jit(means2D, cov2D, mx.sigmoid(opacities), colors, sorted_gaussian_ids, mx.array(tile_boundaries), H, W, tile_size, background)

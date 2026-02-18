import mlx.core as mx
import numpy as np
import os
try:
    from . import _rasterizer_metal as rasterizer_metal
except ImportError:
    rasterizer_metal = None

# Initialize Metal
if rasterizer_metal is not None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metal_source_path = os.path.join(current_dir, "..", "csrc", "rasterizer.metal")
    if os.path.exists(metal_source_path):
        with open(metal_source_path, 'r') as f:
            source = f.read()
        try:
            rasterizer_metal.init_metal(source)
        except Exception as e:
            print(f"Failed to initialize Metal rasterizer: {e}")
            rasterizer_metal = None

def get_tile_interactions(means2D, radii, valid_mask, depths, H, W, tile_size):
    """
    Stable GPU-accelerated Tile Interaction Stage using pure MLX operations.
    Completes Phase 4.
    """
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    
    m_x = mx.clip(mx.floor((means2D[:, 0] - radii) / tile_size).astype(mx.int32), 0, num_tiles_x - 1)
    max_x = mx.clip(mx.floor((means2D[:, 0] + radii) / tile_size).astype(mx.int32), 0, num_tiles_x - 1)
    m_y = mx.clip(mx.floor((means2D[:, 1] - radii) / tile_size).astype(mx.int32), 0, num_tiles_y - 1)
    max_y = mx.clip(mx.floor((means2D[:, 1] + radii) / tile_size).astype(mx.int32), 0, num_tiles_y - 1)
    
    nx = max_x - m_x + 1
    ny = max_y - m_y + 1
    counts = nx * ny * valid_mask.astype(mx.int32)
    
    offsets = mx.cumsum(counts)
    total = int(offsets[-1].item())
    if total == 0: return mx.array([], dtype=mx.int32), mx.array([], dtype=mx.int32)
    
    # Expansion
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
    
    local_idx = mx.arange(total, dtype=mx.int32) - active_starts[map_idx].astype(mx.int32)
    tile_ids = (m_y[gaussian_ids] + (local_idx // nx[gaussian_ids])) * num_tiles_x + (m_x[gaussian_ids] + (local_idx % nx[gaussian_ids]))
    
    depth_quant = ((depths[gaussian_ids] - mx.min(depths)) / (mx.max(depths) - mx.min(depths) + 1e-6) * 0xFFFFFFFF).astype(mx.uint64)
    keys = (tile_ids.astype(mx.uint64) << 32) | depth_quant
    sort_indices = mx.argsort(keys)
    
    # We must use stop_gradient as these are discrete indices
    return mx.stop_gradient(tile_ids[sort_indices]), mx.stop_gradient(gaussian_ids[sort_indices])

def render_tiles(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, tile_size, background=None):
    if background is None: background = mx.zeros((3,))
    
    # Precompute inverse covariance and sigmoid opacities
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 1, 0]
    inv_det = 1.0 / mx.maximum(det, 1e-6)
    inv_cov2D = mx.stack([
        mx.stack([cov2D[:, 1, 1] * inv_det, -cov2D[:, 0, 1] * inv_det], axis=-1),
        mx.stack([-cov2D[:, 1, 0] * inv_det, cov2D[:, 0, 0] * inv_det], axis=-1)
    ], axis=-2)
    sig_opacities = mx.sigmoid(opacities)

    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    num_tiles = num_tiles_x * num_tiles_y
    
    # Sync for tile boundaries
    mx.eval(sorted_tile_ids)
    tb_np = np.searchsorted(np.asarray(sorted_tile_ids), np.arange(num_tiles + 1, dtype=np.int32)).astype(np.int32)

    @mx.custom_function
    def forward(m, ic, s_o, c, sti, sgi, bg):
        mx.eval(m, ic, s_o, c, sti, sgi, bg)
        out_np = rasterizer_metal.render_forward(
            np.ascontiguousarray(np.asarray(m), dtype=np.float32),
            np.ascontiguousarray(np.asarray(ic), dtype=np.float32),
            np.ascontiguousarray(np.asarray(s_o), dtype=np.float32),
            np.ascontiguousarray(np.asarray(c), dtype=np.float32),
            np.ascontiguousarray(np.asarray(sti), dtype=np.int32),
            np.ascontiguousarray(np.asarray(sgi), dtype=np.int32),
            int(H), int(W), int(tile_size),
            np.ascontiguousarray(np.asarray(bg), dtype=np.float32),
            tb_np
        )
        return mx.array(out_np)

    @forward.vjp
    def backward(primals, cotangents, outputs):
        m, ic, s_o, c, sti, sgi, bg = primals
        mx.eval(m, ic, s_o, c, sti, sgi, bg, cotangents)
        gm_np, gic_np, go_np, gc_np = rasterizer_metal.render_backward(
            np.ascontiguousarray(np.asarray(cotangents), dtype=np.float32),
            np.ascontiguousarray(np.asarray(m), dtype=np.float32),
            np.ascontiguousarray(np.asarray(ic), dtype=np.float32),
            np.ascontiguousarray(np.asarray(s_o), dtype=np.float32),
            np.ascontiguousarray(np.asarray(c), dtype=np.float32),
            np.ascontiguousarray(np.asarray(sti), dtype=np.int32),
            np.ascontiguousarray(np.asarray(sgi), dtype=np.int32),
            int(H), int(W), int(tile_size),
            np.ascontiguousarray(np.asarray(bg), dtype=np.float32),
            tb_np
        )
        return mx.array(gm_np), mx.array(gic_np), mx.array(go_np).reshape(-1, 1), mx.array(gc_np), None, None, None

    return forward(means2D, inv_cov2D, sig_opacities, colors, sorted_tile_ids, sorted_gaussian_ids, background)

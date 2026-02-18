import mlx.core as mx
import numpy as np
import os
from . import rasterizer_c_api # Reuse for interactions
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
    else:
        print(f"Metal source not found at {metal_source_path}")
        rasterizer_metal = None

def render_tiles(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, tile_size, background=None):
    if rasterizer_metal is None:
        raise ImportError("Metal rasterizer not available")
    
    if background is None:
        background = mx.zeros((3,))
    
    # Precompute inverse covariance and sigmoid opacities
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 1, 0]
    inv_det = 1.0 / mx.maximum(det, 1e-6)
    
    inv_cov2D = mx.stack([
        mx.stack([cov2D[:, 1, 1] * inv_det, -cov2D[:, 0, 1] * inv_det], axis=-1),
        mx.stack([-cov2D[:, 1, 0] * inv_det, cov2D[:, 0, 0] * inv_det], axis=-1)
    ], axis=-2)
    
    sig_opacities = mx.sigmoid(opacities)

    # Compute tile boundaries
    # We need to do this on CPU for now as searchsorted isn't readily available/easy on GPU without sync
    # sorted_tile_ids is (M,) int32
    # We want boundaries for tiles 0..num_tiles
    num_tiles_x = (W + tile_size - 1) // tile_size
    num_tiles_y = (H + tile_size - 1) // tile_size
    num_tiles = num_tiles_x * num_tiles_y
    
    # Force sync for boundaries computation
    mx.eval(sorted_tile_ids)
    sorted_tile_ids_np = np.asarray(sorted_tile_ids)
    
    # Compute boundaries: indices where tile_id changes
    # searchsorted finds indices where elements should be inserted to maintain order
    # For each tile t in 0..num_tiles, find first index where tile_id >= t
    # Since sorted_tile_ids is sorted, searchsorted works.
    tile_indices = np.arange(num_tiles + 1, dtype=np.int32)
    tile_boundaries_np = np.searchsorted(sorted_tile_ids_np, tile_indices).astype(np.int32)
    
    # We don't need to convert boundaries back to MLX, we pass numpy array to extension
    # But extension expects nb::ndarray which handles numpy.

    @mx.custom_function
    def forward(m, ic, s_o, c, sti, sgi, bg):
        mx.eval(m, ic, s_o, c, sti, sgi, bg)
        
        # Zero-copy views if possible (MLX -> Numpy usually zero-copy if contiguous)
        m_np = np.asarray(m)
        ic_np = np.asarray(ic)
        s_o_np = np.asarray(s_o)
        c_np = np.asarray(c)
        sti_np = np.asarray(sti)
        sgi_np = np.asarray(sgi)
        bg_np = np.asarray(bg)
        
        out_np = rasterizer_metal.render_forward(
            m_np, ic_np, s_o_np, c_np, sti_np, sgi_np, H, W, tile_size, bg_np, tile_boundaries_np
        )
        return mx.array(out_np)

    @forward.vjp
    def forward_vjp(primals, cotangents, outputs):
        m, ic, s_o, c, sti, sgi, bg = primals
        grad_output = cotangents
        
        mx.eval(m, ic, s_o, c, sti, sgi, bg, grad_output)
        
        m_np = np.asarray(m)
        ic_np = np.asarray(ic)
        s_o_np = np.asarray(s_o)
        c_np = np.asarray(c)
        sti_np = np.asarray(sti)
        sgi_np = np.asarray(sgi)
        bg_np = np.asarray(bg)
        grad_output_np = np.asarray(grad_output)
        
        gm_np, gic_np, go_np, gc_np = rasterizer_metal.render_backward(
            grad_output_np, m_np, ic_np, s_o_np, c_np, sti_np, sgi_np, H, W, tile_size, bg_np, tile_boundaries_np
        )
        
        return mx.array(gm_np), mx.array(gic_np), mx.array(go_np), mx.array(gc_np), None, None, None

    return forward(means2D, inv_cov2D, sig_opacities, colors, sorted_tile_ids, sorted_gaussian_ids, background)

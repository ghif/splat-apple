import mlx.core as mx
import numpy as np
from . import rasterizer_c_api

def get_tile_interactions(means2D, radii, valid_mask, depths, H, W, tile_size):
    """
    Compute tile interactions using C++ extension.
    """
    # Force evaluation of MLX arrays before converting to NumPy
    mx.eval(means2D, radii, valid_mask, depths)
    
    means2D_np = np.asarray(means2D)
    radii_np = np.asarray(radii)
    valid_mask_np = np.asarray(valid_mask)
    depths_np = np.asarray(depths)
    
    tile_ids, gaussian_ids = rasterizer_c_api.get_tile_interactions(
        means2D_np, radii_np, valid_mask_np, depths_np, H, W, tile_size
    )
    return mx.array(tile_ids), mx.array(gaussian_ids)

def render_tiles(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, tile_size, background=None):
    """
    Differentiable tile-based rasterization using C++ extension.
    """
    if background is None:
        background = mx.zeros((3,))
    
    # Precompute inverse covariance and sigmoid opacities for the custom function
    # This keeps the custom function pure and focused on the heavy lifting
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 1, 0]
    inv_det = 1.0 / mx.maximum(det, 1e-6)
    
    # Construct inv_cov2D: [[d, -b], [-c, a]] * inv_det
    inv_cov2D = mx.stack([
        mx.stack([cov2D[:, 1, 1] * inv_det, -cov2D[:, 0, 1] * inv_det], axis=-1),
        mx.stack([-cov2D[:, 1, 0] * inv_det, cov2D[:, 0, 0] * inv_det], axis=-1)
    ], axis=-2)
    
    sig_opacities = mx.sigmoid(opacities)

    @mx.custom_function
    def forward(m, ic, s_o, c, sti, sgi, bg):
        # Force evaluation and convert to numpy
        mx.eval(m, ic, s_o, c, sti, sgi, bg)
        
        m_np = np.asarray(m)
        ic_np = np.asarray(ic)
        s_o_np = np.asarray(s_o)
        c_np = np.asarray(c)
        sti_np = np.asarray(sti)
        sgi_np = np.asarray(sgi)
        bg_np = np.asarray(bg)
        
        out_np = rasterizer_c_api.render_tiles_forward(
            m_np, ic_np, s_o_np, c_np, sti_np, sgi_np, H, W, tile_size, bg_np
        )
        return mx.array(out_np)

    @forward.vjp
    def forward_vjp(primals, cotangents, outputs):
        m, ic, s_o, c, sti, sgi, bg = primals
        grad_output = cotangents
        
        # Force evaluation and convert to numpy
        mx.eval(m, ic, s_o, c, sti, sgi, bg, grad_output)
        
        m_np = np.asarray(m)
        ic_np = np.asarray(ic)
        s_o_np = np.asarray(s_o)
        c_np = np.asarray(c)
        sti_np = np.asarray(sti)
        sgi_np = np.asarray(sgi)
        bg_np = np.asarray(bg)
        grad_output_np = np.asarray(grad_output)
        
        gm_np, gic_np, go_np, gc_np = rasterizer_c_api.render_tiles_backward(
            grad_output_np, m_np, ic_np, s_o_np, c_np, sti_np, sgi_np, H, W, tile_size, bg_np
        )
        
        return mx.array(gm_np), mx.array(gic_np), mx.array(go_np), mx.array(gc_np), None, None, None

    return forward(means2D, inv_cov2D, sig_opacities, colors, sorted_tile_ids, sorted_gaussian_ids, background)

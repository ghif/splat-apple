import torch
from torch.autograd import Function
import os
import numpy as np

# Try to import the extension
try:
    from .. import _MPS as mps_ext
except ImportError:
    mps_ext = None

# Initialize Metal kernels for PyTorch
if mps_ext is not None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metal_source_path = os.path.join(current_dir, "..", "..", "mlx_gs", "csrc", "rasterizer.metal")
    if os.path.exists(metal_source_path):
        with open(metal_source_path, 'r') as f:
            source = f.read()
        try:
            mps_ext.init_mps(source)
        except Exception as e:
            print(f"Failed to initialize PyTorch MPS rasterizer: {e}")
            mps_ext = None

def get_ptr(t): return t.data_ptr()
def get_sz(t): return t.element_size() * t.numel()

class RasterizerMPSFunction(Function):
    @staticmethod
    def forward(ctx, means2D, inv_cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, tile_size, background, tile_boundaries):
        ctx.save_for_backward(means2D, inv_cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, background, tile_boundaries)
        ctx.H, ctx.W, ctx.tile_size = H, W, tile_size
        
        # Ensure all are on CPU and contiguous for the robust bypass
        m_c = means2D.cpu().contiguous()
        ic_c = inv_cov2D.cpu().contiguous()
        o_c = opacities.cpu().contiguous()
        c_c = colors.cpu().contiguous()
        sti_c = sorted_tile_ids.cpu().contiguous()
        sgi_c = sorted_gaussian_ids.cpu().contiguous()
        bg_c = background.cpu().contiguous()
        tb_c = tile_boundaries.cpu().contiguous()
        
        output_c = torch.zeros((H, W, 3), dtype=torch.float32)
        
        mps_ext.render_forward(
            get_ptr(m_c), get_ptr(ic_c), get_ptr(o_c), get_ptr(c_c),
            get_ptr(sti_c), get_ptr(sgi_c), get_ptr(bg_c), get_ptr(output_c),
            H, W, tile_size, sorted_tile_ids.shape[0], get_ptr(tb_c),
            get_sz(m_c), get_sz(ic_c), get_sz(o_c), get_sz(c_c),
            get_sz(sti_c), get_sz(sgi_c), get_sz(bg_c), get_sz(output_c), get_sz(tb_c)
        )
        
        return output_c.to(means2D.device)

    @staticmethod
    def backward(ctx, grad_out):
        m, ic, o, c, sti, sgi, bg, tb = ctx.saved_tensors
        
        # Prepare CPU proxies
        go_c = grad_out.cpu().contiguous()
        m_c = m.cpu().contiguous()
        ic_c = ic.cpu().contiguous()
        o_c = o.cpu().contiguous()
        c_c = c.cpu().contiguous()
        sti_c = sti.cpu().contiguous()
        sgi_c = sgi.cpu().contiguous()
        bg_c = bg.cpu().contiguous()
        tb_c = tb.cpu().contiguous()
        
        gm_c = torch.zeros_like(m_c)
        gic_c = torch.zeros_like(ic_c)
        go_c_res = torch.zeros_like(o_c)
        gc_c = torch.zeros_like(c_c)
        
        mps_ext.render_backward(
            get_ptr(go_c), get_ptr(m_c), get_ptr(ic_c), get_ptr(o_c), get_ptr(c_c),
            get_ptr(sti_c), get_ptr(sgi_c), get_ptr(bg_c),
            get_ptr(gm_c), get_ptr(gic_c), get_ptr(go_c_res), get_ptr(gc_c),
            ctx.H, ctx.W, ctx.tile_size, get_ptr(tb_c),
            get_sz(go_c), get_sz(m_c), get_sz(ic_c), get_sz(o_c), get_sz(c_c),
            get_sz(sti_c), get_sz(sgi_c), get_sz(bg_c),
            get_sz(gm_c), get_sz(gic_c), get_sz(go_c_res), get_sz(gc_c), get_sz(tb_c)
        )
        
        device = m.device
        return gm_c.to(device), gic_c.to(device), go_c_res.to(device), gc_c.to(device), None, None, None, None, None, None, None

def render_tiles_mps(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, tile_size, background=None, device="mps"):
    if mps_ext is None: raise ImportError("PyTorch MPS rasterizer not available")
    if background is None: background = torch.zeros(3, device=device)
    
    # Precompute
    det = (cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2).clamp(min=1e-6)
    inv_cov2D = torch.stack([
        torch.stack([cov2D[:, 1, 1]/det, -cov2D[:, 0, 1]/det], -1),
        torch.stack([-cov2D[:, 1, 0]/det, cov2D[:, 0, 0]/det], -1)
    ], -2)
    sig_opacities = torch.sigmoid(opacities)
    
    nt_x, nt_y = (W + tile_size - 1) // tile_size, (H + tile_size - 1) // tile_size
    num_tiles = nt_x * nt_y
    tile_boundaries = torch.searchsorted(sorted_tile_ids, torch.arange(num_tiles + 1, device=device, dtype=torch.int32)).to(torch.int32)
    
    return RasterizerMPSFunction.apply(
        means2D, inv_cov2D, sig_opacities, colors, 
        sorted_tile_ids, sorted_gaussian_ids, 
        H, W, tile_size, background, tile_boundaries
    )

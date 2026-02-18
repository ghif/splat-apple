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
        
        # We MUST move to CPU for the independent Metal implementation to avoid segfaults
        # when passing pointers to another Metal queue.
        # On Mac, this is zero-copy if the tensor is already in shared memory.
        m_cpu = means2D.cpu().contiguous()
        ic_cpu = inv_cov2D.cpu().contiguous()
        o_cpu = opacities.cpu().contiguous()
        c_cpu = colors.cpu().contiguous()
        sti_cpu = sorted_tile_ids.cpu().contiguous()
        sgi_cpu = sorted_gaussian_ids.cpu().contiguous()
        bg_cpu = background.cpu().contiguous()
        tb_cpu = tile_boundaries.cpu().contiguous()
        
        output_cpu = torch.zeros((H, W, 3), dtype=torch.float32)
        
        mps_ext.render_forward(
            get_ptr(m_cpu), get_ptr(ic_cpu), get_ptr(o_cpu), get_ptr(c_cpu),
            get_ptr(sti_cpu), get_ptr(sgi_cpu), get_ptr(bg_cpu), get_ptr(output_cpu),
            H, W, tile_size, sorted_tile_ids.shape[0], get_ptr(tb_cpu),
            get_sz(m_cpu), get_sz(ic_cpu), get_sz(o_cpu), get_sz(c_cpu),
            get_sz(sti_cpu), get_sz(sgi_cpu), get_sz(bg_cpu), get_sz(output_cpu), get_sz(tb_cpu)
        )
        
        return output_cpu.to(means2D.device)

    @staticmethod
    def backward(ctx, grad_out):
        m, ic, o, c, sti, sgi, bg, tb = ctx.saved_tensors
        
        go_cpu = grad_out.cpu().contiguous()
        m_cpu = m.cpu().contiguous()
        ic_cpu = ic.cpu().contiguous()
        o_cpu = o.cpu().contiguous()
        c_cpu = c.cpu().contiguous()
        sti_cpu = sti.cpu().contiguous()
        sgi_cpu = sgi.cpu().contiguous()
        bg_cpu = bg.cpu().contiguous()
        tb_cpu = tb.cpu().contiguous()
        
        gm_cpu = torch.zeros_like(m_cpu)
        gic_cpu = torch.zeros_like(ic_cpu)
        go_cpu_res = torch.zeros_like(o_cpu) # grad_opacities
        gc_cpu = torch.zeros_like(c_cpu)
        
        mps_ext.render_backward(
            get_ptr(go_cpu), get_ptr(m_cpu), get_ptr(ic_cpu), get_ptr(o_cpu), get_ptr(c_cpu),
            get_ptr(sti_cpu), get_ptr(sgi_cpu), get_ptr(bg_cpu),
            get_ptr(gm_cpu), get_ptr(gic_cpu), get_ptr(go_cpu_res), get_ptr(gc_cpu),
            ctx.H, ctx.W, ctx.tile_size, get_ptr(tb_cpu),
            get_sz(go_cpu), get_sz(m_cpu), get_sz(ic_cpu), get_sz(o_cpu), get_sz(c_cpu),
            get_sz(sti_cpu), get_sz(sgi_cpu), get_sz(bg_cpu),
            get_sz(gm_cpu), get_sz(gic_cpu), get_sz(go_cpu_res), get_sz(gc_cpu), get_sz(tb_cpu)
        )
        
        device = m.device
        return gm_cpu.to(device), gic_cpu.to(device), go_cpu_res.to(device), gc_cpu.to(device), None, None, None, None, None, None, None

def render_tiles_mps(means2D, cov2D, opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, tile_size, background=None, device="mps"):
    if mps_ext is None: raise ImportError("PyTorch MPS rasterizer not available")
    if background is None: background = torch.zeros(3, device=device)
    det = (cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2).clamp(min=1e-6)
    inv_cov2D = torch.stack([torch.stack([cov2D[:, 1, 1]/det, -cov2D[:, 0, 1]/det], -1), torch.stack([-cov2D[:, 1, 0]/det, cov2D[:, 0, 0]/det], -1)], -2)
    sig_opacities = torch.sigmoid(opacities)
    nt_x, nt_y = (W + tile_size - 1) // tile_size, (H + tile_size - 1) // tile_size
    num_tiles = nt_x * nt_y
    tile_boundaries = torch.searchsorted(sorted_tile_ids, torch.arange(num_tiles + 1, device=device, dtype=torch.int32)).to(torch.int32)
    return RasterizerMPSFunction.apply(means2D, inv_cov2D, sig_opacities, colors, sorted_tile_ids, sorted_gaussian_ids, H, W, tile_size, background, tile_boundaries)

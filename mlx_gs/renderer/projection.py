import mlx.core as mx
import math
from mlx_gs.core.gaussians import get_covariance_3d

def project_gaussians(gaussians, camera):
    """
    Standard 3D to 2D projection with anti-aliasing bias.
    """
    means3D = gaussians.means
    
    # 1. To camera space
    W2C = camera.W2C
    means_homo = mx.concatenate([means3D, mx.ones((means3D.shape[0], 1))], axis=1)
    means_cam = (W2C @ means_homo.T).T
    
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    
    # Filter points behind camera
    valid_mask = z > 0.01
    
    # 2. Project to screen
    fx, fy = camera.fx, camera.fy
    cx, cy = camera.cx, camera.cy
    
    # Safe division
    z_safe = mx.maximum(z, 1e-6)
    means2D = mx.stack([fx * (x / z_safe) + cx, fy * (y / z_safe) + cy], axis=1)
    
    # 3. 2D Covariance
    Sigma = get_covariance_3d(gaussians.scales, gaussians.quaternions)
    
    # Jacobian
    J = mx.zeros((z.shape[0], 2, 3))
    J[:, 0, 0] = fx / z_safe
    J[:, 0, 2] = -(fx * x) / (z_safe * z_safe)
    J[:, 1, 1] = fy / z_safe
    J[:, 1, 2] = -(fy * y) / (z_safe * z_safe)
    
    W_rot = W2C[:3, :3]
    M = J @ W_rot
    # Use explicit transpose for backprop stability
    cov2D = M @ Sigma @ M.transpose(0, 2, 1)
    
    # Anti-aliasing bias (CRITICAL for visibility)
    cov2D = cov2D + mx.eye(2) * 0.3
    
    # 4. Filter and Radii
    tr = cov2D[:, 0, 0] + cov2D[:, 1, 1]
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 1, 0]
    mid = 0.5 * tr
    
    # Safe sqrt for term
    term = mx.sqrt(mx.maximum(1e-10, mid * mid - det))
    lambda1 = mid + term
    
    # Safe sqrt for radii
    radii = mx.ceil(3.0 * mx.sqrt(mx.maximum(1e-10, lambda1)))
    
    # Mask invalid to ensure they don't contribute
    radii = mx.where(valid_mask, radii, mx.zeros_like(radii))
    
    return means2D, cov2D, radii, valid_mask, z

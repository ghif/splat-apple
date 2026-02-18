import jax
import jax.numpy as jnp
from jax_gs.core.gaussians import Gaussians, get_covariance_3d
from jax_gs.core.camera import Camera

def project_gaussians(gaussians: Gaussians, camera: Camera):
    """
    Project 3D Gaussians to 2D splats.

    Args:
        gaussians: Gaussians dataclass
        camera: Camera dataclass 
    Returns:
        means2D: 2D means of the projected splats
        cov2D: 2D covariance of the projected splats
        radii: Radii of the projected splats
        valid_mask: Valid mask for the projected splats
        z: Depth of the projected splats
    """
    means3D = gaussians.means
    scales = gaussians.scales
    quats = gaussians.quaternions
    
    # 1. Transform means
    means3D_homo = jnp.concatenate([means3D, jnp.ones((means3D.shape[0], 1))], axis=-1)
    means_cam = (means3D_homo @ camera.W2C.T)[:, :3]
    
    x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    
    # 2. Filter 
    valid_mask = z > 0.01
    
    # 3. Covariance
    cov3D = get_covariance_3d(scales, quats)
    
    # 4. Project to 2D
    # Jacobian of the perspective transformation
    # J = [fx/z, 0, -fx*x/z^2]
    #     [0, fy/z, -fy*y/z^2]
    inv_z = 1.0 / z
    inv_z2 = inv_z**2
    
    J = jnp.stack([
        jnp.stack([camera.fx * inv_z, jnp.zeros_like(z), -camera.fx * x * inv_z2], axis=-1),
        jnp.stack([jnp.zeros_like(z), camera.fy * inv_z, -camera.fy * y * inv_z2], axis=-1)
    ], axis=-2)
    
    W_rot = camera.W2C[:3, :3]
    
    # cov2D = J @ W_rot @ cov3D @ W_rot.T @ J.T
    # Pre-multiply J @ W_rot
    M = J @ W_rot
    cov2D = M @ cov3D @ M.transpose(0, 2, 1)

    # Add a small bias for numerical stability (low pass filter)
    # Use broadcasting for consistent speed
    eye2D = jnp.eye(2)[None, :, :]
    cov2D = cov2D + 0.3 * eye2D
    
    # 5. Means 2D
    means2D = jnp.stack([
        camera.fx * x / z + camera.cx,
        camera.fy * y / z + camera.cy
    ], axis=-1)
    
    # 6. Radii for tile interaction
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    trace = cov2D[:, 0, 0] + cov2D[:, 1, 1]
    mid = trace / 2.0
    term = jnp.sqrt(jnp.maximum(mid**2 - det, 0.0))
    lambda1 = mid + term
    max_eigen = lambda1 
    radii = jnp.ceil(3.0 * jnp.sqrt(max_eigen))
    
    return means2D, cov2D, radii, valid_mask, z
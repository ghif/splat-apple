import jax
import jax.numpy as jnp
import chex

@chex.dataclass
class Gaussians:
    means: jnp.ndarray  # (N, 3)
    scales: jnp.ndarray  # (N, 3)
    quaternions: jnp.ndarray  # (N, 4)
    opacities: jnp.ndarray  # (N, 1)
    sh_coeffs: jnp.ndarray  # (N, K, 3) where K is num SH coefficients

def init_gaussians_from_pcd(points: jnp.ndarray, colors: jnp.ndarray):
    """
    Initialize Gaussians from a point cloud.

    Args:
        points: (N, 3)
        colors: (N, 3) in [0, 1]
    Returns:
        gaussians: Gaussians dataclass
    """
    num_points = points.shape[0]
    
    # Position: mean of the point cloud
    means = points
    
    # Scales: log of the distance to the nearest neighbors
    # Initialized to a small value (approx 0.05m)
    scales = jnp.full((num_points, 3), -3.0) 
    
    # Rotations: identity quaternions [1, 0, 0, 0]
    quaternions = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (num_points, 1))
    
    # Opacities: inverse sigmoid of 0.5 = 0.0
    opacities = jnp.full((num_points, 1), 0.0) 
    
    # SH Coefficients (DC term only)
    # SH_DC = (R - 0.5) / 0.28209
    sh_dc = (colors - 0.5) / 0.28209479177387814
    sh_coeffs = jnp.zeros((num_points, 16, 3)) # Degree 3 SH -> 16 coefficients
    sh_coeffs = sh_coeffs.at[:, 0, :].set(sh_dc)
    
    return Gaussians(
        means=means,
        scales=scales,
        quaternions=quaternions,
        opacities=opacities,
        sh_coeffs=sh_coeffs
    )

def get_covariance_3d(scales: jnp.ndarray, quaternions: jnp.ndarray):
    """
    Computes 3D covariance matrix from scales and quaternions.
    Σ = R S S^T R^T

    Args:
        scales: (N, 3)
        quaternions: (N, 4)
    Returns:
        covariance: (N, 3, 3)
    """
    # Normalize quaternions
    q = quaternions / jnp.linalg.norm(quaternions, axis=-1, keepdims=True)
    
    # Rotation matrix from quaternion
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    
    R = jnp.stack([
        jnp.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*r*z, 2*x*z + 2*r*y], axis=-1),
        jnp.stack([2*x*y + 2*r*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*r*x], axis=-1),
        jnp.stack([2*x*z - 2*r*y, 2*y*z + 2*r*x, 1 - 2*x**2 - 2*y**2], axis=-1)
    ], axis=-2)
    
    # Scaling matrix (avoid vmap)
    s = jnp.exp(scales)
    
    # M = R S. Since S is diagonal, this is just scaling columns of R
    M = R * s[:, None, :]
    
    # Σ = M M^T
    Sigma = M @ M.transpose(0, 2, 1)
    
    return Sigma
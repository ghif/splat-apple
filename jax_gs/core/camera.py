from typing import NamedTuple
import jax.numpy as jnp

class Camera(NamedTuple):
    W: int # Image width
    H: int # Image height
    fx: float # Focal length in x direction
    fy: float # Focal length in y direction
    cx: float # Principal point in x direction
    cy: float # Principal point in y direction
    W2C: jnp.ndarray # World to Camera transformation matrix
    full_proj: jnp.ndarray # Full projection matrix

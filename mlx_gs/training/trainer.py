import mlx.core as mx
from mlx_gs.renderer.renderer import render
from mlx_gs.training.losses import l1_loss, d_ssim_loss
from dataclasses import dataclass

@dataclass
class Camera:
    W: int
    H: int
    fx: float
    fy: float
    cx: float
    cy: float
    W2C: mx.array
    full_proj: mx.array = None

def loss_fn(params, target_image, camera, lambda_ssim, rasterizer_type):
    """
    Computes loss for MLX value_and_grad.
    params is a dict or dataclass of MLX arrays.
    """
    image = render(params, camera, rasterizer_type=rasterizer_type)
    
    l1 = l1_loss(image, target_image)
    d_ssim = d_ssim_loss(image, target_image)
    loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * d_ssim
    
    return loss, image

def train_step(params, optimizers, target_image, camera, lambda_ssim=0.2, rasterizer_type="python"):
    """
    Performs a single optimization step using MLX.
    Supports a dict of optimizers for per-parameter learning rates.
    """
    def wrapped_loss(params):
        return loss_fn(params, target_image, camera, lambda_ssim, rasterizer_type)
    
    # MLX computes gradients with respect to the first argument
    loss_and_grad_fn = mx.value_and_grad(wrapped_loss)
    
    (loss, rendered_image), grads = loss_and_grad_fn(params)
    
    # Calculate gradient norms for logging
    grad_norms = {k: mx.linalg.norm(v) for k, v in grads.items()}
    
    # Update parameters using the optimizer(s)
    if isinstance(optimizers, dict):
        for key in grads:
            if key in optimizers:
                # Update specific parameter using its dedicated optimizer
                params[key] = optimizers[key].apply_gradients({key: grads[key]}, {key: params[key]})[key]
    else:
        optimizers.update(params, grads)
        
    # Standard PSNR calculation
    mse = mx.mean(mx.square(rendered_image - target_image))
    psnr = -10.0 * mx.log10(mx.maximum(mse, 1e-10))
        
    return loss, rendered_image, psnr, grad_norms

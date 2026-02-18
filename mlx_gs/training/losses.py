import mlx.core as mx
import math

def l1_loss(output, gt):
    return mx.mean(mx.abs(output - gt))

def l2_loss(output, gt):
    return mx.mean(mx.square(output - gt))

def gaussian_kernel(window_size, sigma):
    x = mx.arange(window_size) - window_size // 2
    gauss = mx.exp(-mx.square(x) / (2 * sigma**2))
    return gauss / mx.sum(gauss)

def create_window(window_size, channel):
    _1d = gaussian_kernel(window_size, 1.5)
    _2d = mx.outer(_1d, _1d)
    # For depthwise convolution in MLX (groups=C):
    # Weight shape should be (C, window_size, window_size, 1)
    return mx.tile(_2d[None, :, :, None], (channel, 1, 1, 1))

def ssim(img1, img2, window_size=11):
    """
    SSIM in MLX. Assuming NHWC inputs.
    """
    C = img1.shape[-1]
    window = create_window(window_size, C)
    
    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Use depthwise convolution (groups=C)
    mu1 = mx.conv2d(img1, window, stride=1, padding=window_size//2, groups=C)
    mu2 = mx.conv2d(img2, window, stride=1, padding=window_size//2, groups=C)
    
    mu1_sq = mx.square(mu1)
    mu2_sq = mx.square(mu2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = mx.conv2d(mx.square(img1), window, stride=1, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = mx.conv2d(mx.square(img2), window, stride=1, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = mx.conv2d(img1 * img2, window, stride=1, padding=window_size//2, groups=C) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return mx.mean(ssim_map)

def d_ssim_loss(img1, img2):
    """
    1 - SSIM loss in MLX.
    Ensures BHWC format.
    """
    if len(img1.shape) == 3:
        img1 = img1[None, ...]
        img2 = img2[None, ...]
        
    return 1.0 - ssim(img1, img2)

"""
Loss functions for Gaussian Splatting training.
Includes L1, L2, and the Structural Similarity Index (SSIM) loss
which is critical for preserving fine details in rendered images.
"""
import torch
import torch.nn.functional as F
from math import exp

def l1_loss(network_output, gt):
    """
    Standard L1 (Mean Absolute Error) loss.
    
    Args:
        network_output (torch.Tensor): Rendered image.
        gt (torch.Tensor): Ground truth image.
        
    Returns:
        torch.Tensor: Scalar L1 loss.
    """
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    """
    Standard L2 (Mean Squared Error) loss.
    
    Args:
        network_output (torch.Tensor): Rendered image.
        gt (torch.Tensor): Ground truth image.
        
    Returns:
        torch.Tensor: Scalar L2 loss.
    """
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    """
    Generates a 1D Gaussian kernel.
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    Creates a 2D Gaussian window for SSIM computation.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """
    Structural Similarity Index (SSIM).
    Measures the similarity between two images based on luminance, contrast, and structure.
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    
    # Ensure window is on the same device as images
    window = window.to(img1.device).type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    Internal SSIM implementation using 2D convolutions.
    """
    # 1. Compute means using a Gaussian window
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 2. Compute variances and covariances
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 3. SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def d_ssim_loss(img1, img2):
    """
    1 - SSIM loss. 
    A higher value indicates more structural difference.
    
    This wrapper handles shape conversion (HWC/BHWC to BCHW) automatically.
    
    Args:
        img1 (torch.Tensor): Rendered image.
        img2 (torch.Tensor): Ground truth image.
        
    Returns:
        torch.Tensor: Scalar (1 - SSIM) loss.
    """
    # Permute to (B, C, H, W) for SSIM calculation
    if img1.ndim == 3:
        img1 = img1.permute(2, 0, 1).unsqueeze(0)
        img2 = img2.permute(2, 0, 1).unsqueeze(0)
    elif img1.ndim == 4:
        img1 = img1.permute(0, 3, 1, 2)
        img2 = img2.permute(0, 3, 1, 2)
        
    return 1.0 - ssim(img1, img2)

# Splat Apple: 3D Gaussian Splatting on Apple Silicon

This repository provides high-performance implementations of 3D Gaussian Splatting optimized for Apple Silicon (M1/M2/M3), supporting both **PyTorch (MPS)** and **MLX**.

## Recent Updates (February 2026)

### MLX Native Rasterizers
- **Metal GPU Rasterizer**: A fully GPU-resident rasterizer implemented in Metal Shading Language. It achieves ~10 it/s on M-series chips, matching specialized PyTorch C++ performance while remaining entirely within the MLX ecosystem.
- **C++ CPU Rasterizer**: A multi-threaded CPU implementation using Apple's Grand Central Dispatch (GCD) for parallel tile-based rendering.

### PyTorch C++ Rasterizer Fixes
- **Differentiable Covariance**: Fixed the `grad_cov2D` implementation to allow Gaussians to grow and rotate during training.
- **Atomic Gradient Accumulation**: Implemented thread-safe gradient updates for overlapping Gaussians on the CPU.

## Setup Instructions

Always use the `gs-mps` conda environment.

### 1. Build PyTorch C++ Extension
Required for the `cpp` rasterizer mode in PyTorch.
```bash
python setup.py build_ext --inplace
```

### 2. Build MLX Native Extensions
Required for the `c_api` (CPU) and `metal` (GPU) rasterizer modes in MLX.
```bash
python setup_mlx.py build_ext --inplace
```

## Running Training

### MLX Training (Recommended)
The MLX implementation is highly optimized for Apple Silicon unified memory.
- **Metal GPU (Fastest)**:
  ```bash
  python train_fern_mlx.py --rasterizer metal
  ```
- **C++ CPU**:
  ```bash
  python train_fern_mlx.py --rasterizer c_api
  ```

### PyTorch Training
- **C++ Optimized**:
  ```bash
  python train_fern_torch.py --rasterizer cpp
  ```

## Benchmarking
Compare the performance of all implementations:
```bash
export PYTHONPATH=$PYTHONPATH:.
python tests/benchmark_mlx_vs_torch.py
```

## Documentation
For technical details on the MLX implementation strategy, see [MLX_C_STRATEGY.md](MLX_C_STRATEGY.md).

# Splat Apple: 3D Gaussian Splatting on Apple Silicon

This repository provides high-performance implementations of 3D Gaussian Splatting (3DGS) optimized for Apple Silicon (M1/M2/M3/M4). It supports both **MLX** and **PyTorch (MPS)** backends, delivering production-grade speeds that rival discrete GPU setups.

## Performance Highlights (Apple M4)

Benchmarked on the standard **Fern** scene (10,091 Gaussians):

| Backend | Implementation | Speed (Steady-State) | Speedup |
| :--- | :--- | :--- | :--- |
| **MLX** | **C++ Metal (GPU-Resident)** | **~42.3 it/s** | **50x** |
| **PyTorch** | **C++ (GCD)** | **~10.6 it/s** | **12x** |
| **MLX** | Pure Python (Reference) | ~1.2 it/s | 1.4x |
| **PyTorch** | Pure Python (Reference) | ~0.8 it/s | 1.0x |

**Fern Scene**  
![Training Progress Fern](resources/training_progress.gif)

**Pinecone Scene**  
![Training Progress Pinecone](resources/pinecone_training.gif)



---

## Notes on C++ Rasterizer
On C++ mode, the rasterizer is implemented differently between MLX and PyTorch. MLX uses the full Metal Performance Shaders (MPS) framework, while PyTorch uses the GCD framework. This makes MLX able to achieve much higher performance than PyTorch, as it can fully utilize the GPU's parallel processing capabilities.

---

## Installation & Setup

### 1. Environment Setup (Conda)
It is recommended to use **Conda** to manage the environment for this project. 

Create and activate the `gs-mps` environment:
```bash
# Create the environment with Python 3.13
conda create -n gs-mps python=3.13 -y

# Activate the environment
conda activate gs-mps

# Install dependencies
pip install -r requirements.txt
```

### 2. Build MLX Native Extensions (Highly Recommended)
Required for the high-performance `cpp` mode in MLX.
```bash
python setup_mlx.py build_ext --inplace
```

### 3. Build PyTorch C++ Extensions
Required for the optimized `cpp` mode in PyTorch.
```bash
python setup.py build_ext --inplace
```

---

## Running Training

### MLX Implementation (Fastest)
The MLX version is designed from the ground up for Apple's Unified Memory Architecture.
```bash
# High-Performance Metal Mode (Recommended)
python train_fern_mlx.py --rasterizer cpp

# High-Quality Python Reference
python train_fern_mlx.py --rasterizer python
```

### PyTorch Implementation (Stable)
```bash
# Multi-threaded C++ Mode
python train_fern_torch.py --rasterizer cpp

# High-Quality Python Reference
python train_fern_torch.py --rasterizer python
```
---

## Benchmarking
To compare the performance of all implementations on your specific hardware:
```bash
export PYTHONPATH=$PYTHONPATH:.
python tests/benchmark_mlx_vs_torch.py --num_runs 10
```
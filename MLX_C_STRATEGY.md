# MLX Rasterizer Strategy

This document explains the implementation and optimization strategy for the MLX-based 3D Gaussian Splatting native rasterizers (CPU and GPU).

## Implementation Overview

We provide two optimized native rasterizers to replace the pure Python/JIT implementation:
1.  **C++ Rasterizer (`mlx_gs/csrc/rasterizer.cpp`)**: A multi-threaded CPU implementation using Apple's Grand Central Dispatch (GCD).
2.  **Metal Rasterizer (`mlx_gs/csrc/rasterizer.metal`)**: A high-performance GPU-resident implementation using Metal Compute Shaders.

Both are integrated into the MLX autograd system via `mx.custom_function` and `nanobind`.

## Key Technologies

### 1. Nanobind (Python/C++ Interop)
We use `nanobind` for both extensions.
- **Efficiency**: It provides zero-copy memory access to MLX arrays. By passing MLX arrays as NumPy-compatible views (`nb::ndarray`), we avoid expensive memory copies between the MLX device memory and the C++/Metal execution contexts.

### 2. Metal (GPU) Rasterization
The Metal implementation (`mlx_gs/renderer/rasterizer_metal.py`) eliminates the CPU-GPU synchronization bottleneck.
- **Metal Shading Language (MSL)**: Forward and backward passes are implemented as compute kernels.
- **Zero-Copy Hosting**: The Objective-C++ host code uses `newBufferWithBytesNoCopy` to wrap MLX's underlying memory directly into Metal buffers.
- **Atomic Gradients**: Thread-safe gradient accumulation is handled on the GPU using atomic operations (mapped via `atomic_uint` loop for float support).

### 3. Grand Central Dispatch (GCD)
The C++ rasterizer utilizes Apple's **GCD (`dispatch_apply`)**.
- **Tile-Parallelism**: Rendering is split into 16x16 tiles, distributed across Performance and Efficiency cores. This serves as a robust fallback or secondary optimization path.

## Optimization Details

- **Custom VJP**: Both extensions implement the Vector-Jacobian Product (VJP). This allows MLX to treat the native rasterizer as a differentiable node in the computation graph.
- **Hybrid Interaction Stage**: Currently, both versions share the C++ interaction pre-computation (tile sorting) to keep the GPU kernels focused on the heavy alpha-blending and gradient integration.
- **Memory Management**: We use `np.asarray()` to ensure that data passed to the extensions remains a view of the original MLX array whenever possible.

## Performance Comparison (Fern Dataset)

Benchmarks show a progressive speedup across implementations:

| Implementation | Avg Time/it | Speed (it/s) | Speedup vs Python |
| :--- | :--- | :--- | :--- |
| **MLX (Python)** | 0.24s | 4.2 it/s | 1.0x |
| **MLX (C++ CPU)** | 0.19s | 5.3 it/s | 1.3x |
| **MLX (Metal GPU)** | **0.10s** | **10.0 it/s** | **2.4x** |
| **PyTorch (C++)** | 0.09s | 11.0 it/s | 2.6x |

The Metal implementation effectively matches the performance of specialized PyTorch C++ extensions on Apple Silicon.

## Status & Roadmap
- [x] **Phase 1**: Optimized C++/GCD CPU implementation.
- [x] **Phase 2**: Metal kernel for forward pass rasterization.
- [x] **Phase 3**: Metal kernel for backward pass (gradient computation).
- [ ] **Phase 4**: Move tile interaction (sorting) to GPU to eliminate remaining `mx.eval` calls.

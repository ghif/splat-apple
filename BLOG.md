# Benchmarking 3D Gaussian Splatting on Apple Silicon M4: MLX vs. PyTorch

The arrival of the Apple M4 chip has brought significant performance improvements to unified memory and GPU throughput on Mac. In this post, we benchmark a custom implementation of **3D Gaussian Splatting (3DGS)**, comparing the efficiency of **MLX** against **PyTorch (MPS)**, and measuring the impact of native C++ extensions.

We explore three main scenarios:
1. **Full Python Implementation**: Using high-level vectorized tensor operations in MLX and PyTorch.
2. **C++ Rasterizer Extension**: Offloading the core blending loop to native C++ for PyTorch.
3. **M4 Performance Scaling**: How these frameworks leverage the latest Apple Silicon hardware.

---

## 1. The Setup

Our benchmark uses the standard **Fern** scene from the LLFF dataset, processed at `images_8` resolution (504x378).
- **Hardware**: Apple M4 (10-core GPU, 16GB Unified Memory).
- **Dataset**: Fern (10,091 initial points).
- **Frameworks**: 
  - `mlx` (v0.21.0+)
  - `torch` (v2.5.0+ with MPS backend)

### Framework Implementations
We compare a **Vectorized Tile-Based Rasterizer** (Python) against a **Native C++ Rasterizer** (PyTorch Extension). Both frameworks utilize the M4's GPU, with the Python versions relying on high-level ops like `torch.cumprod` and `@mx.compile` for optimization.

---

## 2. Benchmark Results: The Latest Figures

Using the latest iteration of our benchmark scripts (`tests/benchmark_mlx_vs_torch.py` and `tests/test_benchmark_training.py`), we measured the average training iteration time (including forward, backward, and optimization).

| Framework | Rasterizer | Backend | Iterations/Sec | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | **Python** | MPS | ~3.35 it/s | 1.0x (Baseline) |
| **MLX** | **Python** | Native | ~4.17 it/s | 1.24x |
| **PyTorch** | **C++** | **MPS/CPU** | **~10.70 it/s** | **3.19x** |

### Analysis: MLX Edge in Python
In pure Python, **MLX holds a slight lead** (4.17 it/s vs 3.35 it/s). MLX’s `@mx.compile` effectively fuses kernels, giving it about a 24% advantage over PyTorch MPS in raw iteration speed. However, both frameworks perform admirably, proving that vectorized Python is a viable path for research and prototyping on Apple Silicon.

### The C++ Performance Leap
The real game-changer is the **PyTorch C++ Extension**. By offloading tile-based sorting and alpha-blending to a native C++ implementation, we achieved **10.70 it/s**—a significant **>3x speedup** over the base Python implementation.

Why C++ wins on M4:
1. **Sorting Efficiency**: C++ handles the primary (tile) and secondary (depth) sorting of 10k+ Gaussians with significantly lower overhead than high-level tensor sorting.
2. **Launch Latency**: The C++ extension reduces the number of individual GPU kernel dispatches, which is often a bottleneck on MPS for complex pipelines like Gaussian Splatting.
3. **Thread Parallelism**: Utilizing `at::parallel_for` for CPU-side management while the GPU processes tiles allows for better resource utilization across the M4's multi-core architecture.

---

## 3. Why the M4 Matters

The Apple M4’s unified memory architecture is uniquely suited for 3D Gaussian Splatting. The high bandwidth allows for rapid gathering of Gaussian parameters (means, covariances, colors) during the blending phase, while the shared memory space ensures that C++ extensions can access Python-allocated tensors with **zero-copy overhead**.

---

## 4. Conclusion

For AI development on Apple Silicon M4:
- **MLX** is the fastest pure-Python option, leveraging native graph compilation for a ~24% boost over PyTorch.
- **PyTorch with C++ Extensions** is the overall performance king, delivering over **10 iterations per second** on the Fern scene.

**Key Takeaways**:
- **Python Vectorization** is surprisingly fast on M4, making both MLX and PyTorch great for iterative research.
- **Native C++ Extensions** remain essential for achieving production-grade training performance.
- The **Apple M4** continues to push the boundaries of what's possible for 3D computer vision on mobile/desktop hardware.

---
*Updated February 18, 2026*

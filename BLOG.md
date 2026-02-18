# Benchmarking 3D Gaussian Splatting on Apple Silicon M4: MLX vs. PyTorch

The arrival of the Apple M4 chip has brought significant performance improvements to unified memory and GPU throughput on Mac. In this post, we benchmark a custom implementation of **3D Gaussian Splatting (3DGS)**, comparing the efficiency of **MLX** against **PyTorch (MPS)**, and measuring the impact of native C++ and Metal extensions.

We explore four main scenarios:
1. **Full Python Implementation**: Using high-level vectorized tensor operations in MLX and PyTorch.
2. **C++ CPU Extension**: A multi-threaded rasterizer using Apple's Grand Central Dispatch (GCD).
3. **Metal GPU Extension**: A fully GPU-resident rasterizer implemented in Metal Shading Language (MSL).
4. **M4 Performance Scaling**: How these frameworks leverage the latest Apple Silicon hardware.

---

## 1. The Setup

Our benchmark uses the standard **Fern** scene from the LLFF dataset, processed at `images_8` resolution (504x378).
- **Hardware**: Apple M4 (10-core GPU, 16GB Unified Memory).
- **Dataset**: Fern (10,091 initial points).
- **Frameworks**: 
  - `mlx` (v0.21.0+)
  - `torch` (v2.5.0+ with MPS backend)

### Framework Implementations
We compare a **Vectorized Tile-Based Rasterizer** (Python) against optimized native implementations. For MLX, we've unified our native optimizations into a single **C++/Metal GPU** rasterizer (accessible via `--rasterizer cpp`). For PyTorch, we use a specialized **C++ extension** that leverages MPS for most operations.

---

## 2. Benchmark Results: The Latest Figures

Using our updated benchmark suite (`tests/benchmark_mlx_vs_torch.py`), we measured the average training iteration time. The results below reflect **steady-state performance** (averaging iterations after the initial JIT/Metal compilation overhead).

| Framework | Rasterizer | Backend | Iterations/Sec | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | **Python** | MPS | ~3.25 it/s | 1.0x (Baseline) |
| **MLX** | **Python** | Native JIT | ~4.11 it/s | 1.26x |
| **MLX** | **C++/Metal** | **GPU (Unified)** | **~9.30 it/s** | **2.86x** |
| **PyTorch** | **C++** | **MPS/CPU** | **~10.23 it/s** | **3.15x** |

### Analysis: The Power of Unified Native Kernels
While MLX continues to lead in pure Python (1.26x faster than PyTorch), the unification of our native logic into the **Metal GPU Rasterizer** has brought MLX to within striking distance of the highly mature PyTorch C++ implementation. By achieving **9.30 it/s**, MLX + Metal delivers a nearly **3x boost** over the standard Python baseline and is functionally on-par with PyTorch C++ for production training workloads.

### The Hybrid Optimization Strategy
The high performance of the native extensions stems from three key optimizations:
1. **Zero-Copy Memory**: Using Apple Silicon's unified memory, our extensions (via `nanobind`) access MLX/PyTorch tensors directly without copying data between CPU and GPU.
2. **Metal Parallelism**: The unified MLX `cpp` rasterizer dispatches tile-based kernels that utilize the M4 GPU's massive thread count, handling the complex alpha-blending and gradient accumulation entirely on-device.
3. **Optimized Pre-sorting**: Critical bottlenecks like tile interaction and depth sorting are handled in optimized C++ before being passed to the GPU kernels, ensuring the GPU stays fed with work.

---

## 3. Why the M4 Matters

The Apple M4’s unified memory architecture is uniquely suited for 3D Gaussian Splatting. The high bandwidth allows for rapid gathering of Gaussian parameters (means, covariances, colors) during the blending phase. Our results show that when combined with native Metal kernels, the M4 can handle professional-grade 3D computer vision tasks with efficiency that rivals discrete workstations.

---

## 4. Conclusion

For AI development on Apple Silicon M4:
- **MLX + Metal** is now a top-tier performer, delivering a **3x boost** over standard Python and matching specialized PyTorch C++ speeds.
- **Pure MLX (Python)** remains the best option for rapid prototyping, offering the fastest out-of-the-box JIT performance.
- **PyTorch C++** remains exceptionally strong, benefiting from years of mature optimization on the MPS backend.

**Key Takeaways**:
- **Native Kernels are Essential**: To truly unlock the M4 GPU, moving critical loops into Metal/C++ is mandatory.
- **Unified Memory is the Secret Weapon**: The ability to share data between Python, C++, and Metal without overhead is why the Mac is becoming a powerhouse for 3D vision research.

---
*Updated February 18, 2026*

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

Using our updated benchmark suite (`tests/benchmark_mlx_vs_torch.py`), we measured the average training iteration time. The results below reflect **steady-state performance** after Phase 4 optimizations (fully GPU-resident interaction and rasterization).

| Framework | Rasterizer | Backend | Iterations/Sec | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | **Python** | MPS | ~3.31 it/s | 1.0x (Baseline) |
| **MLX** | **Python** | Native JIT | ~4.24 it/s | 1.28x |
| **PyTorch** | **C++** | **MPS/CPU** | **~10.64 it/s** | **3.21x** |
| **MLX** | **C++/Metal** | **GPU (Resident)** | **~36.38 it/s** | **10.99x** |

### Analysis: The Phase 4 Breakthrough
With the completion of **Phase 4**, the MLX pipeline is now fully GPU-resident. By moving the tile interaction and expansion stage (previously a major CPU-GPU sync bottleneck) entirely into vectorized GPU operations, we achieved a staggering **36.38 it/s**. 

This makes the MLX implementation **3.4x faster than specialized PyTorch C++ extensions** and **11x faster than standard Python pipelines**.

### The GPU-Resident Strategy
The key to this performance leap was the elimination of almost all CPU-GPU synchronization points:
1. **GPU Interaction Stage**: Gaussian expansion and tile-sorting are handled using pure MLX GPU primitives (`mx.cumsum`, `mx.argsort`), avoiding the need to sync 10k+ points to the CPU every iteration.
2. **Metal Rasterization**: The core alpha-blending and gradient accumulation remain in high-performance Metal kernels, accessing MLX arrays via zero-copy memory mapping.
3. **Lazy Execution**: The entire pipeline from projection to loss calculation is now managed by MLX's lazy execution engine, allowing for optimal scheduling on Apple Silicon.

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

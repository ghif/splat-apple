# Benchmarking 3D Gaussian Splatting on Apple Silicon M4: MLX vs. PyTorch

The arrival of the Apple M4 chip has brought significant performance improvements to unified memory and GPU throughput on Mac. In this post, we benchmark a custom implementation of **3D Gaussian Splatting (3DGS)**, comparing the efficiency of **MLX** against **PyTorch (MPS)**, and measuring the impact of native C++ and Metal extensions.

We explore four main scenarios:
1. **Full Python Implementation**: Using high-level vectorized tensor operations in MLX and PyTorch.
2. **C++ CPU Extension**: A multi-threaded rasterizer using Apple's Grand Central Dispatch (GCD).
3. **Metal GPU Extension**: A high-performance rasterizer implemented in Metal Shading Language (MSL).
4. **GPU-Resident Pipeline**: A fully optimized MLX pipeline that moves expansion and sorting entirely to the GPU.

---

## 1. The Setup

Our benchmark uses the standard **Fern** scene from the LLFF dataset, processed at `images_8` resolution (504x378).
- **Hardware**: Apple M4 (10-core GPU, 16GB Unified Memory).
- **Dataset**: Fern (10,091 initial points).
- **Frameworks**: 
  - `mlx` (v0.30.0+)
  - `torch` (v2.5.0+ with MPS backend)

### Framework Implementations
We compare a **Vectorized Tile-Based Rasterizer** (Python) against optimized native implementations. For MLX, we've unified our native optimizations into a single **C++/Metal GPU** rasterizer (accessible via `--rasterizer cpp`). For PyTorch, we use a specialized **C++ extension** that leverages MPS for most operations.

---

## 2. Benchmark Results: The Latest Figures

Using our updated benchmark suite (`tests/benchmark_mlx_vs_torch.py`), we measured the average training iteration time. The results below reflect **steady-state performance** (averaging iterations after the initial JIT/Metal compilation overhead).

| Framework | Rasterizer | Backend | Iterations/Sec | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | **Python** | MPS | ~3.27 it/s | 1.0x (Baseline) |
| **MLX** | **Python** | Native JIT | ~4.30 it/s | 1.31x |
| **PyTorch** | **C++** | **MPS/CPU** | **~9.47 it/s** | **2.89x** |
| **MLX** | **C++/Metal** | **GPU (Resident)** | **~42.34 it/s** | **12.95x** |

### Analysis: The GPU-Resident Breakthrough
With the completion of our **Phase 4** optimizations, the MLX pipeline is now **fully GPU-resident**. By moving the tile interaction and expansion stage—previously a major CPU-GPU synchronization bottleneck—entirely into vectorized GPU operations, we achieved a staggering **42.34 it/s**.

This makes the MLX implementation **4.4x faster than specialized PyTorch C++ extensions** and **13x faster than standard Python pipelines**.

### The Secret to 40+ IT/S
The performance leap comes from the elimination of almost all CPU-GPU synchronization points:
1. **GPU Interaction Stage**: Gaussian expansion and tile-sorting are handled using pure MLX GPU primitives (`mx.cumsum`, `mx.argsort`), avoiding the need to sync 10k+ points back to the CPU every iteration.
2. **Metal Rasterization**: The core alpha-blending and gradient accumulation remain in high-performance Metal kernels, accessing MLX arrays via **Zero-Copy** memory mapping on the Unified Memory Architecture.
3. **Lazy Execution**: The entire pipeline from projection to loss calculation is managed by MLX's lazy execution engine, allowing the M4 GPU to stay fully saturated.

---

## 3. Why the M4 Matters

The Apple M4’s unified memory architecture is uniquely suited for 3D Gaussian Splatting. The high bandwidth allows for rapid gathering of Gaussian parameters during the blending phase. Our results show that when critical logic is moved to the GPU, the M4 can handle professional-grade 3D computer vision tasks with efficiency that surpasses many discrete workstation setups.

---

## 4. Conclusion

For AI development on Apple Silicon M4:
- **MLX + Metal** is now the definitive performance leader, delivering a **13x boost** over standard Python and significantly outperforming specialized PyTorch extensions.
- **Unified Memory** is the secret weapon: The ability to share data between Python, C++, and Metal without overhead is why the Mac is becoming a powerhouse for 3D vision research.

---
*Updated February 18, 2026*

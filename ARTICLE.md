# Production-Grade 3D Gaussian Splatting on Apple Silicon with Native MPS Kernels

**3D Gaussian Splatting (3DGS)** has emerged as a transformative breakthrough in the field of 3D computer vision and radiance field reconstruction. By representing scenes as a collection of millions of differentiable 3D Gaussians, this technique enables unprecedented visual fidelity and real-time rendering speeds. However, the path to production-ready performance—the kind required for rapid training and high-frame-rate interaction—has historically been paved with NVIDIA GPUs and highly specialized CUDA kernels. This perceived hardware barrier has often relegated the Mac to a "viewer-only" role or a platform for slow prototyping.

This article challenges that narrative. I demonstrate how to achieve **production-grade performance** directly on Apple Silicon by leveraging the power of native C++ and Metal implementations. By utilizing the **Metal Performance Shaders (MPS)** backend and deep learning framework integration with **MLX**, the heavy lifting of the Gaussian Splatting pipeline can be ported to run natively on the Mac. 

It turns out that with the right optimization strategy—moving from high-level Python references to fully GPU-resident C++ Metal kernels—the Apple M4 hardware is not just a capable device, but a definitive powerhouse for 3D Gaussian Splatting. These optimizations deliver up to a **48x speedup**, bringing the training loop from several seconds per iteration to over **38 iterations per second**.

## Why C++ Native Kernels over Pure Python?

While high-level frameworks like MLX and PyTorch offer excellent vectorized operations, custom algorithms like 3D Gaussian Splatting hit several architectural limitations when implemented in pure Python. Moving to native C++ and Metal kernels is not just about raw speed; it is about achieving the architectural freedom required for production-quality results.

### 1. Dynamic Bounding and Tile Expansion
Python JIT compilers perform best when array shapes and loop bounds are predictable. To maintain performance and avoid excessive re-compilation, many pure Python implementations resort to a fixed-grid expansion, such as assuming a Gaussian touches at most an 8x8 tile area. This approach creates significant visual artifacts, as large or extremely close Gaussians are artificially "clipped" into squares at tile boundaries. 

In contrast, native C++ kernels calculate exact bounding boxes for every Gaussian dynamically. They can expand a single splat across the entire screen if necessary, ensuring perfect visual continuity with no performance penalty.

### 2. Elimination of Synchronization Gaps
High-level Python execution frequently requires "sync-points" where the CPU must wait for the GPU to complete a task—such as sorting Gaussians by depth—before it can schedule the next operation. This constant hand-off creates a "ping-pong" effect that leaves the GPU idle for significant portions of the training loop. 

By moving the entire rendering pipeline into a single, continuous sequence of Metal kernels, we eliminate these synchronization gaps. The GPU remains fully saturated throughout the process, which is the primary driver behind the leap from ~1 it/s to over 38 it/s.

### 3. Native Handling of Depth Complexity
Vectorized Python implementations typically process a fixed number of Gaussians per tile to keep memory usage constant for the compiler. However, dense real-world scenes often require blending hundreds or even thousands of Gaussians to reach full opacity. When this count is capped, it results in "black holes" where the background leaks through under-rendered regions. 

Metal kernels overcome this by using dynamic while-loops to iterate through the entire sorted interaction list. They only stop when the pixel transmittance is truly exhausted, providing flawless visual fidelity regardless of the scene's complexity.

### 4. Hardware-Native Gradient Accumulation
During the backward pass of training, gradients must be accumulated from millions of pixels back into the original Gaussian parameters. While Python's vectorized `scatter_add` is functional, it can be extremely memory-intensive at high resolutions. 

The native implementation utilizes thread-safe atomic operations (`atomic_add`) directly on the GPU hardware. This allows thousands of concurrent threads to update the same parameter buffers simultaneously with minimal overhead, providing a massively more efficient path for high-throughput backpropagation.

## Implementation: A Fully GPU-Resident MLX Metal Pipeline

Achieving 40+ iterations per second on a single M4 chip required a fundamental shift in how the rendering pipeline is structured. Instead of treating the GPU as a math accelerator for specific loops, we implemented a **Fully GPU-Resident Pipeline**. 

Here is the step-by-step breakdown of how the MLX Metal renderer is implemented:

### Step 1: Zero-Sync GPU Interaction
The interaction stage determines which Gaussians affect which 16x16 screen tiles. In traditional implementations, this often requires copying data back to the CPU for sorting. We eliminated this bottleneck using pure MLX GPU primitives to perform expansion and bitwise sorting on-device:

```python
# mlx_gs/renderer/rasterizer_metal.py

# Vectorized expansion using cumsum and marking masks
offsets = mx.cumsum(counts)
mark = mx.zeros((total,), dtype=mx.int32)
mark[active_starts] = 1
map_idx = mx.cumsum(mark) - 1
gaussian_ids = active_indices[map_idx]

# High-speed sorting using bitwise-packed 64-bit keys
keys = (tile_ids.astype(mx.uint64) << 32) | depth_quant
sort_indices = mx.argsort(keys)
```

### Step 2: The Host-Side Bridge (Nanobind)
To bridge MLX with native Metal, I developed an Objective-C++ extension using **Nanobind**. The bridge retrieves raw memory pointers and wraps them into Metal handles, leveraging Apple's Unified Memory for zero-copy access:

```cpp
// mlx_gs/csrc/rasterizer_metal.mm

id<MTLBuffer> wrap(void* ptr, size_t size) {
    // Zero-copy wrapping of Python/MLX pointers into Metal buffers
    return [device newBufferWithBytes:ptr length:size options:MTLResourceStorageModeShared];
}

// Dispatching the kernel with 16x16 tile threadgroups
[enc setComputePipelineState:forwardPSO];
[enc setBuffer:b_m offset:0 atIndex:0]; // Means
[enc dispatchThreads:MTLSizeMake(ntx*16, nty*16, 1) 
    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
...
/**
 * Nanobind Module Definition.
 */
NB_MODULE(_rasterizer_metal, m) {
    m.def("init_metal", &init_metal, "Initialize Metal device and compile shaders");
    m.def("render_forward", &render_forward, "Perform forward rasterization on Metal");
    m.def("render_backward", &render_backward, "Compute gradients for rasterization on Metal");
}
```

### Step 3: Tiled Forward Rasterization
The core rendering happens in a Metal compute shader. It implements a transmittance-aware blending loop with a dynamic early exit to maximize performance:

```cpp
// mlx_gs/csrc/rasterizer.metal

for (int i = s_idx; i < e_idx; ++i) {
    int g_idx = sorted_gaussian_ids[i];
    // ... compute Gaussian alpha ...
    float alpha = 1.0f - exp(-exp(power) * opacities[g_idx]);
    float test_T = T * (1.0f - alpha);
    
    // Early exit: Stop if the pixel is already opaque
    if (test_T < 0.0001f) break;
    
    C += color * alpha * T; 
    T = test_T;
}
```

### Step 4: Differentiable Backward Pass
To enable training, the renderer must be differentiable. The thread-safe atomic operations are used to accumulate gradients from millions of pixels back into Gaussian parameters simultaneously:

```cpp
// mlx_gs/csrc/rasterizer.metal

void atomic_add_float(device atomic_uint* addr, float val) {
    uint old_val = atomic_load_explicit(addr, memory_order_relaxed);
    uint new_val;
    do { 
        new_val = as_type<uint>(as_type<float>(old_val) + val); 
    } while (!atomic_compare_exchange_weak_explicit(addr, &old_val, new_val, ...));
}

// Accumulating gradients for means and colors
atomic_add_float(&grad_means2D[g_idx * 2 + 0], dL_dmean.x);
atomic_add_float(&grad_colors[g_idx * 3 + 0], dL_dPixel.x * w);
```

### Step 5: Autograd Integration via `mx.custom_function`
Finally, the native kernels are wrapped into MLX's autograd system using custom VJP (Vector-Jacobian Product) registrations:

```python
# mlx_gs/renderer/rasterizer_metal.py

@mx.custom_function
def forward(m, ic, s_o, c, sti, sgi, bg):
    # Call the C++ extension
    out_np = rasterizer_metal.render_forward(...)
    return mx.array(out_np)

@forward.vjp
def backward(primals, cotangents, outputs):
    # Pass gradients back through the Metal backward kernel
    gm_np, gic_np, go_np, gc_np = rasterizer_metal.render_backward(...)
    return mx.array(gm_np), mx.array(gic_np), mx.array(go_np), ...
```

## The PyTorch Challenge: Why CPU-GCD was the Choice for Stability
Besides MLX, we could also use PyTorch with Python-C++ bindings to speed-up 3D Gaussian Splatting. Therefore, I also attempted to implement a similar "Zero-Sync" Metal extension for PyTorch on MPS. 

However, I encountered several architectural hurdles. These challenges highlight the fundamental differences between frameworks designed for unified memory (MLX) and those with more complex, cross-platform backends (PyTorch).

### 1. Stream Synchronization and Mutex Deadlocks
PyTorch's MPS backend manages a global command buffer and internal hardware state using a strict `recursive_mutex`. Attempting to dispatch manual Metal kernels on the same device often triggered internal deadlocks, resulting in `recursive_mutex lock failed` errors. Because PyTorch does not officially export its command buffer management APIs for third-party extensions, there is currently no stable way to safely interleave custom Metal kernels with PyTorch's native operations without risking a "race to the hardware."

### 2. Symbol Visibility and Portability
Achieving zero-copy performance requires accessing the underlying `id<MTLBuffer>` of an `at::Tensor`. While APIs like `getMTLBuffer_untracked` exist, they are not consistently exported across different PyTorch versions or installation environments. My experiments with runtime symbol lookup (`dlsym`) proved fragile, leading to "Symbol not found" errors on standard user installs. This makes a production-grade GPU-resident extension for PyTorch difficult to distribute reliably.

### 3. Memory Locality and Race Conditions
In my experimental Metal version for PyTorch, I observed significant rendering artifacts, such as massive black regions. These were traced to race conditions between PyTorch's asynchronous memory allocator and my manual command queue. Without deep integration into the `MPSAllocator`, Gaussian data was often outdated or "missing" by the time the Metal kernel executed.

**The Result**: For the PyTorch implementation, I opted for a high-performance **multi-threaded C++ implementation using Apple's Grand Central Dispatch (GCD)**. By distributing tile-based rendering across all available CPU cores, I achieved a reliable **10.6 it/s**—a 13x boost over the Python reference—while maintaining 100% visual accuracy and backend stability.

## Benchmark Analysis: The Performance Landscape

To evaluate the impact of these native optimizations, I benchmarked the training loop using the standard **Fern** dataset (10,091 Gaussians) at a resolution of 504x378 on an **Apple M4 (10-core GPU, 16GB Unified Memory)**.

| Framework | Implementation | Backend | Iterations/Sec | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | Python (Reference) | MPS (Vectorized) | ~0.84 it/s | 1.0x |
| **MLX** | Python | Native JIT | ~1.15 it/s | 1.3x |
| **PyTorch** | **Native C++** | **CPU (GCD)** | **~10.60 it/s** | **12.6x** |
| **MLX** | **Native Metal** | **GPU-Resident** | **~38.50 it/s** | **45.8x** |

### Insights: The Strategy Behind the Numbers

The benchmarking results reveal three critical insights into the current state of 3D computer vision on Apple Silicon:

#### A. MLX and the Power of GPU-Residency
The leap from ~1 it/s in pure Python to **38.5 it/s** in the MLX Metal mode represents a complete paradigm shift. This was made possible by moving the *entire* pipeline—including coordinate projection, tile interaction, and rasterization—into GPU-resident kernels. By utilizing MLX's native support for Apple's Unified Memory Architecture (UMA), the Metal kernels access Gaussian parameters at the hardware's full bandwidth, effectively treating the M4 GPU as a monolithic high-speed processor rather than an external accelerator.

#### B. PyTorch: Balancing Performance and Stability
For the PyTorch implementation, the results highlight a different but equally important optimization path. While I explored an experimental GPU-resident version for PyTorch, the multi-threaded **C++ CPU rasterizer** (leveraging Apple's Grand Central Dispatch) proved to be the superior choice for production stability. It delivers a respectable **10.6 it/s**—a 12x boost over the Python reference—while ensuring 100% visual accuracy and avoiding the stream synchronization deadlocks often encountered in manual Metal-PyTorch interop.

#### C. The Visual Parity Trade-off
Another interesting insight from this experiment was bridging the quality gap between the high-level Python references and the low-level native kernels. Initially, the Python versions suffered from "square clipping" and depth-related artifacts. 
While I successfully achieved **full visual parity** by implementing dynamic tile expansion and increasing depth complexity (1024 Gaussians per tile) in pure Python, this came with a considerable performance cost. 

To maintain this high-fidelity output without custom kernels, the Python implementation must materialize and process massive vectorized tensors, which explains why it remains capped at ~1 it/s. 
This results in a clear hierarchy: the Python implementation serves as a high-quality reference for research and debugging, while the native C++ and Metal kernels provide the same production-grade quality at 10x to 40x the speed.

## Conclusion

The transition from high-level Python references to native Metal extensions has fundamentally redefined what is possible for 3D computer vision on Apple Silicon. For years, production-grade 3D Gaussian Splatting was considered synonymous with NVIDIA hardware and CUDA kernels. This simple project demonstrates the possibility of leveraging Apple Silicon as part of the "first-class citizens" in the production-grade 3D AI landscape.

By combining the **MLX** framework with a fully **GPU-resident Metal pipeline**, we can deliver a solution that bridges the gap between prototyping and production. With a steady-state performance of over **38 iterations per second**, the Mac is no longer a platform restricted to slow experimentation or viewing-only roles. 
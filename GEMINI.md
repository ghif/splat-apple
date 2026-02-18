# Project Instructions
- Always use the conda environment `gs-mps` for this project.

# C++ Rasterizer Fixes (2026-02-17)
The C++ rasterizer (`cpp_gs`) has been fixed and verified to be correct and performant (~3x faster than Python).

### Key Fixes Implemented:
1.  **Differentiable Covariance Matrix (`grad_cov2D`)**:
    -   Previously, the rasterizer did not compute gradients for the 2D covariance matrix (`cov2D`).
    -   This caused Gaussians to be "frozen" in size and rotation, leading to small blobs and black spaces in the rendered image.
    -   **Fix**: Implemented full backpropagation for `cov2D` by differentiating through the inverse covariance matrix computation in `rasterizer.cpp`.

2.  **Environment Mismatch Resolution**:
    -   Encountered `AttributeError: module 'cpp_gs' has no attribute...` due to the extension being built with the system Python (3.11) but run in the `gs-mps` environment (Python 3.13).
    -   **Fix**: Re-built the extension explicitly using the python executable from the `gs-mps` environment.

3.  **Atomic Gradient Accumulation**:
    -   Added thread-safe gradient accumulation using atomic operations (`atomic_add`) for `means2D`, `opacities`, `colors`, and `cov2D` to correctly handle overlapping Gaussians during backpropagation.

### Verification Status
-   **Correctness**: Validated against Python implementation; produces visually growing and densifying Gaussians.
-   **Performance**: ~100ms per iteration (vs ~300ms for Python) on M1/M2/M3 chips.
-   **Usage**: Use `--rasterizer cpp` flag with training scripts.

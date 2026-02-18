#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <Metal/Metal.h>
#include <iostream>

namespace nb = nanobind;

// Global Metal State
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLComputePipelineState> forwardPSO = nil;
static id<MTLComputePipelineState> backwardPSO = nil;

void init_metal(const std::string& source) {
    if (device != nil) return; // Already initialized

    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Failed to find Metal device");
    }
    commandQueue = [device newCommandQueue];
    if (!commandQueue) {
        throw std::runtime_error("Failed to create Command Queue");
    }

    NSError* error = nil;
    NSString* src = [NSString stringWithUTF8String:source.c_str()];
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.fastMathEnabled = YES;
    
    id<MTLLibrary> library = [device newLibraryWithSource:src options:options error:&error];
    if (!library) {
        std::string errStr = error ? [[error localizedDescription] UTF8String] : "Unknown error";
        throw std::runtime_error("Failed to compile Metal library: " + errStr);
    }
    
    id<MTLFunction> forwardFn = [library newFunctionWithName:@"render_tiles_forward"];
    if (!forwardFn) throw std::runtime_error("Function render_tiles_forward not found");
    forwardPSO = [device newComputePipelineStateWithFunction:forwardFn error:&error];
    if (!forwardPSO) throw std::runtime_error("Failed to create forward PSO");
    
    id<MTLFunction> backwardFn = [library newFunctionWithName:@"render_tiles_backward"];
    if (!backwardFn) throw std::runtime_error("Function render_tiles_backward not found");
    backwardPSO = [device newComputePipelineStateWithFunction:backwardFn error:&error];
    if (!backwardPSO) throw std::runtime_error("Failed to create backward PSO");
}

// Helper to wrap pointer in MTLBuffer without copy
id<MTLBuffer> wrap_buffer(void* ptr, size_t size) {
    // Note: This requires the pointer to be page-aligned (usually 4096 bytes) 
    // and allocated in a way compatible with Metal.
    // MLX (and malloc on macOS) usually satisfies this for large allocations.
    // If not, we might need to copy. For now, assume zero-copy works or fallback (which would crash/error).
    return [device newBufferWithBytesNoCopy:ptr length:size options:MTLResourceStorageModeShared deallocator:nil];
}

nb::ndarray<float, nb::numpy> render_forward(
    nb::ndarray<float, nb::ndim<2>> means2D,
    nb::ndarray<float, nb::ndim<3>> inv_cov2D,
    nb::ndarray<float, nb::ndim<2>> sig_opacities,
    nb::ndarray<float, nb::ndim<2>> colors,
    nb::ndarray<int32_t, nb::ndim<1>> sorted_tile_ids,
    nb::ndarray<int32_t, nb::ndim<1>> sorted_gaussian_ids,
    int H, int W, int tile_size,
    nb::ndarray<float, nb::ndim<1>> background,
    nb::ndarray<int32_t, nb::ndim<1>> tile_boundaries
) {
    if (!forwardPSO) throw std::runtime_error("Metal not initialized");

    int num_tiles_x = (W + tile_size - 1) / tile_size;
    int num_tiles_y = (H + tile_size - 1) / tile_size;
    int num_tiles = num_tiles_x * num_tiles_y;
    int num_interactions = (int)sorted_tile_ids.shape(0);

    // Output buffer
    size_t out_size = H * W * 3 * sizeof(float);
    // Allocate using malloc to ensure page alignment? Or let nb::ndarray handle it?
    // We return a numpy array (which creates python object).
    // We need to allocate memory that we can pass to Metal.
    // Let's alloc with new, wrap, then pass ownership to numpy?
    // nb::ndarray can take a pointer.
    
    // Better: Allocate via standard allocator, Metal wraps it.
    float* out_ptr = new float[H * W * 3];
    // Initialize to zero? Kernel overwrites? No, kernel accumulates background.
    // Kernel writes pixel.
    
    // Create buffers
    id<MTLBuffer> b_means = wrap_buffer(means2D.data(), means2D.size() * sizeof(float));
    id<MTLBuffer> b_cov = wrap_buffer(inv_cov2D.data(), inv_cov2D.size() * sizeof(float));
    id<MTLBuffer> b_opac = wrap_buffer(sig_opacities.data(), sig_opacities.size() * sizeof(float));
    id<MTLBuffer> b_cols = wrap_buffer(colors.data(), colors.size() * sizeof(float));
    id<MTLBuffer> b_tids = wrap_buffer(sorted_tile_ids.data(), sorted_tile_ids.size() * sizeof(int32_t));
    id<MTLBuffer> b_gids = wrap_buffer(sorted_gaussian_ids.data(), sorted_gaussian_ids.size() * sizeof(int32_t));
    id<MTLBuffer> b_bg = wrap_buffer(background.data(), background.size() * sizeof(float));
    id<MTLBuffer> b_out = wrap_buffer(out_ptr, out_size);
    id<MTLBuffer> b_bounds = wrap_buffer(tile_boundaries.data(), tile_boundaries.size() * sizeof(int32_t));

    id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:forwardPSO];

    [enc setBuffer:b_means offset:0 atIndex:0];
    [enc setBuffer:b_cov offset:0 atIndex:1];
    [enc setBuffer:b_opac offset:0 atIndex:2];
    [enc setBuffer:b_cols offset:0 atIndex:3];
    [enc setBuffer:b_tids offset:0 atIndex:4];
    [enc setBuffer:b_gids offset:0 atIndex:5];
    [enc setBuffer:b_bg offset:0 atIndex:6];
    [enc setBuffer:b_out offset:0 atIndex:7];
    [enc setBytes:&H length:sizeof(int) atIndex:8];
    [enc setBytes:&W length:sizeof(int) atIndex:9];
    [enc setBytes:&tile_size length:sizeof(int) atIndex:10];
    [enc setBytes:&num_interactions length:sizeof(int) atIndex:11];
    [enc setBuffer:b_bounds offset:0 atIndex:12];

    MTLSize threadsPerGroup = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake(num_tiles_x * 16, num_tiles_y * 16, 1);
    
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // Return ndarray
    nb::capsule free_out(out_ptr, [](void *p) noexcept { delete[] (float *)p; });
    return nb::ndarray<float, nb::numpy>(out_ptr, {(size_t)H, (size_t)W, 3}, free_out);
}

nb::tuple render_backward(
    nb::ndarray<float, nb::ndim<3>> grad_output,
    nb::ndarray<float, nb::ndim<2>> means2D,
    nb::ndarray<float, nb::ndim<3>> inv_cov2D,
    nb::ndarray<float, nb::ndim<2>> sig_opacities,
    nb::ndarray<float, nb::ndim<2>> colors,
    nb::ndarray<int32_t, nb::ndim<1>> sorted_tile_ids,
    nb::ndarray<int32_t, nb::ndim<1>> sorted_gaussian_ids,
    int H, int W, int tile_size,
    nb::ndarray<float, nb::ndim<1>> background,
    nb::ndarray<int32_t, nb::ndim<1>> tile_boundaries
) {
    if (!backwardPSO) throw std::runtime_error("Metal not initialized");

    int num_points = (int)means2D.shape(0);
    
    // Allocate output gradients (initialized to 0)
    // We use calloc for zero init
    float* gm_ptr = new float[num_points * 2]();
    float* gic_ptr = new float[num_points * 4]();
    float* go_ptr = new float[num_points]();
    float* gcol_ptr = new float[num_points * 3]();
    
    id<MTLBuffer> b_grad_out = wrap_buffer(grad_output.data(), grad_output.size() * sizeof(float));
    id<MTLBuffer> b_means = wrap_buffer(means2D.data(), means2D.size() * sizeof(float));
    id<MTLBuffer> b_cov = wrap_buffer(inv_cov2D.data(), inv_cov2D.size() * sizeof(float));
    id<MTLBuffer> b_opac = wrap_buffer(sig_opacities.data(), sig_opacities.size() * sizeof(float));
    id<MTLBuffer> b_cols = wrap_buffer(colors.data(), colors.size() * sizeof(float));
    id<MTLBuffer> b_tids = wrap_buffer(sorted_tile_ids.data(), sorted_tile_ids.size() * sizeof(int32_t));
    id<MTLBuffer> b_gids = wrap_buffer(sorted_gaussian_ids.data(), sorted_gaussian_ids.size() * sizeof(int32_t));
    id<MTLBuffer> b_bg = wrap_buffer(background.data(), background.size() * sizeof(float));
    
    id<MTLBuffer> b_gm = wrap_buffer(gm_ptr, num_points * 2 * sizeof(float));
    id<MTLBuffer> b_gic = wrap_buffer(gic_ptr, num_points * 4 * sizeof(float));
    id<MTLBuffer> b_go = wrap_buffer(go_ptr, num_points * sizeof(float));
    id<MTLBuffer> b_gcol = wrap_buffer(gcol_ptr, num_points * 3 * sizeof(float));
    id<MTLBuffer> b_bounds = wrap_buffer(tile_boundaries.data(), tile_boundaries.size() * sizeof(int32_t));

    id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:backwardPSO];

    [enc setBuffer:b_grad_out offset:0 atIndex:0];
    [enc setBuffer:b_means offset:0 atIndex:1];
    [enc setBuffer:b_cov offset:0 atIndex:2];
    [enc setBuffer:b_opac offset:0 atIndex:3];
    [enc setBuffer:b_cols offset:0 atIndex:4];
    [enc setBuffer:b_tids offset:0 atIndex:5];
    [enc setBuffer:b_gids offset:0 atIndex:6];
    [enc setBuffer:b_bg offset:0 atIndex:7];
    [enc setBuffer:b_gm offset:0 atIndex:8];
    [enc setBuffer:b_gic offset:0 atIndex:9];
    [enc setBuffer:b_go offset:0 atIndex:10];
    [enc setBuffer:b_gcol offset:0 atIndex:11];
    [enc setBytes:&H length:sizeof(int) atIndex:12];
    [enc setBytes:&W length:sizeof(int) atIndex:13];
    [enc setBytes:&tile_size length:sizeof(int) atIndex:14];
    [enc setBuffer:b_bounds offset:0 atIndex:15];

    int num_tiles_x = (W + tile_size - 1) / tile_size;
    int num_tiles_y = (H + tile_size - 1) / tile_size;
    
    MTLSize threadsPerGroup = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake(num_tiles_x * 16, num_tiles_y * 16, 1);
    
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    nb::capsule free_gm(gm_ptr, [](void *p) noexcept { delete[] (float *)p; });
    nb::capsule free_gic(gic_ptr, [](void *p) noexcept { delete[] (float *)p; });
    nb::capsule free_go(go_ptr, [](void *p) noexcept { delete[] (float *)p; });
    nb::capsule free_gcol(gcol_ptr, [](void *p) noexcept { delete[] (float *)p; });

    return nb::make_tuple(
        nb::ndarray<float, nb::numpy>(gm_ptr, {(size_t)num_points, 2}, free_gm),
        nb::ndarray<float, nb::numpy>(gic_ptr, {(size_t)num_points, 2, 2}, free_gic),
        nb::ndarray<float, nb::numpy>(go_ptr, {(size_t)num_points}, free_go),
        nb::ndarray<float, nb::numpy>(gcol_ptr, {(size_t)num_points, 3}, free_gcol)
    );
}

NB_MODULE(_rasterizer_metal, m) {
    m.def("init_metal", &init_metal);
    m.def("render_forward", &render_forward);
    m.def("render_backward", &render_backward);
}

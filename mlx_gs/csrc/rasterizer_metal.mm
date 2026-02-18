#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <Metal/Metal.h>
#include <iostream>

namespace nb = nanobind;

/**
 * Global Metal State.
 * Using static handles to avoid re-initializing the Metal device and pipeline states
 * on every function call.
 */
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLComputePipelineState> forwardPSO = nil;
static id<MTLComputePipelineState> backwardPSO = nil;

/**
 * Initialize Metal Device and Pipeline States.
 * This is called once from Python when the rasterizer_metal module is loaded.
 */
void init_metal(const std::string& source) {
    if (device != nil) return; // Already initialized

    device = MTLCreateSystemDefaultDevice();
    commandQueue = [device newCommandQueue];
    
    NSError* error = nil;
    NSString* src = [NSString stringWithUTF8String:source.c_str()];
    
    // Enable fast math for performance optimization
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.fastMathEnabled = YES;
    
    // Compile Metal source code into a library
    id<MTLLibrary> library = [device newLibraryWithSource:src options:options error:&error];
    if (!library) {
        std::string errStr = error ? [[error localizedDescription] UTF8String] : "Unknown error";
        throw std::runtime_error("Metal compile failed: " + errStr);
    }
    
    // Helper to create compute pipeline states from function names
    auto create = [&](NSString* name) {
        id<MTLFunction> fn = [library newFunctionWithName:name];
        return [device newComputePipelineStateWithFunction:fn error:nil];
    };
    
    // Initialize PSOs for forward and backward passes
    forwardPSO = create(@"render_tiles_forward");
    backwardPSO = create(@"render_tiles_backward");
}

/**
 * Helper to wrap a raw pointer into a Metal buffer.
 * On Apple Silicon, this is zero-copy if the memory is appropriately aligned.
 */
id<MTLBuffer> wrap(void* ptr, size_t size) {
    if (size == 0) return [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    return [device newBufferWithBytes:ptr length:size options:MTLResourceStorageModeShared];
}

/**
 * Bridge for Forward Rasterization.
 * Takes NumPy arrays (from MLX) and dispatches the Metal kernel.
 */
nb::ndarray<float, nb::numpy> render_forward(
    nb::ndarray<float, nb::ndim<2>> m, nb::ndarray<float, nb::ndim<3>> ic,
    nb::ndarray<float, nb::ndim<2>> so, nb::ndarray<float, nb::ndim<2>> c,
    nb::ndarray<int32_t, nb::ndim<1>> sti, nb::ndarray<int32_t, nb::ndim<1>> sgi,
    int H, int W, int ts, nb::ndarray<float, nb::ndim<1>> bg, nb::ndarray<int32_t, nb::ndim<1>> tb
) {
    int ntx = (W+ts-1)/ts, nty = (H+ts-1)/ts, ni = (int)sti.shape(0);
    
    // 1. Map input arrays into Metal buffers
    id<MTLBuffer> b_m = wrap(m.data(), m.size()*4); 
    id<MTLBuffer> b_ic = wrap(ic.data(), ic.size()*4);
    id<MTLBuffer> b_so = wrap(so.data(), so.size()*4); 
    id<MTLBuffer> b_c = wrap(c.data(), c.size()*4);
    id<MTLBuffer> b_sti = wrap(sti.data(), sti.size()*4); 
    id<MTLBuffer> b_sgi = wrap(sgi.data(), sgi.size()*4);
    id<MTLBuffer> b_bg = wrap(bg.data(), bg.size()*4); 
    id<MTLBuffer> b_tb = wrap(tb.data(), tb.size()*4);
    
    // 2. Allocate output buffer
    id<MTLBuffer> b_out = [device newBufferWithLength:H*W*3*4 options:MTLResourceStorageModeShared];
    
    // 3. Dispatch the compute kernel
    id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:forwardPSO];
    
    // Bind all buffers to their indices defined in the .metal file
    [enc setBuffer:b_m offset:0 atIndex:0]; 
    [enc setBuffer:b_ic offset:0 atIndex:1];
    [enc setBuffer:b_so offset:0 atIndex:2]; 
    [enc setBuffer:b_c offset:0 atIndex:3];
    [enc setBuffer:b_sti offset:0 atIndex:4]; 
    [enc setBuffer:b_sgi offset:0 atIndex:5];
    [enc setBuffer:b_bg offset:0 atIndex:6]; 
    [enc setBuffer:b_out offset:0 atIndex:7];
    
    // Bind constants
    [enc setBytes:&H length:4 atIndex:8]; 
    [enc setBytes:&W length:4 atIndex:9];
    [enc setBytes:&ts length:4 atIndex:10]; 
    [enc setBytes:&ni length:4 atIndex:11];
    [enc setBuffer:b_tb offset:0 atIndex:12];
    
    // Define thread topology: each tile is a threadgroup of 16x16 threads
    [enc dispatchThreads:MTLSizeMake(ntx*16, nty*16, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    
    [enc endEncoding]; 
    [cmd commit]; 
    [cmd waitUntilCompleted]; // Mandatory synchronization for this CPU-Metal bridge
    
    // 4. Cleanup temporary wrappers
    [b_m release]; [b_ic release]; [b_so release]; [b_c release]; 
    [b_sti release]; [b_sgi release]; [b_bg release]; [b_tb release];
    
    // 5. Transfer output to Python-owned memory (via Nanobind)
    float* p = new float[H*W*3]; 
    memcpy(p, [b_out contents], H*W*3*4); 
    [b_out release];
    
    // Wrap pointer in a capsule to ensure it's deleted when the Python object is GC'd
    nb::capsule cap(p, [](void* p) noexcept { delete[] (float*)p; });
    return nb::ndarray<float, nb::numpy>(p, {(size_t)H, (size_t)W, 3}, cap);
}

/**
 * Bridge for Backward Pass (Gradients).
 */
nb::tuple render_backward(
    nb::ndarray<float, nb::ndim<3>> go, nb::ndarray<float, nb::ndim<2>> m,
    nb::ndarray<float, nb::ndim<3>> ic, nb::ndarray<float, nb::ndim<2>> so,
    nb::ndarray<float, nb::ndim<2>> c, nb::ndarray<int32_t, nb::ndim<1>> sti,
    nb::ndarray<int32_t, nb::ndim<1>> sgi, int H, int W, int ts,
    nb::ndarray<float, nb::ndim<1>> bg, nb::ndarray<int32_t, nb::ndim<1>> tb
) {
    int np = (int)m.shape(0);
    
    // 1. Allocate and ZERO out gradient buffers
    // Gradient accumulation on GPU REQUIRES starting from zero.
    id<MTLBuffer> b_gm = [device newBufferWithLength:np*2*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_gic = [device newBufferWithLength:np*4*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_gso = [device newBufferWithLength:np*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_gc = [device newBufferWithLength:np*3*4 options:MTLResourceStorageModeShared];
    
    memset([b_gm contents], 0, np*2*4); 
    memset([b_gic contents], 0, np*4*4);
    memset([b_gso contents], 0, np*4); 
    memset([b_gc contents], 0, np*3*4);
    
    // 2. Wrap inputs
    id<MTLBuffer> b_go = wrap(go.data(), go.size()*4); 
    id<MTLBuffer> b_m = wrap(m.data(), m.size()*4);
    id<MTLBuffer> b_ic = wrap(ic.data(), ic.size()*4); 
    id<MTLBuffer> b_so = wrap(so.data(), so.size()*4);
    id<MTLBuffer> b_c = wrap(c.data(), c.size()*4); 
    id<MTLBuffer> b_sti = wrap(sti.data(), sti.size()*4);
    id<MTLBuffer> b_sgi = wrap(sgi.data(), sgi.size()*4); 
    id<MTLBuffer> b_bg = wrap(bg.data(), bg.size()*4);
    id<MTLBuffer> b_tb = wrap(tb.data(), tb.size()*4);
    
    // 3. Dispatch gradient kernel
    id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:backwardPSO];
    
    // Input bindings
    [enc setBuffer:b_go offset:0 atIndex:0]; 
    [enc setBuffer:b_m offset:0 atIndex:1];
    [enc setBuffer:b_ic offset:0 atIndex:2]; 
    [enc setBuffer:b_so offset:0 atIndex:3];
    [enc setBuffer:b_c offset:0 atIndex:4]; 
    [enc setBuffer:b_sti offset:0 atIndex:5];
    [enc setBuffer:b_sgi offset:0 atIndex:6]; 
    [enc setBuffer:b_bg offset:0 atIndex:7];
    
    // Output bindings (Gradients)
    [enc setBuffer:b_gm offset:0 atIndex:8]; 
    [enc setBuffer:b_gic offset:0 atIndex:9];
    [enc setBuffer:b_gso offset:0 atIndex:10]; 
    [enc setBuffer:b_gc offset:0 atIndex:11];
    
    // Constants
    [enc setBytes:&H length:4 atIndex:12]; 
    [enc setBytes:&W length:4 atIndex:13];
    [enc setBytes:&ts length:4 atIndex:14]; 
    [enc setBuffer:b_tb offset:0 atIndex:15];
    
    [enc dispatchThreads:MTLSizeMake(((W+ts-1)/ts)*16, ((H+ts-1)/ts)*16, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    
    [enc endEncoding]; 
    [cmd commit]; 
    [cmd waitUntilCompleted];
    
    // 4. Cleanup temporary wrappers
    [b_go release]; [b_m release]; [b_ic release]; [b_so release]; [b_c release]; 
    [b_sti release]; [b_sgi release]; [b_bg release]; [b_tb release];
    
    // 5. Transfer gradients back to Python
    auto mk = [](float* p, size_t s1, size_t s2=0, size_t s3=0) {
        nb::capsule cap(p, [](void* p) noexcept { delete[] (float*)p; });
        if (s3 > 0) return nb::ndarray<float, nb::numpy>(p, {s1, s2, s3}, cap);
        if (s2 > 0) return nb::ndarray<float, nb::numpy>(p, {s1, s2}, cap);
        return nb::ndarray<float, nb::numpy>(p, {s1}, cap);
    };
    
    float* gm_p = new float[np*2]; memcpy(gm_p, [b_gm contents], np*2*4); [b_gm release];
    float* gic_p = new float[np*4]; memcpy(gic_p, [b_gic contents], np*4*4); [b_gic release];
    float* gso_p = new float[np]; memcpy(gso_p, [b_gso contents], np*4); [b_gso release];
    float* gc_p = new float[np*3]; memcpy(gc_p, [b_gc contents], np*3*4); [b_gc release];
    
    return nb::make_tuple(mk(gm_p, np, 2), mk(gic_p, np, 2, 2), mk(gso_p, np), mk(gc_p, np, 3));
}

/**
 * Nanobind Module Definition.
 */
NB_MODULE(_rasterizer_metal, m) {
    m.def("init_metal", &init_metal, "Initialize Metal device and compile shaders");
    m.def("render_forward", &render_forward, "Perform forward rasterization on Metal");
    m.def("render_backward", &render_backward, "Compute gradients for rasterization on Metal");
}

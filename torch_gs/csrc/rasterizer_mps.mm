#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <Metal/Metal.h>
#include <iostream>
#include <vector>

namespace nb = nanobind;

static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLComputePipelineState> forwardPSO = nil;
static id<MTLComputePipelineState> backwardPSO = nil;

void init_mps(const std::string& source) {
    if (device != nil) return;
    device = MTLCreateSystemDefaultDevice();
    commandQueue = [device newCommandQueue];
    NSError* error = nil;
    NSString* src = [NSString stringWithUTF8String:source.c_str()];
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.fastMathEnabled = YES;
    id<MTLLibrary> library = [device newLibraryWithSource:src options:options error:&error];
    if (!library) throw std::runtime_error("Metal compile failed");
    auto create = [&](NSString* name) {
        id<MTLFunction> fn = [library newFunctionWithName:name];
        return [device newComputePipelineStateWithFunction:fn error:nil];
    };
    forwardPSO = create(@"render_tiles_forward");
    backwardPSO = create(@"render_tiles_backward");
}

id<MTLBuffer> wrap_ptr(size_t ptr, size_t size, std::vector<id<MTLBuffer>>& tracker) {
    if (size == 0) size = 16;
    // We create a new buffer that wraps the existing memory.
    // On Apple Silicon, this is zero-copy if aligned to 4096.
    // Since we're using .cpu().contiguous() in Python, we're passing CPU pointers.
    id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:(void*)ptr length:size options:MTLResourceStorageModeShared deallocator:nil];
    if (!buf) {
        // Fallback to copy if alignment fails
        buf = [device newBufferWithBytes:(void*)ptr length:size options:MTLResourceStorageModeShared];
    }
    tracker.push_back(buf);
    return buf;
}

void render_forward_mps(
    size_t m, size_t ic, size_t so, size_t c,
    size_t sti, size_t sgi, size_t bg, size_t out,
    int H, int W, int ts, int ni, size_t tb,
    size_t sz_m, size_t sz_ic, size_t sz_so, size_t sz_c,
    size_t sz_sti, size_t sz_sgi, size_t sz_bg, size_t sz_out, size_t sz_tb
) {
    @autoreleasepool {
        std::vector<id<MTLBuffer>> tracker;
        id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:forwardPSO];
        
        // Use offset 0 for ALL because wrap_ptr already points to the start of data
        [enc setBuffer:wrap_ptr(m, sz_m, tracker) offset:0 atIndex:0];
        [enc setBuffer:wrap_ptr(ic, sz_ic, tracker) offset:0 atIndex:1];
        [enc setBuffer:wrap_ptr(so, sz_so, tracker) offset:0 atIndex:2];
        [enc setBuffer:wrap_ptr(c, sz_c, tracker) offset:0 atIndex:3];
        [enc setBuffer:wrap_ptr(sti, sz_sti, tracker) offset:0 atIndex:4];
        [enc setBuffer:wrap_ptr(sgi, sz_sgi, tracker) offset:0 atIndex:5];
        [enc setBuffer:wrap_ptr(bg, sz_bg, tracker) offset:0 atIndex:6];
        
        id<MTLBuffer> b_out = [device newBufferWithBytesNoCopy:(void*)out length:sz_out options:MTLResourceStorageModeShared deallocator:nil];
        bool out_copied = false;
        if (!b_out) {
            b_out = [device newBufferWithLength:sz_out options:MTLResourceStorageModeShared];
            out_copied = true;
        }
        [enc setBuffer:b_out offset:0 atIndex:7];
        
        [enc setBytes:&H length:4 atIndex:8];
        [enc setBytes:&W length:4 atIndex:9];
        [enc setBytes:&ts length:4 atIndex:10];
        [enc setBytes:&ni length:4 atIndex:11];
        [enc setBuffer:wrap_ptr(tb, sz_tb, tracker) offset:0 atIndex:12];
        
        [enc dispatchThreads:MTLSizeMake(((W+ts-1)/ts)*16, ((H+ts-1)/ts)*16, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        
        if (out_copied) {
            memcpy((void*)out, [b_out contents], sz_out);
        }
        [b_out release];
        for (auto b : tracker) [b release];
    }
}

void render_backward_mps(
    size_t go, size_t m, size_t ic, size_t so, size_t c,
    size_t sti, size_t sgi, size_t bg,
    size_t gm, size_t gic, size_t gso, size_t gc,
    int H, int W, int ts, size_t tb,
    size_t sz_go, size_t sz_m, size_t sz_ic, size_t sz_so, size_t sz_c,
    size_t sz_sti, size_t sz_sgi, size_t sz_bg,
    size_t sz_gm, size_t sz_gic, size_t sz_gso, size_t sz_gc, size_t sz_tb
) {
    @autoreleasepool {
        std::vector<id<MTLBuffer>> tracker;
        id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:backwardPSO];
        
        [enc setBuffer:wrap_ptr(go, sz_go, tracker) offset:0 atIndex:0];
        [enc setBuffer:wrap_ptr(m, sz_m, tracker) offset:0 atIndex:1];
        [enc setBuffer:wrap_ptr(ic, sz_ic, tracker) offset:0 atIndex:2];
        [enc setBuffer:wrap_ptr(so, sz_so, tracker) offset:0 atIndex:3];
        [enc setBuffer:wrap_ptr(c, sz_c, tracker) offset:0 atIndex:4];
        [enc setBuffer:wrap_ptr(sti, sz_sti, tracker) offset:0 atIndex:5];
        [enc setBuffer:wrap_ptr(sgi, sz_sgi, tracker) offset:0 atIndex:6];
        [enc setBuffer:wrap_ptr(bg, sz_bg, tracker) offset:0 atIndex:7];
        
        auto wrap_grad = [&](size_t ptr, size_t sz, int idx) {
            id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:(void*)ptr length:sz options:MTLResourceStorageModeShared deallocator:nil];
            if (buf) {
                [enc setBuffer:buf offset:0 atIndex:idx];
                return std::make_pair(buf, false);
            } else {
                buf = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];
                memset([buf contents], 0, sz);
                [enc setBuffer:buf offset:0 atIndex:idx];
                return std::make_pair(buf, true);
            }
        };

        auto res_gm = wrap_grad(gm, sz_gm, 8);
        auto res_gic = wrap_grad(gic, sz_gic, 9);
        auto res_gso = wrap_grad(gso, sz_gso, 10);
        auto res_gc = wrap_grad(gc, sz_gc, 11);
        
        [enc setBytes:&H length:4 atIndex:12];
        [enc setBytes:&W length:4 atIndex:13];
        [enc setBytes:&ts length:4 atIndex:14];
        [enc setBuffer:wrap_ptr(tb, sz_tb, tracker) offset:0 atIndex:15];
        
        [enc dispatchThreads:MTLSizeMake(((W+ts-1)/ts)*16, ((H+ts-1)/ts)*16, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (res_gm.second) memcpy((void*)gm, [res_gm.first contents], sz_gm);
        if (res_gic.second) memcpy((void*)gic, [res_gic.first contents], sz_gic);
        if (res_gso.second) memcpy((void*)gso, [res_gso.first contents], sz_gso);
        if (res_gc.second) memcpy((void*)gc, [res_gc.first contents], sz_gc);

        [res_gm.first release]; [res_gic.first release]; [res_gso.first release]; [res_gc.first release];
        for (auto b : tracker) [b release];
    }
}

NB_MODULE(_MPS, m) {
    m.def("init_mps", &init_mps);
    m.def("render_forward", &render_forward_mps);
    m.def("render_backward", &render_backward_mps);
}

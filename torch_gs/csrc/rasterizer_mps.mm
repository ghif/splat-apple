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
    forwardPSO = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"render_tiles_forward"] error:nil];
    backwardPSO = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"render_tiles_backward"] error:nil];
}

id<MTLBuffer> copy_to_metal(const void* ptr, size_t size, std::vector<id<MTLBuffer>>& tracker) {
    if (size == 0) size = 16;
    id<MTLBuffer> buf = [device newBufferWithBytes:ptr length:size options:MTLResourceStorageModeShared];
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
        
        [enc setBuffer:copy_to_metal((void*)m, sz_m, tracker) offset:0 atIndex:0];
        [enc setBuffer:copy_to_metal((void*)ic, sz_ic, tracker) offset:0 atIndex:1];
        [enc setBuffer:copy_to_metal((void*)so, sz_so, tracker) offset:0 atIndex:2];
        [enc setBuffer:copy_to_metal((void*)c, sz_c, tracker) offset:0 atIndex:3];
        [enc setBuffer:copy_to_metal((void*)sti, sz_sti, tracker) offset:0 atIndex:4];
        [enc setBuffer:copy_to_metal((void*)sgi, sz_sgi, tracker) offset:0 atIndex:5];
        [enc setBuffer:copy_to_metal((void*)bg, sz_bg, tracker) offset:0 atIndex:6];
        
        id<MTLBuffer> b_out = [device newBufferWithLength:sz_out options:MTLResourceStorageModeShared];
        [enc setBuffer:b_out offset:0 atIndex:7];
        
        [enc setBytes:&H length:4 atIndex:8]; [enc setBytes:&W length:4 atIndex:9];
        [enc setBytes:&ts length:4 atIndex:10]; [enc setBytes:&ni length:4 atIndex:11];
        [enc setBuffer:copy_to_metal((void*)tb, sz_tb, tracker) offset:0 atIndex:12];
        
        [enc dispatchThreads:MTLSizeMake(((W+ts-1)/ts)*16, ((H+ts-1)/ts)*16, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        
        memcpy((void*)out, [b_out contents], sz_out);
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
        
        [enc setBuffer:copy_to_metal((void*)go, sz_go, tracker) offset:0 atIndex:0];
        [enc setBuffer:copy_to_metal((void*)m, sz_m, tracker) offset:0 atIndex:1];
        [enc setBuffer:copy_to_metal((void*)ic, sz_ic, tracker) offset:0 atIndex:2];
        [enc setBuffer:copy_to_metal((void*)so, sz_so, tracker) offset:0 atIndex:3];
        [enc setBuffer:copy_to_metal((void*)c, sz_c, tracker) offset:0 atIndex:4];
        [enc setBuffer:copy_to_metal((void*)sti, sz_sti, tracker) offset:0 atIndex:5];
        [enc setBuffer:copy_to_metal((void*)sgi, sz_sgi, tracker) offset:0 atIndex:6];
        [enc setBuffer:copy_to_metal((void*)bg, sz_bg, tracker) offset:0 atIndex:7];
        
        id<MTLBuffer> b_gm = [device newBufferWithLength:sz_gm options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_gic = [device newBufferWithLength:sz_gic options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_gso = [device newBufferWithLength:sz_gso options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_gc = [device newBufferWithLength:sz_gc options:MTLResourceStorageModeShared];
        
        [enc setBuffer:b_gm offset:0 atIndex:8]; [enc setBuffer:b_gic offset:0 atIndex:9];
        [enc setBuffer:b_gso offset:0 atIndex:10]; [enc setBuffer:b_gc offset:0 atIndex:11];
        
        [enc setBytes:&H length:4 atIndex:12]; [enc setBytes:&W length:4 atIndex:13];
        [enc setBytes:&ts length:4 atIndex:14]; [enc setBuffer:copy_to_metal((void*)tb, sz_tb, tracker) offset:0 atIndex:15];
        
        [enc dispatchThreads:MTLSizeMake(((W+ts-1)/ts)*16, ((H+ts-1)/ts)*16, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        memcpy((void*)gm, [b_gm contents], sz_gm);
        memcpy((void*)gic, [b_gic contents], sz_gic);
        memcpy((void*)gso, [b_gso contents], sz_gso);
        memcpy((void*)gc, [b_gc contents], sz_gc);

        [b_gm release]; [b_gic release]; [b_gso release]; [b_gc release];
        for (auto b : tracker) [b release];
    }
}

NB_MODULE(_MPS, m) {
    m.def("init_mps", &init_mps);
    m.def("render_forward", &render_forward_mps);
    m.def("render_backward", &render_backward_mps);
}

#include <metal_stdlib>
using namespace metal;

// struct float3 { float x, y, z; }; // Already defined in metal_stdlib

// Kernel for forward rasterization
kernel void render_tiles_forward(
    device const float* means2D [[buffer(0)]],
    device const float* inv_cov2D [[buffer(1)]],
    device const float* opacities [[buffer(2)]],
    device const float* colors [[buffer(3)]],
    device const int* sorted_tile_ids [[buffer(4)]],
    device const int* sorted_gaussian_ids [[buffer(5)]],
    device const float* background [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant int& H [[buffer(8)]],
    constant int& W [[buffer(9)]],
    constant int& tile_size [[buffer(10)]],
    constant int& num_interactions [[buffer(11)]],
    constant int* tile_boundaries [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]], // Global pixel coordinate
    uint2 tid [[thread_position_in_threadgroup]], // Local pixel coordinate in tile
    uint2 group_id [[threadgroup_position_in_grid]] // Tile coordinate
) {
    // 1. Calculate pixel coordinate
    int px = gid.x;
    int py = gid.y;
    
    // Check bounds
    if (px >= W || py >= H) return;

    // 2. Determine Tile ID (linear)
    int tiles_x = (W + tile_size - 1) / tile_size;
    int tile_idx = group_id.y * tiles_x + group_id.x;

    // 3. Get start and end indices for this tile
    // tile_boundaries must be precomputed on CPU or in a separate kernel
    // It maps tile_idx -> start_index in sorted_gaussian_ids
    int start_idx = tile_boundaries[tile_idx];
    int end_idx = tile_boundaries[tile_idx + 1];

    // 4. Accumulate color
    float T = 1.0f;
    float3 C = float3(0.0f);

    for (int i = start_idx; i < end_idx; ++i) {
        int g_idx = sorted_gaussian_ids[i];

        // Fetch Gaussian data
        float2 mean = float2(means2D[g_idx * 2 + 0], means2D[g_idx * 2 + 1]);
        float2 d = float2(px, py) - mean;
        
        // Conic matrix (inverse covariance)
        // Stored as [a, b, c, d] in flattened array? No, checking C++:
        // inv_cov2D is (N, 2, 2) flattened. So index * 4.
        float4 conic = float4(
            inv_cov2D[g_idx * 4 + 0], 
            inv_cov2D[g_idx * 4 + 1], 
            inv_cov2D[g_idx * 4 + 2], 
            inv_cov2D[g_idx * 4 + 3]
        );
        // conic.x = a, conic.y = b, conic.z = c, conic.w = d
        // Power = -0.5 * (dx^2 * a + dx*dy*b + dy*dx*c + dy^2*d)
        // Usually b=c for symmetric matrix, but let's stick to general form
        float power = -0.5f * (d.x * d.x * conic.x + d.x * d.y * (conic.y + conic.z) + d.y * d.y * conic.w);

        if (power > 0.0f) continue;

        float alpha = 1.0f - exp(-exp(power) * opacities[g_idx]); // Using exp(power) * opacity directly
        // Wait, C++ uses: alpha = 1.0f - exp(-gauss_value * op_ptr[g_idx]);
        // where gauss_value = exp(power);
        // So same thing.
        
        if (alpha < 1.0f/255.0f) continue;

        float test_T = T * (1.0f - alpha);
        if (test_T < 0.0001f) break;

        // Color
        float3 color = float3(colors[g_idx * 3 + 0], colors[g_idx * 3 + 1], colors[g_idx * 3 + 2]);
        C += color * alpha * T;
        T = test_T;
    }

    // Blend with background
    float3 bg = float3(background[0], background[1], background[2]);
    C += bg * T;

    // Write output
    int pixel_idx = (py * W + px) * 3;
    output[pixel_idx + 0] = C.x;
    output[pixel_idx + 1] = C.y;
    output[pixel_idx + 2] = C.z;
}

// Helper to add atomic float
void atomic_add_float(device atomic_uint* addr, float val) {
    uint old_val = atomic_load_explicit(addr, memory_order_relaxed);
    uint new_val;
    do {
        new_val = as_type<uint>(as_type<float>(old_val) + val);
    } while (!atomic_compare_exchange_weak_explicit(addr, &old_val, new_val, memory_order_relaxed, memory_order_relaxed));
}

// Kernel for backward rasterization
// Note: Atomic float addition is needed for gradients. 
// Metal doesn't support atomic_float natively on all devices, but M1/M2/M3 supports it via atomic_uint cast loop.
kernel void render_tiles_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* means2D [[buffer(1)]],
    device const float* inv_cov2D [[buffer(2)]],
    device const float* opacities [[buffer(3)]],
    device const float* colors [[buffer(4)]],
    device const int* sorted_tile_ids [[buffer(5)]],
    device const int* sorted_gaussian_ids [[buffer(6)]],
    device const float* background [[buffer(7)]],
    device atomic_uint* grad_means2D [[buffer(8)]], // Treat as atomic_uint for atomic_add logic
    device atomic_uint* grad_inv_cov2D [[buffer(9)]],
    device atomic_uint* grad_opacities [[buffer(10)]],
    device atomic_uint* grad_colors [[buffer(11)]],
    constant int& H [[buffer(12)]],
    constant int& W [[buffer(13)]],
    constant int& tile_size [[buffer(14)]],
    constant int* tile_boundaries [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 group_id [[threadgroup_position_in_grid]]
) {
    int px = gid.x;
    int py = gid.y;
    if (px >= W || py >= H) return;

    int tiles_x = (W + tile_size - 1) / tile_size;
    int tile_idx = group_id.y * tiles_x + group_id.x;
    int start_idx = tile_boundaries[tile_idx];
    int end_idx = tile_boundaries[tile_idx + 1];

    // Recompute blending (forward pass logic to get alpha/T)
    // This is redundant but necessary without saving per-pixel lists
    // Optimization: Store accumulated T or similar? Standard implementation recomputes.

    // First, we need to collect the alphas and Ts for this pixel
    // To avoid dynamic allocation, we might run two passes? 
    // Or just store locally if max interactions is small? 
    // Standard approach: Re-walk the list.
    
    // We need to store: alpha, T for each interaction.
    // Since we don't have dynamic memory in kernel, we usually just re-run the loop 
    // to find the range of contributors, then iterate backwards.
    
    // Forward pass to collect contributions
    float T = 1.0f;
    int num_contributors = 0;
    
    // Max contributors per pixel? 
    // We can't store infinite. Let's assume a hard limit or just use a second loop logic.
    // A better way: 
    // 1. Iterate forward to compute final T (and implicitly verify which ones contributed).
    // 2. Iterate backward from the last contributor to the first, updating gradients.
    
    // BUT, to iterate backward, we need the values of alpha/T at each step.
    // Recomputing them is expensive.
    
    // Simplified Backward Pass Strategy:
    // 1. Run forward loop to determine the *set* of contributing Gaussians and their alphas.
    //    Store indices in a local array? Stack size is limited (thread memory).
    //    Typically max 32-64 contributors per pixel are significant.
    //    Let's try a fixed size buffer.
    
    struct Contributor {
        int index;
        float alpha;
    };
    
    Contributor contributors[256]; // Max 256 contributors per pixel
    int count = 0;
    
    for (int i = start_idx; i < end_idx; ++i) {
        int g_idx = sorted_gaussian_ids[i];
        float2 mean = float2(means2D[g_idx * 2 + 0], means2D[g_idx * 2 + 1]);
        float2 d = float2(px, py) - mean;
        float4 conic = float4(inv_cov2D[g_idx * 4 + 0], inv_cov2D[g_idx * 4 + 1], inv_cov2D[g_idx * 4 + 2], inv_cov2D[g_idx * 4 + 3]);
        float power = -0.5f * (d.x * d.x * conic.x + d.x * d.y * (conic.y + conic.z) + d.y * d.y * conic.w);
        
        if (power > 0.0f) continue;
        float alpha = 1.0f - exp(-exp(power) * opacities[g_idx]);
        if (alpha < 1.0f/255.0f) continue;
        
        if (count < 256) {
            contributors[count].index = g_idx;
            contributors[count].alpha = alpha;
            count++;
        }
        
        T *= (1.0f - alpha);
        if (T < 0.0001f) break;
    }
    
    // Now Backward
    // We need T_final from forward pass? No, gradients flow from final pixel color.
    // dL/dC_pixel is given.
    // T accumulates: T_i+1 = T_i * (1 - alpha_i). T_0 = 1.
    // Final Color C = sum(c_i * alpha_i * T_i)
    
    int pixel_idx = (py * W + px) * 3;
    float3 dL_dPixel = float3(grad_output[pixel_idx + 0], grad_output[pixel_idx + 1], grad_output[pixel_idx + 2]);
    
    // We need to reconstruct T_i.
    // T at end of loop was T_final.
    // But we need T at start of each contribution.
    // T_0 = 1.
    // T_{i} = T_{i-1} * (1 - alpha_{i-1})
    
    // Let's re-calculate T per contributor.
    float T_curr = 1.0f;
    
    // We also need gradients w.r.t alpha and T.
    // The C++ code iterates backwards. 
    // It maintains an accumulated gradient for T? No, C++ code iterates backwards to propagate dL/dT.
    // Wait, the C++ code uses:
    /*
        for (int i = gs.size() - 1; i >= 0; --i) {
            float alpha = as[i];
            float T_i = Ts[i]; // T right before this gaussian
            // ...
        }
    */
    // We didn't store Ts. We can recompute them easily since we stored alphas.
    // T_0 = 1. T_1 = 1 * (1-a0). T_2 = T_1 * (1-a1)...
    // So T_i depends on all previous alphas.
    
    // So we can iterate forward through our `contributors` list to compute T_i for each.
    // Store them in a temp array? Or just recompute on the fly?
    // Since we limited to 256, we can store Ts too.
    
    float Ts[256];
    float run_T = 1.0f;
    for (int i = 0; i < count; ++i) {
        Ts[i] = run_T;
        run_T *= (1.0f - contributors[i].alpha);
    }
    
    // Now iterate backwards
    float3 dL_dAccum = float3(0.0f); // Accumulator for dL/dT?
    // Actually, looking at C++ code:
    // It accumulates contributions to dL/dalpha and dL/dcolor.
    // The term involving T is dL/dalpha = dL/dC * (color * T - (AccumulatedColorFromHereOn)) ... wait.
    
    // C++ Code logic:
    // float dL_da = dL_dC.x*(T_i*cs[0] - pc.x/(1-alpha)) + ...
    // where pc is accumulated color from *background* up to here (backwards).
    // Ah, `pc` in C++ starts as background and accumulates `w * color`.
    // So it represents the "tail" color integral.
    
    float3 pc = float3(background[0], background[1], background[2]);
    
    for (int i = count - 1; i >= 0; --i) {
        int g_idx = contributors[i].index;
        float alpha = contributors[i].alpha;
        float T_i = Ts[i];
        float w = alpha * T_i; // weight
        
        float3 c_i = float3(colors[g_idx * 3 + 0], colors[g_idx * 3 + 1], colors[g_idx * 3 + 2]);
        
        // 1. Gradient w.r.t Color
        // dC/dc_i = w
        atomic_add_float(&grad_colors[g_idx * 3 + 0], dL_dPixel.x * w);
        atomic_add_float(&grad_colors[g_idx * 3 + 1], dL_dPixel.y * w);
        atomic_add_float(&grad_colors[g_idx * 3 + 2], dL_dPixel.z * w);
        
        // 2. Gradient w.r.t Alpha
        // dC/da_i = T_i * c_i - (sum_{j>i} c_j * a_j * T_j) * (1/(1-a_i))
        // The term in paren is `pc` (accumulated color from back) MINUS background?
        // No, `pc` tracks the color formed by layers *behind* current one (including background).
        // C++: pc starts at bg. pc += w * color (after processing).
        // So at step i (going backwards), pc is color of everything behind i.
        // dL/da = dL/dC * ( T_i * c_i - pc * (1/(1-alpha)) )? 
        // Let's verify C++ formula:
        // dL_da = dL_dC * (T_i * c - pc / (1-alpha))
        // Wait, division by (1-alpha) comes from derivative of T_{j} w.r.t alpha_i.
        // T_{j} = ... * (1-alpha_i) * ...
        // So d/dalpha_i (T_j) = - T_j / (1-alpha_i).
        // Correct.
        
        float scale = (alpha < 0.99f) ? (1.0f / (1.0f - alpha + 1e-6f)) : 0.0f; // Avoid divide by zero
        float dL_da = dot(dL_dPixel, (c_i * T_i - pc * scale));
        
        // 3. Gradient w.r.t Inputs (Opacity, Means, Cov) via Alpha
        // alpha = 1 - exp(-exp(power) * opacity)
        // Let v = exp(power). alpha = 1 - exp(-v * opacity).
        // dalpha/dopacity = -exp(...) * (-v) = (1-alpha) * v
        // dalpha/dpower   = -exp(...) * (-opacity * exp(power)) = (1-alpha) * opacity * v
        
        // Recompute v (exp(power))
        // We need inputs again.
        float2 mean = float2(means2D[g_idx * 2 + 0], means2D[g_idx * 2 + 1]);
        float2 d = float2(px, py) - mean;
        float4 conic = float4(inv_cov2D[g_idx * 4 + 0], inv_cov2D[g_idx * 4 + 1], inv_cov2D[g_idx * 4 + 2], inv_cov2D[g_idx * 4 + 3]);
        float power = -0.5f * (d.x * d.x * conic.x + d.x * d.y * (conic.y + conic.z) + d.y * d.y * conic.w);
        float v = exp(power);
        
        float dL_do = dL_da * (1.0f - alpha) * v;
        float dL_dp = dL_da * (1.0f - alpha) * opacities[g_idx] * v;
        
        atomic_add_float(&grad_opacities[g_idx], dL_do);
        
        // Grads for mean and conic
        // dp/dx = - (dx*a + dy*b)
        // dp/dy = - (dx*c + dy*d)
        // dp/da = -0.5 * dx*dx ...
        
        float2 dL_dmean = float2(
            dL_dp * (d.x * conic.x + d.y * conic.y), // conic.y is b
            dL_dp * (d.x * conic.z + d.y * conic.w)  // conic.z is c
        );
        atomic_add_float(&grad_means2D[g_idx * 2 + 0], dL_dmean.x);
        atomic_add_float(&grad_means2D[g_idx * 2 + 1], dL_dmean.y);
        
        atomic_add_float(&grad_inv_cov2D[g_idx * 4 + 0], dL_dp * -0.5f * d.x * d.x);
        atomic_add_float(&grad_inv_cov2D[g_idx * 4 + 1], dL_dp * -0.5f * d.x * d.y);
        atomic_add_float(&grad_inv_cov2D[g_idx * 4 + 2], dL_dp * -0.5f * d.x * d.y);
        atomic_add_float(&grad_inv_cov2D[g_idx * 4 + 3], dL_dp * -0.5f * d.y * d.y);
        
        // Update pc (accumulate color from back)
        pc += w * c_i;
    }
}

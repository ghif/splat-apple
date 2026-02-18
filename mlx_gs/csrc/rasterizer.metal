#include <metal_stdlib>
using namespace metal;

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
    uint2 gid [[thread_position_in_grid]],
    uint2 group_id [[threadgroup_position_in_grid]]
) {
    int px = gid.x; int py = gid.y;
    if (px >= W || py >= H) return;
    int tiles_x = (W + tile_size - 1) / tile_size;
    int tile_idx = group_id.y * tiles_x + group_id.x;
    int s_idx = tile_boundaries[tile_idx];
    int e_idx = tile_boundaries[tile_idx + 1];
    float T = 1.0f; float3 C = float3(0.0f);
    for (int i = s_idx; i < e_idx; ++i) {
        int g_idx = sorted_gaussian_ids[i];
        float2 mean = float2(means2D[g_idx * 2 + 0], means2D[g_idx * 2 + 1]);
        float2 d = float2(px, py) - mean;
        float4 conic = float4(inv_cov2D[g_idx * 4 + 0], inv_cov2D[g_idx * 4 + 1], inv_cov2D[g_idx * 4 + 2], inv_cov2D[g_idx * 4 + 3]);
        float power = -0.5f * (d.x * d.x * conic.x + d.x * d.y * (conic.y + conic.z) + d.y * d.y * conic.w);
        if (power > 0.0f) continue;
        float alpha = 1.0f - exp(-exp(power) * opacities[g_idx]);
        if (alpha < 1.0f/255.0f) continue;
        float test_T = T * (1.0f - alpha);
        if (test_T < 0.0001f) break;
        float3 color = float3(colors[g_idx * 3 + 0], colors[g_idx * 3 + 1], colors[g_idx * 3 + 2]);
        C += color * alpha * T; T = test_T;
    }
    C += float3(background[0], background[1], background[2]) * T;
    int pixel_idx = (py * W + px) * 3;
    output[pixel_idx + 0] = C.x; output[pixel_idx + 1] = C.y; output[pixel_idx + 2] = C.z;
}

void atomic_add_float(device atomic_uint* addr, float val) {
    uint old_val = atomic_load_explicit(addr, memory_order_relaxed);
    uint new_val;
    do { new_val = as_type<uint>(as_type<float>(old_val) + val); } 
    while (!atomic_compare_exchange_weak_explicit(addr, &old_val, new_val, memory_order_relaxed, memory_order_relaxed));
}

kernel void render_tiles_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* means2D [[buffer(1)]],
    device const float* inv_cov2D [[buffer(2)]],
    device const float* opacities [[buffer(3)]],
    device const float* colors [[buffer(4)]],
    device const int* sorted_tile_ids [[buffer(5)]],
    device const int* sorted_gaussian_ids [[buffer(6)]],
    device const float* background [[buffer(7)]],
    device atomic_uint* grad_means2D [[buffer(8)]],
    device atomic_uint* grad_inv_cov2D [[buffer(9)]],
    device atomic_uint* grad_opacities [[buffer(10)]],
    device atomic_uint* grad_colors [[buffer(11)]],
    constant int& H [[buffer(12)]], constant int& W [[buffer(13)]], constant int& tile_size [[buffer(14)]],
    constant int* tile_boundaries [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]], uint2 group_id [[threadgroup_position_in_grid]]
) {
    int px = gid.x; int py = gid.y;
    if (px >= W || py >= H) return;
    int tiles_x = (W + tile_size - 1) / tile_size;
    int tile_idx = group_id.y * tiles_x + group_id.x;
    int s_idx = tile_boundaries[tile_idx];
    int e_idx = tile_boundaries[tile_idx + 1];
    float T = 1.0f;
    struct Contributor { int index; float alpha; };
    Contributor contributors[256]; int count = 0;
    for (int i = s_idx; i < e_idx; ++i) {
        int g_idx = sorted_gaussian_ids[i];
        float2 mean = float2(means2D[g_idx * 2 + 0], means2D[g_idx * 2 + 1]);
        float2 d = float2(px, py) - mean;
        float4 conic = float4(inv_cov2D[g_idx * 4 + 0], inv_cov2D[g_idx * 4 + 1], inv_cov2D[g_idx * 4 + 2], inv_cov2D[g_idx * 4 + 3]);
        float power = -0.5f * (d.x * d.x * conic.x + d.x * d.y * (conic.y + conic.z) + d.y * d.y * conic.w);
        if (power > 0.0f) continue;
        float alpha = 1.0f - exp(-exp(power) * opacities[g_idx]);
        if (alpha < 1.0f/255.0f) continue;
        if (count < 256) { contributors[count].index = g_idx; contributors[count].alpha = alpha; count++; }
        T *= (1.0f - alpha); if (T < 0.0001f) break;
    }
    float3 dL_dPixel = float3(grad_output[(py * W + px) * 3 + 0], grad_output[(py * W + px) * 3 + 1], grad_output[(py * W + px) * 3 + 2]);
    float Ts[256]; float run_T = 1.0f;
    for (int i = 0; i < count; ++i) { Ts[i] = run_T; run_T *= (1.0f - contributors[i].alpha); }
    float3 pc = float3(background[0], background[1], background[2]);
    for (int i = count - 1; i >= 0; --i) {
        int g_idx = contributors[i].index; float alpha = contributors[i].alpha; float T_i = Ts[i]; float w = alpha * T_i;
        float3 c_i = float3(colors[g_idx * 3 + 0], colors[g_idx * 3 + 1], colors[g_idx * 3 + 2]);
        atomic_add_float(&grad_colors[g_idx * 3 + 0], dL_dPixel.x * w); atomic_add_float(&grad_colors[g_idx * 3 + 1], dL_dPixel.y * w); atomic_add_float(&grad_colors[g_idx * 3 + 2], dL_dPixel.z * w);
        float scale = (alpha < 0.99f) ? (1.0f / (1.0f - alpha + 1e-6f)) : 0.0f;
        float dL_da = dot(dL_dPixel, (c_i * T_i - pc * scale));
        float2 mean = float2(means2D[g_idx * 2 + 0], means2D[g_idx * 2 + 1]);
        float2 d = float2(px, py) - mean;
        float4 conic = float4(inv_cov2D[g_idx * 4 + 0], inv_cov2D[g_idx * 4 + 1], inv_cov2D[g_idx * 4 + 2], inv_cov2D[g_idx * 4 + 3]);
        float power = -0.5f * (d.x * d.x * conic.x + d.x * d.y * (conic.y + conic.z) + d.y * d.y * conic.w);
        float v = exp(power);
        float dL_do = dL_da * (1.0f - alpha) * v; float dL_dp = dL_da * (1.0f - alpha) * opacities[g_idx] * v;
        atomic_add_float(&grad_opacities[g_idx], dL_do);
        float2 dL_dmean = float2(dL_dp * (d.x * conic.x + d.y * conic.y), dL_dp * (d.x * conic.z + d.y * conic.w));
        atomic_add_float(&grad_means2D[g_idx * 2 + 0], dL_dmean.x); atomic_add_float(&grad_means2D[g_idx * 2 + 1], dL_dmean.y);
        atomic_add_float(&grad_inv_cov2D[g_idx * 4 + 0], dL_dp * -0.5f * d.x * d.x); atomic_add_float(&grad_inv_cov2D[g_idx * 4 + 1], dL_dp * -0.5f * d.x * d.y);
        atomic_add_float(&grad_inv_cov2D[g_idx * 4 + 2], dL_dp * -0.5f * d.x * d.y); atomic_add_float(&grad_inv_cov2D[g_idx * 4 + 3], dL_dp * -0.5f * d.y * d.y);
        pc += w * c_i;
    }
}

// Phase 4: Stable GPU Interaction Stage

kernel void count_tiles(
    device const float* means2D [[buffer(0)]],
    device const float* radii [[buffer(1)]],
    device const uint* valid_mask [[buffer(2)]],
    device uint* counts [[buffer(3)]],
    constant int& num_points [[buffer(4)]],
    constant int& H [[buffer(5)]], constant int& W [[buffer(6)]], constant int& tile_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)num_points) return;
    if (!valid_mask[gid]) { counts[gid] = 0; return; }
    float x = means2D[gid * 2 + 0]; float y = means2D[gid * 2 + 1]; float r = radii[gid];
    int tx = (W + tile_size - 1) / tile_size; int ty = (H + tile_size - 1) / tile_size;
    int min_x = max(0, (int)floor((x - r) / tile_size)); int max_x = min(tx - 1, (int)floor((x + r) / tile_size));
    int min_y = max(0, (int)floor((y - r) / tile_size)); int max_y = min(ty - 1, (int)floor((y + r) / tile_size));
    if (x + r < 0 || x - r >= W || y + r < 0 || y - r >= H) counts[gid] = 0;
    else counts[gid] = (max_x - min_x + 1) * (max_y - min_y + 1);
}

kernel void expand_gaussians(
    device const uint* offsets [[buffer(0)]],
    device int* gaussian_ids [[buffer(1)]],
    constant int& num_points [[buffer(2)]],
    constant int& total_interactions [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)total_interactions) return;
    int low = 0, high = num_points - 1;
    while (low < high) {
        int mid = (low + high) / 2;
        if (offsets[mid] <= gid) low = mid + 1;
        else high = mid;
    }
    gaussian_ids[gid] = low;
}

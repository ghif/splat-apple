#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "rasterizer.h"

struct float3 { float x, y, z; };

// Atomic add for float on CPU
inline void atomic_add(float* addr, float val) {
    auto* atomic_addr = reinterpret_cast<std::atomic<float>*>(addr);
    float expected = atomic_addr->load(std::memory_order_relaxed);
    while (!atomic_addr->compare_exchange_weak(expected, expected + val, std::memory_order_relaxed));
}

// Helper to calculate tile bounds for a Gaussian
struct TileBounds {
    int min_x, max_x, min_y, max_y;
};

TileBounds get_tile_bounds(float x, float y, float r, int W, int H, int tile_size) {
    int min_x = std::max(0, (int)std::floor((x - r) / tile_size));
    int max_x = std::min((W + tile_size - 1) / tile_size - 1, (int)std::floor((x + r) / tile_size));
    int min_y = std::max(0, (int)std::floor((y - r) / tile_size));
    int max_y = std::min((H + tile_size - 1) / tile_size - 1, (int)std::floor((y + r) / tile_size));
    return {min_x, max_x, min_y, max_y};
}

std::tuple<at::Tensor, at::Tensor> get_tile_interactions_cpp(
    at::Tensor means2D,
    at::Tensor radii,
    at::Tensor valid_mask,
    at::Tensor depths,
    int H, int W,
    int tile_size
) {
    auto device = means2D.device();
    int num_points = means2D.size(0);
    int num_tiles_x = (W + tile_size - 1) / tile_size;
    
    auto means2D_host = means2D.to(at::kCPU);
    auto radii_host = radii.to(at::kCPU);
    auto valid_mask_host = valid_mask.to(at::kCPU);
    auto depths_host = depths.to(at::kCPU);

    auto m2d_ptr = means2D_host.accessor<float, 2>();
    auto r_ptr = radii_host.accessor<float, 1>();
    auto v_ptr = valid_mask_host.accessor<bool, 1>();
    auto d_ptr = depths_host.accessor<float, 1>();

    std::vector<int32_t> tile_ids;
    std::vector<int32_t> gaussian_ids;
    std::vector<float> interaction_depths;

    for (int i = 0; i < num_points; ++i) {
        if (!v_ptr[i]) continue;
        float x = m2d_ptr[i][0];
        float y = m2d_ptr[i][1];
        float r = r_ptr[i];
        if (x + r < 0 || x - r >= W || y + r < 0 || y - r >= H) continue;

        auto bounds = get_tile_bounds(x, y, r, W, H, tile_size);
        for (int ty = bounds.min_y; ty <= bounds.max_y; ++ty) {
            for (int tx = bounds.min_x; tx <= bounds.max_x; ++tx) {
                tile_ids.push_back(ty * num_tiles_x + tx);
                gaussian_ids.push_back(i);
                interaction_depths.push_back(d_ptr[i]);
            }
        }
    }

    if (tile_ids.empty()) {
        return std::make_tuple(at::empty({0}, at::kInt).to(device), at::empty({0}, at::kInt).to(device));
    }

    std::vector<size_t> indices(tile_ids.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        if (tile_ids[i] != tile_ids[j]) return tile_ids[i] < tile_ids[j];
        return interaction_depths[i] < interaction_depths[j];
    });

    at::Tensor out_tile_ids = at::empty({(long)tile_ids.size()}, at::kInt);
    at::Tensor out_gaussian_ids = at::empty({(long)gaussian_ids.size()}, at::kInt);
    auto oti_ptr = out_tile_ids.accessor<int, 1>();
    auto ogi_ptr = out_gaussian_ids.accessor<int, 1>();
    for (size_t i = 0; i < indices.size(); ++i) {
        oti_ptr[i] = tile_ids[indices[i]];
        ogi_ptr[i] = gaussian_ids[indices[i]];
    }
    return std::make_tuple(out_tile_ids.to(device), out_gaussian_ids.to(device));
}

at::Tensor render_tiles_cpp_forward_internal(
    at::Tensor means2D, at::Tensor cov2D, at::Tensor opacities, at::Tensor colors,
    at::Tensor sorted_tile_ids, at::Tensor sorted_gaussian_ids,
    int H, int W, int tile_size, at::Tensor background
) {
    auto device = means2D.device();
    int num_tiles_x = (W + tile_size - 1) / tile_size;
    int num_tiles_y = (H + tile_size - 1) / tile_size;
    int num_tiles = num_tiles_x * num_tiles_y;
    at::Tensor output = at::zeros({H, W, 3}, at::kFloat);
    
    auto det = (cov2D.index({at::indexing::Slice(), 0, 0}) * cov2D.index({at::indexing::Slice(), 1, 1}) - 
                cov2D.index({at::indexing::Slice(), 0, 1}).pow(2)).clamp(1e-6);
    auto inv_cov2D = at::stack({
        at::stack({cov2D.index({at::indexing::Slice(), 1, 1}) / det, -cov2D.index({at::indexing::Slice(), 0, 1}) / det}, -1),
        at::stack({-cov2D.index({at::indexing::Slice(), 0, 1}) / det, cov2D.index({at::indexing::Slice(), 0, 0}) / det}, -1)
    }, -2);
    auto sig_opacities = at::sigmoid(opacities);

    auto sti_cpu = sorted_tile_ids.to(at::kCPU);
    auto sti_ptr = sti_cpu.accessor<int, 1>();
    std::vector<int> tile_boundaries(num_tiles + 1, 0);
    int curr = 0;
    for (int t = 0; t <= num_tiles; ++t) {
        while (curr < sti_cpu.size(0) && sti_ptr[curr] < t) curr++;
        tile_boundaries[t] = curr;
    }

    auto m2d_cpu = means2D.to(at::kCPU); auto icov_cpu = inv_cov2D.to(at::kCPU);
    auto ops_cpu = sig_opacities.to(at::kCPU); auto cols_cpu = colors.to(at::kCPU);
    auto sgi_cpu = sorted_gaussian_ids.to(at::kCPU); auto bg_cpu = background.to(at::kCPU);
    auto m2d_acc = m2d_cpu.accessor<float, 2>(); auto icov_acc = icov_cpu.accessor<float, 3>();
    auto ops_acc = ops_cpu.accessor<float, 2>(); auto cols_acc = cols_cpu.accessor<float, 2>();
    auto sgi_acc = sgi_cpu.accessor<int, 1>(); auto bg_acc = bg_cpu.accessor<float, 1>();
    auto out_acc = output.accessor<float, 3>();

    at::parallel_for(0, num_tiles, 1, [&](int64_t start, int64_t end) {
        for (int t = start; t < end; ++t) {
            int ty = t / num_tiles_x, tx = t % num_tiles_x;
            int px_min = tx * tile_size, py_min = ty * tile_size;
            int s_idx = tile_boundaries[t], e_idx = tile_boundaries[t + 1];
            for (int py = 0; py < tile_size; ++py) {
                int gy = py_min + py; if (gy >= H) continue;
                for (int px = 0; px < tile_size; ++px) {
                    int gx = px_min + px; if (gx >= W) continue;
                    float T = 1.0f, r = 0, g = 0, b = 0;
                    for (int i = s_idx; i < e_idx; ++i) {
                        int g_idx = sgi_acc[i];
                        float dx = (float)gx - m2d_acc[g_idx][0], dy = (float)gy - m2d_acc[g_idx][1];
                        float a = icov_acc[g_idx][0][0], bc = icov_acc[g_idx][0][1], d = icov_acc[g_idx][1][1];
                        float power = -0.5f * (dx * dx * a + dx * dy * 2.0f * bc + dy * dy * d);
                        if (power > 0) continue;
                        float alpha = std::min(0.99f, (float)std::exp(power) * ops_acc[g_idx][0]);
                        if (alpha < 1.0f/255.0f) continue;
                        float w = alpha * T;
                        r += cols_acc[g_idx][0] * w; g += cols_acc[g_idx][1] * w; b += cols_acc[g_idx][2] * w;
                        T *= (1.0f - alpha); if (T < 0.0001f) break;
                    }
                    out_acc[gy][gx][0] = r + T * bg_acc[0]; out_acc[gy][gx][1] = g + T * bg_acc[1]; out_acc[gy][gx][2] = b + T * bg_acc[2];
                }
            }
        }
    });
    return output.to(device);
}

at::Tensor GaussianRasterizer::forward(torch::autograd::AutogradContext *ctx, at::Tensor m, at::Tensor c, at::Tensor o, at::Tensor col, at::Tensor sti, at::Tensor sgi, int H, int W, int ts, at::Tensor bg) {
    ctx->save_for_backward({m, c, o, col, sti, sgi, bg});
    ctx->saved_data["H"] = H; ctx->saved_data["W"] = W; ctx->saved_data["ts"] = ts;
    return render_tiles_cpp_forward_internal(m, c, o, col, sti, sgi, H, W, ts, bg);
}

torch::autograd::variable_list GaussianRasterizer::backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    at::Tensor m = saved[0], c = saved[1], o = saved[2], col = saved[3], sti = saved[4], sgi = saved[5], bg = saved[6];
    int H = ctx->saved_data["H"].toInt(), W = ctx->saved_data["W"].toInt(), ts = ctx->saved_data["ts"].toInt();
    auto grad_image = grad_outputs[0].to(at::kCPU); auto gi_acc = grad_image.accessor<float, 3>();
    auto device = m.device(); int num_tiles_x = (W + ts - 1) / ts, num_tiles = num_tiles_x * ((H + ts - 1) / ts);
    at::Tensor gm = at::zeros_like(m).to(at::kCPU), gc = at::zeros_like(c).to(at::kCPU), go = at::zeros_like(o).to(at::kCPU), gcol = at::zeros_like(col).to(at::kCPU);
    auto gm_acc = gm.data_ptr<float>(), gc_acc = gc.data_ptr<float>(), go_acc = go.data_ptr<float>(), gcol_acc = gcol.data_ptr<float>();
    
    auto det = (c.index({at::indexing::Slice(), 0, 0}) * c.index({at::indexing::Slice(), 1, 1}) - c.index({at::indexing::Slice(), 0, 1}).pow(2)).clamp(1e-6);
    auto icov = at::stack({at::stack({c.index({at::indexing::Slice(), 1, 1}) / det, -c.index({at::indexing::Slice(), 0, 1}) / det}, -1), at::stack({-c.index({at::indexing::Slice(), 0, 1}) / det, c.index({at::indexing::Slice(), 0, 0}) / det}, -1)}, -2);
    auto sig_o = at::sigmoid(o);
    auto sti_cpu = sti.to(at::kCPU); auto sti_ptr = sti_cpu.accessor<int, 1>();
    std::vector<int> tile_boundaries(num_tiles + 1, 0);
    int curr = 0; for (int t = 0; t <= num_tiles; ++t) { while (curr < sti_cpu.size(0) && sti_ptr[curr] < t) curr++; tile_boundaries[t] = curr; }
    
    auto m_cpu = m.to(at::kCPU); auto icov_cpu = icov.to(at::kCPU); auto ops_cpu = sig_o.to(at::kCPU); auto cols_cpu = col.to(at::kCPU); auto sgi_cpu = sgi.to(at::kCPU); auto bg_cpu = bg.to(at::kCPU);
    auto m_acc = m_cpu.accessor<float, 2>(); auto ic_acc = icov_cpu.accessor<float, 3>(); auto ops_acc = ops_cpu.accessor<float, 2>(); auto cs_acc = cols_cpu.accessor<float, 2>(); auto sgi_acc = sgi_cpu.accessor<int, 1>(); auto bg_acc = bg_cpu.accessor<float, 1>();

    at::parallel_for(0, num_tiles, 1, [&](int64_t start, int64_t end) {
        for (int t = start; t < end; ++t) {
            int ty = t / num_tiles_x, tx = t % num_tiles_x, px_min = tx * ts, py_min = ty * ts;
            int s_idx = tile_boundaries[t], e_idx = tile_boundaries[t+1];
            for (int py = 0; py < ts; ++py) {
                int gy = py_min + py; if (gy >= H) continue;
                for (int px = 0; px < ts; ++px) {
                    int gx = px_min+px; if (gx >= W) continue;
                    float T = 1.0f; std::vector<float> as, Ts; std::vector<int> gs;
                    for (int i = s_idx; i < e_idx; ++i) {
                        int g_idx = sgi_acc[i];
                        float dx = (float)gx - m_acc[g_idx][0], dy = (float)gy - m_acc[g_idx][1];
                        float a = ic_acc[g_idx][0][0], bc = ic_acc[g_idx][0][1], d = ic_acc[g_idx][1][1];
                        float p = -0.5f * (dx * dx * a + dx * dy * 2.0f * bc + dy * dy * d);
                        if (p > 0) continue;
                        float alpha = std::min(0.99f, (float)std::exp(p) * ops_acc[g_idx][0]);
                        if (alpha < 1.0f/255.0f) continue;
                        as.push_back(alpha); Ts.push_back(T); gs.push_back(g_idx);
                        T *= (1.0f - alpha); if (T < 0.0001f) break;
                    }
                    float3 dL_dC = {gi_acc[gy][gx][0], gi_acc[gy][gx][1], gi_acc[gy][gx][2]}, pc = {bg_acc[0], bg_acc[1], bg_acc[2]};
                    for (int i = (int)gs.size() - 1; i >= 0; --i) {
                        int g_idx = gs[i]; float alpha = as[i], T_i = Ts[i], w = alpha * T_i;
                        atomic_add(&gcol_acc[g_idx*3+0], dL_dC.x * w); atomic_add(&gcol_acc[g_idx*3+1], dL_dC.y * w); atomic_add(&gcol_acc[g_idx*3+2], dL_dC.z * w);
                        float dL_da = dL_dC.x*(T_i*cs_acc[g_idx][0]-pc.x/(1.0f-alpha+1e-6)) + dL_dC.y*(T_i*cs_acc[g_idx][1]-pc.y/(1.0f-alpha+1e-6)) + dL_dC.z*(T_i*cs_acc[g_idx][2]-pc.z/(1.0f-alpha+1e-6));
                        atomic_add(&go_acc[g_idx], dL_da * (alpha / (ops_acc[g_idx][0] + 1e-6)));
                        // Grad for power: alpha = exp(p) * op -> dL/dp = dL/da * alpha
                        float dL_dp = dL_da * alpha;
                        float dx = (float)gx - m_acc[g_idx][0], dy = (float)gy - m_acc[g_idx][1];
                        float a = ic_acc[g_idx][0][0], bc = ic_acc[g_idx][0][1], d = ic_acc[g_idx][1][1];
                        // dL/dm_x = dL/dp * dL/dp/ddx * ddx/dm_x = dL/dp * -(dx*a + dy*bc) * -1 = dL/dp * (dx*a + dy*bc)
                        atomic_add(&gm_acc[g_idx*2+0], dL_dp * (dx * a + dy * bc));
                        atomic_add(&gm_acc[g_idx*2+1], dL_dp * (dx * bc + dy * d));
                        // dL/dicov: power = -0.5 * (dx^2 * a + dx*dy*2*bc + dy^2 * d)
                        atomic_add(&gc_acc[g_idx*4+0], dL_dp * -0.5f * dx * dx);
                        atomic_add(&gc_acc[g_idx*4+1], dL_dp * -0.5f * dx * dy);
                        atomic_add(&gc_acc[g_idx*4+2], dL_dp * -0.5f * dx * dy);
                        atomic_add(&gc_acc[g_idx*4+3], dL_dp * -0.5f * dy * dy);
                        pc.x += w * cs_acc[g_idx][0]; pc.y += w * cs_acc[g_idx][1]; pc.z += w * cs_acc[g_idx][2];
                    }
                }
            }
        }
    });

    // Compute grad_cov2D from grad_icov (inv_cov2D) using d(A^-1) = -A^-1 * dA * A^-1
    // matrix multiplication: grad_cov2D = - icov * grad_icov * icov
    auto gc_device = gc.to(device);
    auto grad_cov2D = -at::matmul(icov, at::matmul(gc_device, icov));

    return {gm.to(device), grad_cov2D, go.to(device), gcol.to(device), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_tile_interactions", &get_tile_interactions_cpp);
    m.def("render_tiles", [](at::Tensor m, at::Tensor c, at::Tensor o, at::Tensor col, at::Tensor sti, at::Tensor sgi, int H, int W, int ts, at::Tensor bg) {
        return GaussianRasterizer::apply(m, c, o, col, sti, sgi, H, W, ts, bg);
    });
}

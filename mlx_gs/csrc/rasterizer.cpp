#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <dispatch/dispatch.h>
#include <numeric>

namespace nb = nanobind;

struct float3 { float x, y, z; };

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

inline void atomic_add(float* addr, float val) {
    auto* atomic_addr = reinterpret_cast<std::atomic<float>*>(addr);
    float expected = atomic_addr->load(std::memory_order_relaxed);
    while (!atomic_addr->compare_exchange_weak(expected, expected + val, std::memory_order_relaxed));
}

nb::tuple get_tile_interactions(
    nb::ndarray<float, nb::ndim<2>> means2D,
    nb::ndarray<float, nb::ndim<1>> radii,
    nb::ndarray<bool, nb::ndim<1>> valid_mask,
    nb::ndarray<float, nb::ndim<1>> depths,
    int H, int W, int tile_size
) {
    int num_points = (int)means2D.shape(0);
    int num_tiles_x = (W + tile_size - 1) / tile_size;
    
    std::vector<int32_t> tile_ids;
    std::vector<int32_t> gaussian_ids;
    std::vector<float> interaction_depths;

    auto m_ptr = means2D.data();
    auto r_ptr = radii.data();
    auto v_ptr = valid_mask.data();
    auto d_ptr = depths.data();

    for (int i = 0; i < num_points; ++i) {
        if (!v_ptr[i]) continue;
        float x = m_ptr[i * 2 + 0];
        float y = m_ptr[i * 2 + 1];
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
        int32_t* empty_data = new int32_t[0];
        nb::capsule free_empty(empty_data, [](void *p) noexcept { delete[] (int32_t *)p; });
        return nb::make_tuple(
            nb::ndarray<int32_t, nb::numpy>(empty_data, {0}, free_empty),
            nb::ndarray<int32_t, nb::numpy>(empty_data, {0}, free_empty)
        );
    }

    std::vector<size_t> indices(tile_ids.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        if (tile_ids[i] != tile_ids[j]) return tile_ids[i] < tile_ids[j];
        return interaction_depths[i] < interaction_depths[j];
    });

    auto out_tile_ids = new int32_t[tile_ids.size()];
    auto out_gaussian_ids = new int32_t[gaussian_ids.size()];
    for (size_t i = 0; i < indices.size(); ++i) {
        out_tile_ids[i] = tile_ids[indices[i]];
        out_gaussian_ids[i] = gaussian_ids[indices[i]];
    }

    nb::capsule free_tile_ids(out_tile_ids, [](void *p) noexcept { delete[] (int32_t *)p; });
    nb::capsule free_gaussian_ids(out_gaussian_ids, [](void *p) noexcept { delete[] (int32_t *)p; });

    return nb::make_tuple(
        nb::ndarray<int32_t, nb::numpy>(out_tile_ids, {tile_ids.size()}, free_tile_ids),
        nb::ndarray<int32_t, nb::numpy>(out_gaussian_ids, {gaussian_ids.size()}, free_gaussian_ids)
    );
}

nb::ndarray<float, nb::numpy> render_tiles_forward(
    nb::ndarray<float, nb::ndim<2>> means2D,
    nb::ndarray<float, nb::ndim<3>> inv_cov2D,
    nb::ndarray<float, nb::ndim<2>> sig_opacities,
    nb::ndarray<float, nb::ndim<2>> colors,
    nb::ndarray<int32_t, nb::ndim<1>> sorted_tile_ids,
    nb::ndarray<int32_t, nb::ndim<1>> sorted_gaussian_ids,
    int H, int W, int tile_size,
    nb::ndarray<float, nb::ndim<1>> background
) {
    int num_tiles_x = (W + tile_size - 1) / tile_size;
    int num_tiles_y = (H + tile_size - 1) / tile_size;
    int num_tiles = num_tiles_x * num_tiles_y;
    
    float* output_data = new float[H * W * 3]();
    
    auto m_ptr = means2D.data();
    auto ic_ptr = inv_cov2D.data();
    auto op_ptr = sig_opacities.data();
    auto col_ptr = colors.data();
    auto sti_ptr = sorted_tile_ids.data();
    auto sgi_ptr = sorted_gaussian_ids.data();
    auto bg_ptr = background.data();

    std::vector<int> tile_boundaries(num_tiles + 1, 0);
    int curr = 0;
    int num_interactions = (int)sorted_tile_ids.shape(0);
    for (int t = 0; t <= num_tiles; ++t) {
        while (curr < num_interactions && sti_ptr[curr] < t) curr++;
        tile_boundaries[t] = curr;
    }

    dispatch_apply(num_tiles, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t t) {
        int ty = (int)t / num_tiles_x, tx = (int)t % num_tiles_x;
        int px_min = tx * tile_size, py_min = ty * tile_size;
        int s_idx = tile_boundaries[t], e_idx = tile_boundaries[t + 1];
        
        for (int py = 0; py < tile_size; ++py) {
            int gy = py_min + py; if (gy >= H) continue;
            for (int px = 0; px < tile_size; ++px) {
                int gx = px_min + px; if (gx >= W) continue;
                
                float T = 1.0f, r = 0, g = 0, b = 0;
                for (int i = s_idx; i < e_idx; ++i) {
                    int g_idx = sgi_ptr[i];
                    float dx = (float)gx - m_ptr[g_idx * 2 + 0];
                    float dy = (float)gy - m_ptr[g_idx * 2 + 1];
                    float a = ic_ptr[g_idx * 4 + 0];
                    float bc = ic_ptr[g_idx * 4 + 1];
                    float d = ic_ptr[g_idx * 4 + 3];
                    float power = -0.5f * (dx * dx * a + dx * dy * 2.0f * bc + dy * dy * d);
                    if (power > 0) continue;
                    float gauss_value = std::exp(power);
                    float alpha = 1.0f - std::exp(-gauss_value * op_ptr[g_idx]);
                    if (alpha < 1e-6f) continue;
                    
                    float w = alpha * T;
                    r += col_ptr[g_idx * 3 + 0] * w;
                    g += col_ptr[g_idx * 3 + 1] * w;
                    b += col_ptr[g_idx * 3 + 2] * w;
                    T *= (1.0f - alpha);
                    if (T < 0.0001f) break;
                }
                output_data[(gy * W + gx) * 3 + 0] = r + T * bg_ptr[0];
                output_data[(gy * W + gx) * 3 + 1] = g + T * bg_ptr[1];
                output_data[(gy * W + gx) * 3 + 2] = b + T * bg_ptr[2];
            }
        }
    });

    nb::capsule free_output(output_data, [](void *p) noexcept { delete[] (float *)p; });
    return nb::ndarray<float, nb::numpy>(output_data, {(size_t)H, (size_t)W, 3}, free_output);
}

nb::tuple render_tiles_backward(
    nb::ndarray<float, nb::ndim<3>> grad_output,
    nb::ndarray<float, nb::ndim<2>> means2D,
    nb::ndarray<float, nb::ndim<3>> inv_cov2D,
    nb::ndarray<float, nb::ndim<2>> sig_opacities,
    nb::ndarray<float, nb::ndim<2>> colors,
    nb::ndarray<int32_t, nb::ndim<1>> sorted_tile_ids,
    nb::ndarray<int32_t, nb::ndim<1>> sorted_gaussian_ids,
    int H, int W, int tile_size,
    nb::ndarray<float, nb::ndim<1>> background
) {
    int num_tiles_x = (W + tile_size - 1) / tile_size;
    int num_tiles = num_tiles_x * ((H + tile_size - 1) / tile_size);
    int num_points = (int)means2D.shape(0);

    float* gm = new float[num_points * 2]();
    float* gic = new float[num_points * 4](); 
    float* go = new float[num_points]();
    float* gcol = new float[num_points * 3]();

    auto m_ptr = means2D.data();
    auto ic_ptr = inv_cov2D.data();
    auto op_ptr = sig_opacities.data();
    auto col_ptr = colors.data();
    auto sti_ptr = sorted_tile_ids.data();
    auto sgi_ptr = sorted_gaussian_ids.data();
    auto bg_ptr = background.data();
    auto go_ptr = grad_output.data();

    std::vector<int> tile_boundaries(num_tiles + 1, 0);
    int curr = 0;
    int num_interactions = (int)sorted_tile_ids.shape(0);
    for (int t = 0; t <= num_tiles; ++t) {
        while (curr < num_interactions && sti_ptr[curr] < t) curr++;
        tile_boundaries[t] = curr;
    }

    dispatch_apply(num_tiles, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t t) {
        int ty = (int)t / num_tiles_x, tx = (int)t % num_tiles_x, px_min = tx * tile_size, py_min = ty * tile_size;
        int s_idx = tile_boundaries[t], e_idx = tile_boundaries[t + 1];
        
        for (int py = 0; py < tile_size; ++py) {
            int gy = py_min + py; if (gy >= H) continue;
            for (int px = 0; px < tile_size; ++px) {
                int gx = px_min + px; if (gx >= W) continue;

                float T = 1.0f;
                std::vector<float> as, Ts;
                std::vector<int> gs;
                for (int i = s_idx; i < e_idx; ++i) {
                    int g_idx = sgi_ptr[i];
                    float dx = (float)gx - m_ptr[g_idx * 2 + 0];
                    float dy = (float)gy - m_ptr[g_idx * 2 + 1];
                    float a = ic_ptr[g_idx * 4 + 0], bc = ic_ptr[g_idx * 4 + 1], d = ic_ptr[g_idx * 4 + 3];
                    float p = -0.5f * (dx * dx * a + dx * dy * 2.0f * bc + dy * dy * d);
                    if (p > 0) continue;
                    float v = std::exp(p);
                    float alpha = 1.0f - std::exp(-v * op_ptr[g_idx]);
                    if (alpha < 1e-6f) continue;
                    as.push_back(alpha); Ts.push_back(T); gs.push_back(g_idx);
                    T *= (1.0f - alpha); if (T < 0.0001f) break;
                }

                float3 dL_dC = {go_ptr[(gy * W + gx) * 3 + 0], go_ptr[(gy * W + gx) * 3 + 1], go_ptr[(gy * W + gx) * 3 + 2]};
                float3 pc = {bg_ptr[0], bg_ptr[1], bg_ptr[2]};
                
                for (int i = (int)gs.size() - 1; i >= 0; --i) {
                    int g_idx = gs[i];
                    float alpha = as[i], T_i = Ts[i], w = alpha * T_i;
                    
                    atomic_add(&gcol[g_idx * 3 + 0], dL_dC.x * w);
                    atomic_add(&gcol[g_idx * 3 + 1], dL_dC.y * w);
                    atomic_add(&gcol[g_idx * 3 + 2], dL_dC.z * w);
                    
                    float dL_da = dL_dC.x * (T_i * col_ptr[g_idx * 3 + 0] - pc.x / (1.0f - alpha + 1e-6)) +
                                  dL_dC.y * (T_i * col_ptr[g_idx * 3 + 1] - pc.y / (1.0f - alpha + 1e-6)) +
                                  dL_dC.z * (T_i * col_ptr[g_idx * 3 + 2] - pc.z / (1.0f - alpha + 1e-6));
                    
                    float dx = (float)gx - m_ptr[g_idx * 2 + 0];
                    float dy = (float)gy - m_ptr[g_idx * 2 + 1];
                    float a = ic_ptr[g_idx * 4 + 0], bc = ic_ptr[g_idx * 4 + 1], d = ic_ptr[g_idx * 4 + 3];
                    float p = -0.5f * (dx * dx * a + dx * dy * 2.0f * bc + dy * dy * d);
                    float v = std::exp(p);
                    
                    // dL/do = dL/da * exp(-v*o) * v = dL/da * (1-alpha) * v
                    atomic_add(&go[g_idx], dL_da * (1.0f - alpha) * v);
                    
                    // dL/dp = dL/da * exp(-v*o) * o * v = dL/da * (1-alpha) * o * v
                    float dL_dp = dL_da * (1.0f - alpha) * op_ptr[g_idx] * v;
                    
                    atomic_add(&gm[g_idx * 2 + 0], dL_dp * (dx * a + dy * bc));
                    atomic_add(&gm[g_idx * 2 + 1], dL_dp * (dx * bc + dy * d));
                    
                    atomic_add(&gic[g_idx * 4 + 0], dL_dp * -0.5f * dx * dx);
                    atomic_add(&gic[g_idx * 4 + 1], dL_dp * -0.5f * dx * dy);
                    atomic_add(&gic[g_idx * 4 + 2], dL_dp * -0.5f * dx * dy);
                    atomic_add(&gic[g_idx * 4 + 3], dL_dp * -0.5f * dy * dy);
                    
                    pc.x += w * col_ptr[g_idx * 3 + 0];
                    pc.y += w * col_ptr[g_idx * 3 + 1];
                    pc.z += w * col_ptr[g_idx * 3 + 2];
                }
            }
        }
    });

    nb::capsule free_gm(gm, [](void *p) noexcept { delete[] (float *)p; });
    nb::capsule free_gic(gic, [](void *p) noexcept { delete[] (float *)p; });
    nb::capsule free_go(go, [](void *p) noexcept { delete[] (float *)p; });
    nb::capsule free_gcol(gcol, [](void *p) noexcept { delete[] (float *)p; });

    return nb::make_tuple(
        nb::ndarray<float, nb::numpy>(gm, {(size_t)num_points, 2}, free_gm),
        nb::ndarray<float, nb::numpy>(gic, {(size_t)num_points, 2, 2}, free_gic),
        nb::ndarray<float, nb::numpy>(go, {(size_t)num_points}, free_go),
        nb::ndarray<float, nb::numpy>(gcol, {(size_t)num_points, 3}, free_gcol)
    );
}

NB_MODULE(rasterizer_c_api, m) {
    m.def("get_tile_interactions", &get_tile_interactions);
    m.def("render_tiles_forward", &render_tiles_forward);
    m.def("render_tiles_backward", &render_tiles_backward);
}

#pragma once
#include <torch/extension.h>
#include <vector>

std::tuple<at::Tensor, at::Tensor> get_tile_interactions_cpp(
    at::Tensor means2D,
    at::Tensor radii,
    at::Tensor valid_mask,
    at::Tensor depths,
    int H, int W,
    int tile_size
);

at::Tensor render_tiles_cpp_forward(
    at::Tensor means2D,
    at::Tensor cov2D,
    at::Tensor opacities,
    at::Tensor colors,
    at::Tensor sorted_tile_ids,
    at::Tensor sorted_gaussian_ids,
    int H, int W,
    int tile_size,
    at::Tensor background
);

// Autograd function declaration
class GaussianRasterizer : public torch::autograd::Function<GaussianRasterizer> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor means2D,
        at::Tensor cov2D,
        at::Tensor opacities,
        at::Tensor colors,
        at::Tensor sorted_tile_ids,
        at::Tensor sorted_gaussian_ids,
        int H, int W,
        int tile_size,
        at::Tensor background
    );

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    );
};

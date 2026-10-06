#pragma once

#include <torch/torch.h>

#include <tuple>

namespace pcops
{
torch::Tensor ball_query_cuda(torch::Tensor new_xyz, torch::Tensor xyz, double radius, int64_t nsample);
torch::Tensor group_points_cuda(torch::Tensor features, torch::Tensor idx);
torch::Tensor group_points_backward_cuda(torch::Tensor grad_out, torch::Tensor idx, int64_t n);
torch::Tensor gather_points_cuda(torch::Tensor features, torch::Tensor idx);
torch::Tensor gather_points_backward_cuda(torch::Tensor grad_out, torch::Tensor idx, int64_t n);
std::tuple<torch::Tensor, torch::Tensor> three_nn_cuda(torch::Tensor unknown, torch::Tensor known);
torch::Tensor three_interpolate_cuda(torch::Tensor features, torch::Tensor idx, torch::Tensor weight);
torch::Tensor three_interpolate_backward_cuda(torch::Tensor grad_out, torch::Tensor idx, torch::Tensor weight, int64_t m);
} // namespace pcops

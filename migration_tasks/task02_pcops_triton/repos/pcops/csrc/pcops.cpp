// pcops - operator registration.
//
// Every operator lives in the `pcops` namespace of the PyTorch dispatcher
// (torch.ops.pcops.*) with a CUDA kernel. Autograd formulas and the fake
// (meta) implementations used by torch.compile are registered from Python,
// see pcops/_registration.py.
#include <Python.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "cluster/cuda/fps_cuda.h"
#include "cluster/cuda/grid_cuda.h"
#include "cluster/cuda/knn_cuda.h"
#include "cluster/cuda/radius_cuda.h"
#include "pointnet/pointnet_cuda.h"
#include "scatter/cuda/scatter_cuda.h"
#include "scatter/cuda/segment_csr_cuda.h"

// `import pcops._C` loads this library, whose static initialisers register the
// operators; the module itself is empty.
extern "C" PyObject *PyInit__C(void)
{
    static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
    return PyModule_Create(&module_def);
}

namespace
{
torch::Tensor fps(torch::Tensor src, torch::Tensor ptr, double ratio, bool random_start)
{
    TORCH_CHECK(ratio > 0.0 && ratio <= 1.0, "fps: ratio must be in (0, 1], got ", ratio);
    auto r = torch::full({ptr.numel() - 1}, ratio, src.options());
    return fps_cuda(src, ptr, r, random_start);
}

torch::Tensor knn(torch::Tensor x, torch::Tensor y, std::optional<torch::Tensor> ptr_x, std::optional<torch::Tensor> ptr_y,
                  int64_t k, bool cosine)
{
    return knn_cuda(x.contiguous(), y.contiguous(), ptr_x, ptr_y, k, cosine);
}

torch::Tensor radius(torch::Tensor x, torch::Tensor y, std::optional<torch::Tensor> ptr_x, std::optional<torch::Tensor> ptr_y,
                     double r, int64_t max_num_neighbors, bool ignore_same_index)
{
    return radius_cuda(x.contiguous(), y.contiguous(), ptr_x, ptr_y, r, max_num_neighbors, ignore_same_index);
}

torch::Tensor grid(torch::Tensor pos, torch::Tensor size, std::optional<torch::Tensor> start, std::optional<torch::Tensor> end)
{
    return grid_cuda(pos, size, start, end);
}

// Returns (out, arg_out); arg_out is an empty int64 tensor for sum and mean.
std::tuple<torch::Tensor, torch::Tensor> segment_csr(torch::Tensor src, torch::Tensor indptr, std::string reduce)
{
    TORCH_CHECK(reduce == "sum" || reduce == "mean" || reduce == "min" || reduce == "max", "segment_csr: unsupported reduce '", reduce, "'");
    auto result = segment_csr_cuda(src, indptr, std::nullopt, reduce);
    auto arg    = std::get<1>(result);
    return {std::get<0>(result), arg.has_value() ? arg.value() : torch::empty({0}, indptr.options())};
}

torch::Tensor gather_csr(torch::Tensor src, torch::Tensor indptr) { return gather_csr_cuda(src, indptr, std::nullopt); }

// reduce in {sum, mean, min, max}; index has the shape of src. As in
// torch_scatter.scatter_mean, mean divides the sums by the element counts
// clamped to >= 1 (floor division for integer types).
std::tuple<torch::Tensor, torch::Tensor> scatter(torch::Tensor src, torch::Tensor index, int64_t dim, int64_t dim_size, std::string reduce)
{
    TORCH_CHECK(reduce == "sum" || reduce == "mean" || reduce == "min" || reduce == "max", "scatter: unsupported reduce '", reduce, "'");
    dim = dim < 0 ? dim + src.dim() : dim;
    if(reduce == "mean")
    {
        auto sum   = std::get<0>(scatter_cuda(src, index, dim, std::nullopt, dim_size, "sum"));
        auto ones  = torch::ones(index.sizes(), src.options());
        auto count = std::get<0>(scatter_cuda(ones, index, dim, std::nullopt, dim_size, "sum"));
        count.masked_fill_(count < 1, 1);
        auto out = sum.is_floating_point() ? sum.div_(count) : sum.div_(count, "floor");
        return {out, torch::empty({0}, index.options())};
    }
    auto result = scatter_cuda(src, index, dim, std::nullopt, dim_size, reduce);
    auto arg    = std::get<1>(result);
    return {std::get<0>(result), arg.has_value() ? arg.value() : torch::empty({0}, index.options())};
}
} // namespace

TORCH_LIBRARY(pcops, m)
{
    m.def("fps(Tensor src, Tensor ptr, float ratio, bool random_start) -> Tensor");
    m.def("knn(Tensor x, Tensor y, Tensor? ptr_x, Tensor? ptr_y, int k, bool cosine) -> Tensor");
    m.def("radius(Tensor x, Tensor y, Tensor? ptr_x, Tensor? ptr_y, float r, int max_num_neighbors, bool ignore_same_index) -> Tensor");
    m.def("grid(Tensor pos, Tensor size, Tensor? start, Tensor? end) -> Tensor");
    m.def("segment_csr(Tensor src, Tensor indptr, str reduce) -> (Tensor, Tensor)");
    m.def("gather_csr(Tensor src, Tensor indptr) -> Tensor");
    m.def("scatter(Tensor src, Tensor index, int dim, int dim_size, str reduce) -> (Tensor, Tensor)");
    m.def("ball_query(Tensor new_xyz, Tensor xyz, float radius, int nsample) -> Tensor");
    m.def("group_points(Tensor features, Tensor idx) -> Tensor");
    m.def("group_points_backward(Tensor grad_out, Tensor idx, int n) -> Tensor");
    m.def("gather_points(Tensor features, Tensor idx) -> Tensor");
    m.def("gather_points_backward(Tensor grad_out, Tensor idx, int n) -> Tensor");
    m.def("three_nn(Tensor unknown, Tensor known) -> (Tensor, Tensor)");
    m.def("three_interpolate(Tensor features, Tensor idx, Tensor weight) -> Tensor");
    m.def("three_interpolate_backward(Tensor grad_out, Tensor idx, Tensor weight, int m) -> Tensor");
}

TORCH_LIBRARY_IMPL(pcops, CUDA, m)
{
    m.impl("fps", &fps);
    m.impl("knn", &knn);
    m.impl("radius", &radius);
    m.impl("grid", &grid);
    m.impl("segment_csr", &segment_csr);
    m.impl("gather_csr", &gather_csr);
    m.impl("scatter", &scatter);
    m.impl("ball_query", &pcops::ball_query_cuda);
    m.impl("group_points", &pcops::group_points_cuda);
    m.impl("group_points_backward", &pcops::group_points_backward_cuda);
    m.impl("gather_points", &pcops::gather_points_cuda);
    m.impl("gather_points_backward", &pcops::gather_points_backward_cuda);
    m.impl("three_nn", &pcops::three_nn_cuda);
    m.impl("three_interpolate", &pcops::three_interpolate_cuda);
    m.impl("three_interpolate_backward", &pcops::three_interpolate_backward_cuda);
}

"""Fake (meta) implementations and autograd formulas of the pcops operators.

The kernels are registered by the compiled extension (csrc/pcops.cpp) under
torch.ops.pcops.*. This module adds what torch.compile and autograd need:
shape propagation for every operator and derivatives for the differentiable
ones. Operators whose output size depends on the data (fps, knn, radius,
gather_csr) allocate unbacked dynamic sizes.
"""
import torch

_lib = "pcops::"


def _fake(name):
    return torch.library.register_fake(_lib + name)


# ---------------------------------------------------------------- fake impls

@_fake("fps")
def _fps_fake(src, ptr, ratio, random_start):
    n = torch.library.get_ctx().new_dynamic_size()
    return src.new_empty(n, dtype=torch.long)


@_fake("knn")
def _knn_fake(x, y, ptr_x, ptr_y, k, cosine):
    e = torch.library.get_ctx().new_dynamic_size()
    return x.new_empty(2, e, dtype=torch.long)


@_fake("radius")
def _radius_fake(x, y, ptr_x, ptr_y, r, max_num_neighbors, ignore_same_index):
    e = torch.library.get_ctx().new_dynamic_size()
    return x.new_empty(2, e, dtype=torch.long)


@_fake("grid")
def _grid_fake(pos, size, start, end):
    return pos.new_empty(pos.size(0), dtype=torch.long)


def _segment_shape(src, indptr):
    dim = indptr.dim() - 1
    shape = list(src.shape)
    shape[dim] = max(indptr.size(-1) - 1, 0)
    return shape


@_fake("segment_csr")
def _segment_csr_fake(src, indptr, reduce):
    shape = _segment_shape(src, indptr)
    out = src.new_empty(shape)
    arg = src.new_empty(shape, dtype=torch.long) if reduce in ("min", "max") else src.new_empty(0, dtype=torch.long)
    return out, arg


@_fake("gather_csr")
def _gather_csr_fake(src, indptr):
    dim = indptr.dim() - 1
    shape = list(src.shape)
    shape[dim] = torch.library.get_ctx().new_dynamic_size()
    return src.new_empty(shape)


@_fake("scatter")
def _scatter_fake(src, index, dim, dim_size, reduce):
    dim = dim % src.dim()
    shape = list(src.shape)
    shape[dim] = dim_size
    out = src.new_empty(shape)
    arg = src.new_empty(shape, dtype=torch.long) if reduce in ("min", "max") else src.new_empty(0, dtype=torch.long)
    return out, arg


@_fake("ball_query")
def _ball_query_fake(new_xyz, xyz, radius, nsample):
    return xyz.new_empty(new_xyz.size(0), new_xyz.size(1), nsample, dtype=torch.int32)


@_fake("group_points")
def _group_points_fake(features, idx):
    return features.new_empty(features.size(0), features.size(1), idx.size(1), idx.size(2))


@_fake("group_points_backward")
def _group_points_backward_fake(grad_out, idx, n):
    return grad_out.new_empty(grad_out.size(0), grad_out.size(1), n)


@_fake("gather_points")
def _gather_points_fake(features, idx):
    return features.new_empty(features.size(0), features.size(1), idx.size(1))


@_fake("gather_points_backward")
def _gather_points_backward_fake(grad_out, idx, n):
    return grad_out.new_empty(grad_out.size(0), grad_out.size(1), n)


@_fake("three_nn")
def _three_nn_fake(unknown, known):
    return unknown.new_empty(unknown.shape), unknown.new_empty(unknown.shape, dtype=torch.int32)


@_fake("three_interpolate")
def _three_interpolate_fake(features, idx, weight):
    return features.new_empty(features.size(0), features.size(1), idx.size(1))


@_fake("three_interpolate_backward")
def _three_interpolate_backward_fake(grad_out, idx, weight, m):
    return grad_out.new_empty(grad_out.size(0), grad_out.size(1), m)


# ---------------------------------------------------------------- autograd

def _group_setup(ctx, inputs, output):
    features, idx = inputs
    ctx.save_for_backward(idx)
    ctx.n = features.size(2)


def _group_backward(ctx, grad):
    (idx,) = ctx.saved_tensors
    return torch.ops.pcops.group_points_backward(grad.contiguous(), idx, ctx.n), None


torch.library.register_autograd(_lib + "group_points", _group_backward, setup_context=_group_setup)


def _gather_setup(ctx, inputs, output):
    features, idx = inputs
    ctx.save_for_backward(idx)
    ctx.n = features.size(2)


def _gather_backward(ctx, grad):
    (idx,) = ctx.saved_tensors
    return torch.ops.pcops.gather_points_backward(grad.contiguous(), idx, ctx.n), None


torch.library.register_autograd(_lib + "gather_points", _gather_backward, setup_context=_gather_setup)


def _interp_setup(ctx, inputs, output):
    features, idx, weight = inputs
    ctx.save_for_backward(idx, weight)
    ctx.m = features.size(2)


def _interp_backward(ctx, grad):
    idx, weight = ctx.saved_tensors
    # `weight` is treated as a constant (as in PointNet++)
    return torch.ops.pcops.three_interpolate_backward(grad.contiguous(), idx, weight, ctx.m), None, None


torch.library.register_autograd(_lib + "three_interpolate", _interp_backward, setup_context=_interp_setup)


def _segment_setup(ctx, inputs, output):
    src, indptr, reduce = inputs
    out, arg = output
    ctx.reduce = reduce
    ctx.src_shape = src.shape
    ctx.save_for_backward(indptr, arg)


def _segment_backward(ctx, grad_out, grad_arg):
    indptr, arg = ctx.saved_tensors
    dim = indptr.dim() - 1
    if ctx.reduce == "sum":
        return torch.ops.pcops.gather_csr(grad_out.contiguous(), indptr), None, None
    if ctx.reduce == "mean":
        count = (indptr.narrow(dim, 1, indptr.size(dim) - 1) - indptr.narrow(dim, 0, indptr.size(dim) - 1)).clamp_(min=1)
        count = count.to(grad_out.dtype).view(count.shape + (1,) * (grad_out.dim() - count.dim()))
        return torch.ops.pcops.gather_csr((grad_out / count).contiguous(), indptr), None, None
    # min / max: the gradient flows to the selected element; empty segments
    # carry arg == src.size(dim) and are routed into a scratch slot.
    shape = list(ctx.src_shape)
    shape[dim] += 1
    grad_in = grad_out.new_zeros(shape)
    grad_in.scatter_(dim, arg, grad_out)
    return grad_in.narrow(dim, 0, ctx.src_shape[dim]), None, None


torch.library.register_autograd(_lib + "segment_csr", _segment_backward, setup_context=_segment_setup)


def _scatter_setup(ctx, inputs, output):
    src, index, dim, dim_size, reduce = inputs
    out, arg = output
    ctx.reduce = reduce
    ctx.dim = dim % src.dim()
    ctx.src_shape = src.shape
    ctx.dim_size = dim_size
    ctx.save_for_backward(index, arg)


def _broadcast_index(index, src_shape, dim):
    if index.dim() == 1:
        view = [1] * len(src_shape)
        view[dim] = -1
        index = index.view(view)
    return index.expand(src_shape)


def _scatter_backward(ctx, grad_out, grad_arg):
    index, arg = ctx.saved_tensors
    dim = ctx.dim
    idx = _broadcast_index(index, ctx.src_shape, dim)
    if ctx.reduce == "sum":
        return grad_out.gather(dim, idx), None, None, None, None
    if ctx.reduce == "mean":
        ones = torch.ones(ctx.src_shape, dtype=grad_out.dtype, device=grad_out.device)
        count = torch.ops.pcops.scatter(ones, idx.contiguous(), dim, ctx.dim_size, "sum")[0].clamp_(min=1)
        return (grad_out / count).gather(dim, idx), None, None, None, None
    shape = list(ctx.src_shape)
    shape[dim] += 1
    grad_in = grad_out.new_zeros(shape)
    grad_in.scatter_(dim, arg, grad_out)
    return grad_in.narrow(dim, 0, ctx.src_shape[dim]), None, None, None, None


torch.library.register_autograd(_lib + "scatter", _scatter_backward, setup_context=_scatter_setup)


def _gather_csr_setup(ctx, inputs, output):
    src, indptr = inputs
    ctx.save_for_backward(indptr)


def _gather_csr_backward(ctx, grad):
    (indptr,) = ctx.saved_tensors
    return torch.ops.pcops.segment_csr(grad.contiguous(), indptr, "sum")[0], None


torch.library.register_autograd(_lib + "gather_csr", _gather_csr_backward, setup_context=_gather_csr_setup)

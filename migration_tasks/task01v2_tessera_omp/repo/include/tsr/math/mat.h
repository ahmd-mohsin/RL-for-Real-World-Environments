// Tessera - small fixed-size matrices for device code.
//
// Column-major storage: Mat<C, R> has C columns of R rows and m[c][r] is the
// element in column c, row r (the GLSL/glm convention). Constructors take
// elements column by column. Products follow ordinary linear algebra:
// Mat<C, R> * Mat<K, C> -> Mat<K, R> and Mat<C, R> * vec<C> -> vec<R>.
#pragma once

#include "tsr/device/backend.h"
#include "tsr/math/vec.h"

namespace tsr::math
{
template<int N>
struct vec_of;
template<>
struct vec_of<2>
{
    using type = float2;
};
template<>
struct vec_of<3>
{
    using type = float3;
};
template<>
struct vec_of<4>
{
    using type = float4;
};
template<int N>
using vec_t = typename vec_of<N>::type;

TSR_HDI float comp(const float2 &v, int i) { return i == 0 ? v.x : v.y; }
TSR_HDI float comp(const float3 &v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
TSR_HDI float comp(const float4 &v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w)); }

template<int N>
TSR_HDI vec_t<N> vec_from(const float *a)
{
    if constexpr(N == 2)
    {
        return make_float2(a[0], a[1]);
    }
    else if constexpr(N == 3)
    {
        return make_float3(a[0], a[1], a[2]);
    }
    else
    {
        return make_float4(a[0], a[1], a[2], a[3]);
    }
}

template<int C, int R>
struct Mat
{
    float m[C][R];

    TSR_HDI float *operator[](int c) { return m[c]; }
    TSR_HDI const float *operator[](int c) const { return m[c]; }

    TSR_HDI static Mat zero()
    {
        Mat out{};
        TSR_UNROLL
        for(int c = 0; c < C; ++c)
        {
            TSR_UNROLL
            for(int r = 0; r < R; ++r)
            {
                out.m[c][r] = 0.f;
            }
        }
        return out;
    }

    TSR_HDI static Mat identity()
    {
        static_assert(C == R, "identity requires a square matrix");
        Mat out = zero();
        TSR_UNROLL
        for(int i = 0; i < C; ++i)
        {
            out.m[i][i] = 1.f;
        }
        return out;
    }

    TSR_HDI static Mat diagonal(const vec_t<C> &d)
    {
        static_assert(C == R, "diagonal requires a square matrix");
        Mat out = zero();
        TSR_UNROLL
        for(int i = 0; i < C; ++i)
        {
            out.m[i][i] = comp(d, i);
        }
        return out;
    }

    TSR_HDI vec_t<R> col(int c) const { return vec_from<R>(m[c]); }

    TSR_HDI vec_t<C> row(int r) const
    {
        float tmp[C];
        TSR_UNROLL
        for(int c = 0; c < C; ++c)
        {
            tmp[c] = m[c][r];
        }
        return vec_from<C>(tmp);
    }
};

using mat2   = Mat<2, 2>;
using mat3   = Mat<3, 3>;
using mat4   = Mat<4, 4>;
using mat3x2 = Mat<3, 2>; // 3 columns, 2 rows: maps R^3 -> R^2
using mat2x3 = Mat<2, 3>;

// Column-major element-list constructors.
TSR_HDI mat2 make_mat2(float c0r0, float c0r1, float c1r0, float c1r1)
{
    return mat2{{{c0r0, c0r1}, {c1r0, c1r1}}};
}

TSR_HDI mat3 make_mat3(float c0r0, float c0r1, float c0r2, float c1r0, float c1r1, float c1r2, float c2r0, float c2r1, float c2r2)
{
    return mat3{{{c0r0, c0r1, c0r2}, {c1r0, c1r1, c1r2}, {c2r0, c2r1, c2r2}}};
}

TSR_HDI mat3x2 make_mat3x2(float c0r0, float c0r1, float c1r0, float c1r1, float c2r0, float c2r1)
{
    return mat3x2{{{c0r0, c0r1}, {c1r0, c1r1}, {c2r0, c2r1}}};
}

TSR_HDI mat3 make_mat3_cols(const float3 &c0, const float3 &c1, const float3 &c2)
{
    return make_mat3(c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z);
}

template<int C, int R, int K>
TSR_HDI Mat<K, R> operator*(const Mat<C, R> &a, const Mat<K, C> &b)
{
    Mat<K, R> out;
    TSR_UNROLL
    for(int k = 0; k < K; ++k)
    {
        TSR_UNROLL
        for(int r = 0; r < R; ++r)
        {
            float acc = 0.f;
            TSR_UNROLL
            for(int c = 0; c < C; ++c)
            {
                acc += a.m[c][r] * b.m[k][c];
            }
            out.m[k][r] = acc;
        }
    }
    return out;
}

template<int C, int R>
TSR_HDI vec_t<R> operator*(const Mat<C, R> &a, const vec_t<C> &v)
{
    float tmp[R];
    TSR_UNROLL
    for(int r = 0; r < R; ++r)
    {
        float acc = 0.f;
        TSR_UNROLL
        for(int c = 0; c < C; ++c)
        {
            acc += a.m[c][r] * comp(v, c);
        }
        tmp[r] = acc;
    }
    return vec_from<R>(tmp);
}

// Explicit overloads so float2/3/4 arguments deduce without help.
TSR_HDI float2 operator*(const mat2 &a, const float2 &v) { return operator*<2, 2>(a, v); }
TSR_HDI float3 operator*(const mat3 &a, const float3 &v) { return operator*<3, 3>(a, v); }
TSR_HDI float4 operator*(const mat4 &a, const float4 &v) { return operator*<4, 4>(a, v); }
TSR_HDI float2 operator*(const mat3x2 &a, const float3 &v) { return operator*<3, 2>(a, v); }

template<int C, int R>
TSR_HDI Mat<C, R> operator+(const Mat<C, R> &a, const Mat<C, R> &b)
{
    Mat<C, R> out;
    TSR_UNROLL
    for(int c = 0; c < C; ++c)
    {
        TSR_UNROLL
        for(int r = 0; r < R; ++r)
        {
            out.m[c][r] = a.m[c][r] + b.m[c][r];
        }
    }
    return out;
}

template<int C, int R>
TSR_HDI Mat<C, R> operator*(const Mat<C, R> &a, float s)
{
    Mat<C, R> out;
    TSR_UNROLL
    for(int c = 0; c < C; ++c)
    {
        TSR_UNROLL
        for(int r = 0; r < R; ++r)
        {
            out.m[c][r] = a.m[c][r] * s;
        }
    }
    return out;
}

template<int C, int R>
TSR_HDI Mat<C, R> operator*(float s, const Mat<C, R> &a)
{
    return a * s;
}

template<int C, int R>
TSR_HDI Mat<R, C> transpose(const Mat<C, R> &a)
{
    Mat<R, C> out;
    TSR_UNROLL
    for(int c = 0; c < C; ++c)
    {
        TSR_UNROLL
        for(int r = 0; r < R; ++r)
        {
            out.m[r][c] = a.m[c][r];
        }
    }
    return out;
}

TSR_HDI float determinant(const mat2 &a) { return a.m[0][0] * a.m[1][1] - a.m[1][0] * a.m[0][1]; }

TSR_HDI float determinant(const mat3 &a)
{
    return a.m[0][0] * (a.m[1][1] * a.m[2][2] - a.m[2][1] * a.m[1][2])
         - a.m[1][0] * (a.m[0][1] * a.m[2][2] - a.m[2][1] * a.m[0][2])
         + a.m[2][0] * (a.m[0][1] * a.m[1][2] - a.m[1][1] * a.m[0][2]);
}

TSR_HDI mat2 inverse(const mat2 &a)
{
    const float inv_det = 1.f / determinant(a);
    return make_mat2(a.m[1][1] * inv_det, -a.m[0][1] * inv_det, -a.m[1][0] * inv_det, a.m[0][0] * inv_det);
}

TSR_HDI mat3 inverse(const mat3 &a)
{
    const float inv_det = 1.f / determinant(a);
    mat3 out;
    out.m[0][0] = (a.m[1][1] * a.m[2][2] - a.m[2][1] * a.m[1][2]) * inv_det;
    out.m[1][0] = -(a.m[1][0] * a.m[2][2] - a.m[2][0] * a.m[1][2]) * inv_det;
    out.m[2][0] = (a.m[1][0] * a.m[2][1] - a.m[2][0] * a.m[1][1]) * inv_det;
    out.m[0][1] = -(a.m[0][1] * a.m[2][2] - a.m[2][1] * a.m[0][2]) * inv_det;
    out.m[1][1] = (a.m[0][0] * a.m[2][2] - a.m[2][0] * a.m[0][2]) * inv_det;
    out.m[2][1] = -(a.m[0][0] * a.m[2][1] - a.m[2][0] * a.m[0][1]) * inv_det;
    out.m[0][2] = (a.m[0][1] * a.m[1][2] - a.m[1][1] * a.m[0][2]) * inv_det;
    out.m[1][2] = -(a.m[0][0] * a.m[1][2] - a.m[1][0] * a.m[0][2]) * inv_det;
    out.m[2][2] = (a.m[0][0] * a.m[1][1] - a.m[1][0] * a.m[0][1]) * inv_det;
    return out;
}

// a * b^T for column vectors a (rows) and b (columns).
TSR_HDI mat3 outer(const float3 &a, const float3 &b)
{
    return make_mat3(a.x * b.x, a.y * b.x, a.z * b.x, a.x * b.y, a.y * b.y, a.z * b.y, a.x * b.z, a.y * b.z, a.z * b.z);
}

// Upper-triangle packing of a symmetric 2x2 matrix: (xx, xy, yy).
TSR_HDI float3 pack_sym(const mat2 &a) { return make_float3(a.m[0][0], a.m[0][1], a.m[1][1]); }
TSR_HDI mat2 unpack_sym(const float3 &p) { return make_mat2(p.x, p.y, p.y, p.z); }

} // namespace tsr::math

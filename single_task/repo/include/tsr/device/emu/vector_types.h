// Tessera - emulated device backend
// Built-in vector types with the same names, layout and alignment as CUDA's
// <vector_types.h>, so device code compiles unchanged on the host backend.
#pragma once

#include <cstddef>
#include <cstdint>

#define TSR_EMU_VEC2(T, NAME, ALIGN)       \
    struct alignas(ALIGN) NAME             \
    {                                      \
        T x, y;                            \
    };                                     \
    inline constexpr NAME make_##NAME(T x, T y) { return NAME{x, y}; }

#define TSR_EMU_VEC3(T, NAME)              \
    struct NAME                            \
    {                                      \
        T x, y, z;                         \
    };                                     \
    inline constexpr NAME make_##NAME(T x, T y, T z) { return NAME{x, y, z}; }

#define TSR_EMU_VEC4(T, NAME, ALIGN)       \
    struct alignas(ALIGN) NAME             \
    {                                      \
        T x, y, z, w;                      \
    };                                     \
    inline constexpr NAME make_##NAME(T x, T y, T z, T w) { return NAME{x, y, z, w}; }

TSR_EMU_VEC2(float, float2, 8)
TSR_EMU_VEC3(float, float3)
TSR_EMU_VEC4(float, float4, 16)
TSR_EMU_VEC2(int, int2, 8)
TSR_EMU_VEC3(int, int3)
TSR_EMU_VEC4(int, int4, 16)
TSR_EMU_VEC2(unsigned int, uint2, 8)
TSR_EMU_VEC3(unsigned int, uint3)
TSR_EMU_VEC4(unsigned int, uint4, 16)
TSR_EMU_VEC2(double, double2, 16)
TSR_EMU_VEC2(long long, longlong2, 16)
TSR_EMU_VEC2(unsigned long long, ulonglong2, 16)
TSR_EMU_VEC2(unsigned short, ushort2, 4)
TSR_EMU_VEC4(unsigned char, uchar4, 4)
TSR_EMU_VEC4(signed char, char4, 4)

#undef TSR_EMU_VEC2
#undef TSR_EMU_VEC3
#undef TSR_EMU_VEC4

struct dim3
{
    unsigned int x, y, z;
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    constexpr operator uint3() const { return uint3{x, y, z}; }
};

static_assert(sizeof(float2) == 8 && alignof(float2) == 8, "float2 layout");
static_assert(sizeof(float3) == 12 && alignof(float3) == 4, "float3 layout");
static_assert(sizeof(float4) == 16 && alignof(float4) == 16, "float4 layout");
static_assert(sizeof(int2) == 8 && sizeof(uint4) == 16, "int vector layout");

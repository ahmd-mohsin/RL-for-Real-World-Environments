// Tessera - emulated device backend: CUDA language surface.
//
// Execution-space qualifiers, built-in variables, synchronization, warp
// intrinsics, atomics and the device math library, implemented on the host.
// This header is only ever included when compiling without nvcc.
#pragma once

#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>

#include "tsr/device/emu/runtime.h"
#include "tsr/device/emu/vector_types.h"

// ---- qualifiers -------------------------------------------------------------
#define __host__
#define __device__
#define __global__
#define __constant__
#define __managed__
#define __forceinline__ inline __attribute__((always_inline))
#define __launch_bounds__(...)
#define __grid_constant__
// Static shared memory: one instance per worker thread. A worker runs one
// block at a time and that block's fibers share the worker's thread_local
// storage, which is exactly CUDA's per-block lifetime.
#define __shared__ static thread_local

// ---- built-in variables -------------------------------------------------------
#define threadIdx (::tsr::emu::current_thread().thread_idx)
#define blockIdx (::tsr::emu::current_thread().block_idx)
#define blockDim (::tsr::emu::current_thread().block_dim)
#define gridDim (::tsr::emu::current_thread().grid_dim)
constexpr int warpSize = 32;

// ---- synchronization ----------------------------------------------------------
inline void __syncthreads() { ::tsr::emu::block_barrier(); }
inline int __syncthreads_count(int pred) { return ::tsr::emu::block_barrier_count(pred); }
inline int __syncthreads_and(int pred) { return ::tsr::emu::block_barrier_and(pred); }
inline int __syncthreads_or(int pred) { return ::tsr::emu::block_barrier_or(pred); }
inline void __syncwarp(unsigned mask = 0xffffffffu) { ::tsr::emu::warp_barrier(mask); }
inline void __threadfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
inline void __threadfence_block() { std::atomic_thread_fence(std::memory_order_seq_cst); }
inline unsigned __activemask() { return ::tsr::emu::warp_live_mask(); }
[[noreturn]] inline void __trap() { std::abort(); }

inline unsigned __lane_id() { return ::tsr::emu::current_thread().lane; }

// ---- warp shuffles --------------------------------------------------------------
namespace tsr::emu::detail
{
inline unsigned shfl_segment_base(unsigned lane, int width) { return lane & ~static_cast<unsigned>(width - 1); }
} // namespace tsr::emu::detail

template<typename T>
inline T __shfl_sync(unsigned mask, T var, int src_lane, int width = warpSize)
{
    const unsigned lane = __lane_id();
    const unsigned src  = ::tsr::emu::detail::shfl_segment_base(lane, width) + (static_cast<unsigned>(src_lane) & static_cast<unsigned>(width - 1));
    const std::uint64_t *slots = ::tsr::emu::warp_exchange(mask, ::tsr::emu::pack_bits(var));
    return ::tsr::emu::unpack_bits<T>(slots[src]);
}

template<typename T>
inline T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = warpSize)
{
    const unsigned lane        = __lane_id();
    const unsigned in_segment  = lane & static_cast<unsigned>(width - 1);
    const std::uint64_t *slots = ::tsr::emu::warp_exchange(mask, ::tsr::emu::pack_bits(var));
    return in_segment >= delta ? ::tsr::emu::unpack_bits<T>(slots[lane - delta]) : var;
}

template<typename T>
inline T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = warpSize)
{
    const unsigned lane        = __lane_id();
    const unsigned in_segment  = lane & static_cast<unsigned>(width - 1);
    const std::uint64_t *slots = ::tsr::emu::warp_exchange(mask, ::tsr::emu::pack_bits(var));
    return in_segment + delta < static_cast<unsigned>(width) ? ::tsr::emu::unpack_bits<T>(slots[lane + delta]) : var;
}

template<typename T>
inline T __shfl_xor_sync(unsigned mask, T var, int lane_mask, int width = warpSize)
{
    const unsigned lane        = __lane_id();
    const unsigned src         = lane ^ static_cast<unsigned>(lane_mask);
    const unsigned base        = ::tsr::emu::detail::shfl_segment_base(lane, width);
    const std::uint64_t *slots = ::tsr::emu::warp_exchange(mask, ::tsr::emu::pack_bits(var));
    return (src >= base && src < base + static_cast<unsigned>(width)) ? ::tsr::emu::unpack_bits<T>(slots[src]) : var;
}

// ---- warp votes -----------------------------------------------------------------
inline unsigned __ballot_sync(unsigned mask, int pred)
{
    unsigned lanes             = 0;
    const std::uint64_t *slots = ::tsr::emu::warp_exchange(mask, pred != 0 ? 1u : 0u, &lanes);
    unsigned result            = 0;
    for(unsigned l = 0; l < 32; ++l)
    {
        if(((lanes >> l) & 1u) && slots[l] != 0)
        {
            result |= 1u << l;
        }
    }
    return result;
}

inline int __any_sync(unsigned mask, int pred) { return __ballot_sync(mask, pred) != 0; }

inline int __all_sync(unsigned mask, int pred)
{
    unsigned lanes             = 0;
    const std::uint64_t *slots = ::tsr::emu::warp_exchange(mask, pred != 0 ? 1u : 0u, &lanes);
    for(unsigned l = 0; l < 32; ++l)
    {
        if(((lanes >> l) & 1u) && slots[l] == 0)
        {
            return 0;
        }
    }
    return 1;
}

// ---- atomics ----------------------------------------------------------------------
template<typename T>
inline T atomicAdd(T *address, T val)
{
    return std::atomic_ref<T>(*address).fetch_add(val, std::memory_order_relaxed);
}

template<typename T>
inline T atomicSub(T *address, T val)
{
    return std::atomic_ref<T>(*address).fetch_sub(val, std::memory_order_relaxed);
}

template<typename T>
inline T atomicExch(T *address, T val)
{
    return std::atomic_ref<T>(*address).exchange(val, std::memory_order_relaxed);
}

template<typename T>
inline T atomicCAS(T *address, T compare, T val)
{
    std::atomic_ref<T>(*address).compare_exchange_strong(compare, val, std::memory_order_relaxed);
    return compare; // holds the old value whether or not the swap happened
}

template<typename T>
inline T atomicMax(T *address, T val)
{
    std::atomic_ref<T> ref(*address);
    T old = ref.load(std::memory_order_relaxed);
    while(old < val && !ref.compare_exchange_weak(old, val, std::memory_order_relaxed))
    {
    }
    return old;
}

template<typename T>
inline T atomicMin(T *address, T val)
{
    std::atomic_ref<T> ref(*address);
    T old = ref.load(std::memory_order_relaxed);
    while(val < old && !ref.compare_exchange_weak(old, val, std::memory_order_relaxed))
    {
    }
    return old;
}

template<typename T>
inline T atomicOr(T *address, T val)
{
    return std::atomic_ref<T>(*address).fetch_or(val, std::memory_order_relaxed);
}

template<typename T>
inline T atomicAnd(T *address, T val)
{
    return std::atomic_ref<T>(*address).fetch_and(val, std::memory_order_relaxed);
}

template<typename T>
inline T atomicXor(T *address, T val)
{
    return std::atomic_ref<T>(*address).fetch_xor(val, std::memory_order_relaxed);
}

// ((old >= val) ? 0 : (old + 1)), as specified for atomicInc.
inline unsigned atomicInc(unsigned *address, unsigned val)
{
    std::atomic_ref<unsigned> ref(*address);
    unsigned old = ref.load(std::memory_order_relaxed);
    while(!ref.compare_exchange_weak(old, old >= val ? 0u : old + 1u, std::memory_order_relaxed))
    {
    }
    return old;
}

// ---- integer intrinsics -------------------------------------------------------
inline int __popc(unsigned x) { return std::popcount(x); }
inline int __popcll(unsigned long long x) { return std::popcount(x); }
inline int __clz(int x) { return std::countl_zero(static_cast<unsigned>(x)); }
inline int __clzll(long long x) { return std::countl_zero(static_cast<unsigned long long>(x)); }
inline int __ffs(int x) { return x == 0 ? 0 : std::countr_zero(static_cast<unsigned>(x)) + 1; }
inline int __ffsll(long long x) { return x == 0 ? 0 : std::countr_zero(static_cast<unsigned long long>(x)) + 1; }

inline unsigned __brev(unsigned x)
{
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    return (x >> 16) | (x << 16);
}

inline unsigned __umulhi(unsigned a, unsigned b)
{
    return static_cast<unsigned>((static_cast<unsigned long long>(a) * b) >> 32);
}
inline int __mul24(int a, int b) { return ((a << 8) >> 8) * ((b << 8) >> 8); }
inline unsigned __umul24(unsigned a, unsigned b) { return (a & 0xffffffu) * (b & 0xffffffu); }

// ---- bit reinterpretation --------------------------------------------------------
inline int __float_as_int(float x) { return std::bit_cast<int>(x); }
inline unsigned __float_as_uint(float x) { return std::bit_cast<unsigned>(x); }
inline float __int_as_float(int x) { return std::bit_cast<float>(x); }
inline float __uint_as_float(unsigned x) { return std::bit_cast<float>(x); }
inline long long __double_as_longlong(double x) { return std::bit_cast<long long>(x); }
inline double __longlong_as_double(long long x) { return std::bit_cast<double>(x); }

// ---- min / max (device overloads) ------------------------------------------------
inline int min(int a, int b) { return a < b ? a : b; }
inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
inline unsigned min(int a, unsigned b) { return min(static_cast<unsigned>(a), b); }
inline unsigned min(unsigned a, int b) { return min(a, static_cast<unsigned>(b)); }
inline long long min(long long a, long long b) { return a < b ? a : b; }
inline unsigned long long min(unsigned long long a, unsigned long long b) { return a < b ? a : b; }
inline long min(long a, long b) { return a < b ? a : b; }
inline unsigned long min(unsigned long a, unsigned long b) { return a < b ? a : b; }
inline float min(float a, float b) { return fminf(a, b); }
inline double min(double a, double b) { return fmin(a, b); }
inline int max(int a, int b) { return a > b ? a : b; }
inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
inline unsigned max(int a, unsigned b) { return max(static_cast<unsigned>(a), b); }
inline unsigned max(unsigned a, int b) { return max(a, static_cast<unsigned>(b)); }
inline long long max(long long a, long long b) { return a > b ? a : b; }
inline unsigned long long max(unsigned long long a, unsigned long long b) { return a > b ? a : b; }
inline long max(long a, long b) { return a > b ? a : b; }
inline unsigned long max(unsigned long a, unsigned long b) { return a > b ? a : b; }
inline float max(float a, float b) { return fmaxf(a, b); }
inline double max(double a, double b) { return fmax(a, b); }

// ---- float intrinsics ------------------------------------------------------------
inline float __expf(float x) { return std::exp(x); }
inline float __exp10f(float x) { return std::pow(10.0f, x); }
inline float __logf(float x) { return std::log(x); }
inline float __log2f(float x) { return std::log2(x); }
inline float __log10f(float x) { return std::log10(x); }
inline float __powf(float x, float y) { return std::pow(x, y); }
inline float __sinf(float x) { return std::sin(x); }
inline float __cosf(float x) { return std::cos(x); }
inline float __tanf(float x) { return std::tan(x); }
inline void __sincosf(float x, float *s, float *c)
{
    *s = std::sin(x);
    *c = std::cos(x);
}
inline void sincosf(float x, float *s, float *c) { __sincosf(x, s, c); }
inline float __fdividef(float a, float b) { return a / b; }
inline float __frcp_rn(float x) { return 1.0f / x; }
inline float __fsqrt_rn(float x) { return std::sqrt(x); }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
inline float __saturatef(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline double rsqrt(double x) { return 1.0 / std::sqrt(x); }
inline float rcbrtf(float x) { return 1.0f / std::cbrt(x); }
inline float exp10f(float x) { return std::pow(10.0f, x); }
inline int __float2int_rn(float x) { return static_cast<int>(std::nearbyint(x)); }
inline int __float2int_rd(float x) { return static_cast<int>(std::floor(x)); }
inline int __float2int_ru(float x) { return static_cast<int>(std::ceil(x)); }
inline int __float2int_rz(float x) { return static_cast<int>(x); }
inline unsigned __float2uint_rn(float x) { return static_cast<unsigned>(std::nearbyint(x)); }
inline unsigned __float2uint_rd(float x) { return static_cast<unsigned>(std::floor(x)); }
inline float __int2float_rn(int x) { return static_cast<float>(x); }
inline float __uint2float_rn(unsigned x) { return static_cast<float>(x); }

// ---- read-only data cache -------------------------------------------------------
template<typename T>
inline T __ldg(const T *ptr)
{
    return *ptr;
}

// ---- timing ----------------------------------------------------------------------
inline long long clock64()
{
    static std::atomic<long long> ticks{0};
    return ticks.fetch_add(1, std::memory_order_relaxed);
}

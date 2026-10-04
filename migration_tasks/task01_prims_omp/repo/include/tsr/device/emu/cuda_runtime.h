// Tessera - emulated device backend: CUDA runtime API subset.
//
// Host-side API with the same names and semantics as the CUDA runtime for the
// calls Tessera uses. Device memory is host memory; streams are serial queues
// that execute work eagerly; events record host timestamps.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "tsr/device/emu/runtime.h"
#include "tsr/device/emu/vector_types.h"

enum cudaError_t : int
{
    cudaSuccess                    = 0,
    cudaErrorInvalidValue          = 1,
    cudaErrorMemoryAllocation      = 2,
    cudaErrorInitializationError   = 3,
    cudaErrorInvalidConfiguration  = 9,
    cudaErrorInvalidDevicePointer  = 17,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorNotReady              = 600,
    cudaErrorNotSupported          = 801,
    cudaErrorUnknown               = 999,
};

enum cudaMemcpyKind : int
{
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4,
};

enum cudaFuncAttribute : int
{
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
};

struct CUstream_st;
struct CUevent_st;
using cudaStream_t = CUstream_st *;
using cudaEvent_t  = CUevent_st *;

struct cudaDeviceProp
{
    char name[256];
    std::size_t totalGlobalMem;
    std::size_t sharedMemPerBlock;
    std::size_t sharedMemPerBlockOptin;
    int warpSize;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int multiProcessorCount;
    int major;
    int minor;
};

// --- error handling -------------------------------------------------------
const char *cudaGetErrorString(cudaError_t error);
const char *cudaGetErrorName(cudaError_t error);
cudaError_t cudaGetLastError();
cudaError_t cudaPeekAtLastError();

namespace tsr::emu
{
// Records a launch/runtime failure in the calling thread's sticky slot.
void set_last_error(cudaError_t error);

// Number of live allocations made through cudaMalloc/cudaMallocHost.
std::size_t live_allocations();
} // namespace tsr::emu

// --- device management ----------------------------------------------------
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
cudaError_t cudaDeviceSynchronize();

// --- memory ---------------------------------------------------------------
cudaError_t cudaMalloc(void **ptr, std::size_t bytes);
cudaError_t cudaMallocHost(void **ptr, std::size_t bytes);
cudaError_t cudaFree(void *ptr);
cudaError_t cudaFreeHost(void *ptr);
cudaError_t cudaMemcpy(void *dst, const void *src, std::size_t bytes, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream = nullptr);
cudaError_t cudaMemset(void *dst, int value, std::size_t bytes);
cudaError_t cudaMemsetAsync(void *dst, int value, std::size_t bytes, cudaStream_t stream = nullptr);

template<typename T>
inline cudaError_t cudaMalloc(T **ptr, std::size_t bytes)
{
    return cudaMalloc(reinterpret_cast<void **>(ptr), bytes);
}

template<typename T>
inline cudaError_t cudaMallocHost(T **ptr, std::size_t bytes)
{
    return cudaMallocHost(reinterpret_cast<void **>(ptr), bytes);
}

// __constant__ / __device__ symbols are plain globals on this backend.
template<typename T>
inline cudaError_t cudaMemcpyToSymbol(T &symbol, const void *src, std::size_t bytes, std::size_t offset = 0,
                                      cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
    (void)kind;
    if(offset + bytes > sizeof(T))
    {
        return cudaErrorInvalidValue;
    }
    std::memcpy(reinterpret_cast<char *>(&symbol) + offset, src, bytes);
    return cudaSuccess;
}

template<typename T>
inline cudaError_t cudaMemcpyFromSymbol(void *dst, const T &symbol, std::size_t bytes, std::size_t offset = 0,
                                        cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
{
    (void)kind;
    if(offset + bytes > sizeof(T))
    {
        return cudaErrorInvalidValue;
    }
    std::memcpy(dst, reinterpret_cast<const char *>(&symbol) + offset, bytes);
    return cudaSuccess;
}

// --- streams and events ---------------------------------------------------
cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags = 0);
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = nullptr);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

constexpr unsigned cudaStreamNonBlocking = 0x01;

// --- kernel attributes ----------------------------------------------------
template<typename... KArgs>
inline cudaError_t cudaFuncSetAttribute(void (*kernel)(KArgs...), cudaFuncAttribute attr, int value)
{
    if(attr != cudaFuncAttributeMaxDynamicSharedMemorySize || value < 0
       || value > tsr::emu::limits().shared_mem_per_block_optin)
    {
        return cudaErrorInvalidValue;
    }
    tsr::emu::set_max_dynamic_smem(reinterpret_cast<const void *>(kernel), value);
    return cudaSuccess;
}

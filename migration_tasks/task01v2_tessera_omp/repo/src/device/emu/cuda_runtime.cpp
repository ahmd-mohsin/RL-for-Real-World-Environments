// Tessera - emulated device backend: CUDA runtime API subset.
#include "tsr/device/emu/cuda_runtime.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>
#include <unordered_map>

struct CUstream_st
{
    unsigned flags = 0;
};

struct CUevent_st
{
    std::chrono::steady_clock::time_point stamp{};
    bool recorded = false;
};

namespace
{
thread_local cudaError_t tl_last_error = cudaSuccess;

constexpr std::size_t kAllocAlignment = 256; // matches cudaMalloc's guarantee

std::mutex g_alloc_mutex;
std::unordered_map<void *, std::size_t> g_allocations;

cudaError_t record(cudaError_t e)
{
    if(e != cudaSuccess)
    {
        tl_last_error = e;
    }
    return e;
}

cudaError_t tracked_alloc(void **ptr, std::size_t bytes)
{
    if(ptr == nullptr)
    {
        return record(cudaErrorInvalidValue);
    }
    if(bytes == 0)
    {
        *ptr = nullptr;
        return cudaSuccess;
    }
    const std::size_t rounded = (bytes + kAllocAlignment - 1) / kAllocAlignment * kAllocAlignment;
    void *p                   = std::aligned_alloc(kAllocAlignment, rounded);
    if(p == nullptr)
    {
        *ptr = nullptr;
        return record(cudaErrorMemoryAllocation);
    }
    {
        const std::scoped_lock lock(g_alloc_mutex);
        g_allocations.emplace(p, bytes);
    }
    *ptr = p;
    return cudaSuccess;
}

cudaError_t tracked_free(void *ptr)
{
    if(ptr == nullptr)
    {
        return cudaSuccess;
    }
    {
        const std::scoped_lock lock(g_alloc_mutex);
        const auto it = g_allocations.find(ptr);
        if(it == g_allocations.end())
        {
            return record(cudaErrorInvalidDevicePointer);
        }
        g_allocations.erase(it);
    }
    std::free(ptr);
    return cudaSuccess;
}
} // namespace

namespace tsr::emu
{
void set_last_error(cudaError_t error) { record(error); }

std::size_t live_allocations()
{
    const std::scoped_lock lock(g_alloc_mutex);
    return g_allocations.size();
}
} // namespace tsr::emu

const char *cudaGetErrorName(cudaError_t error)
{
    switch(error)
    {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError: return "cudaErrorInitializationError";
    case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";
    case cudaErrorNotReady: return "cudaErrorNotReady";
    case cudaErrorNotSupported: return "cudaErrorNotSupported";
    case cudaErrorUnknown: return "cudaErrorUnknown";
    }
    return "cudaErrorUnknown";
}

const char *cudaGetErrorString(cudaError_t error)
{
    switch(error)
    {
    case cudaSuccess: return "no error";
    case cudaErrorInvalidValue: return "invalid argument";
    case cudaErrorMemoryAllocation: return "out of memory";
    case cudaErrorInitializationError: return "initialization error";
    case cudaErrorInvalidConfiguration: return "invalid configuration argument";
    case cudaErrorInvalidDevicePointer: return "invalid device pointer";
    case cudaErrorInvalidMemcpyDirection: return "invalid copy direction for memcpy";
    case cudaErrorInvalidResourceHandle: return "invalid resource handle";
    case cudaErrorNotReady: return "device not ready";
    case cudaErrorNotSupported: return "operation not supported";
    case cudaErrorUnknown: return "unknown error";
    }
    return "unrecognized error code";
}

cudaError_t cudaGetLastError()
{
    const cudaError_t e = tl_last_error;
    tl_last_error       = cudaSuccess;
    return e;
}

cudaError_t cudaPeekAtLastError() { return tl_last_error; }

cudaError_t cudaGetDeviceCount(int *count)
{
    if(count == nullptr)
    {
        return record(cudaErrorInvalidValue);
    }
    *count = 1;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) { return device == 0 ? cudaSuccess : record(cudaErrorInvalidValue); }

cudaError_t cudaGetDevice(int *device)
{
    if(device == nullptr)
    {
        return record(cudaErrorInvalidValue);
    }
    *device = 0;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
{
    if(prop == nullptr || device != 0)
    {
        return record(cudaErrorInvalidValue);
    }
    const tsr::emu::DeviceLimits &l = tsr::emu::limits();
    *prop                           = cudaDeviceProp{};
    std::snprintf(prop->name, sizeof(prop->name), "Tessera emulated device (%d workers)", tsr::emu::worker_count());
    prop->totalGlobalMem          = std::size_t{16} << 30;
    prop->sharedMemPerBlock       = static_cast<std::size_t>(l.shared_mem_per_block);
    prop->sharedMemPerBlockOptin  = static_cast<std::size_t>(l.shared_mem_per_block_optin);
    prop->warpSize                = l.warp_size;
    prop->maxThreadsPerBlock      = l.max_threads_per_block;
    prop->maxThreadsDim[0]        = 1024;
    prop->maxThreadsDim[1]        = 1024;
    prop->maxThreadsDim[2]        = 64;
    prop->maxGridSize[0]          = l.max_grid_x;
    prop->maxGridSize[1]          = l.max_grid_yz;
    prop->maxGridSize[2]          = l.max_grid_yz;
    prop->multiProcessorCount     = tsr::emu::worker_count();
    prop->major                   = 8;
    prop->minor                   = 0;
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize() { return cudaSuccess; } // all work is synchronous

cudaError_t cudaMalloc(void **ptr, std::size_t bytes) { return tracked_alloc(ptr, bytes); }
cudaError_t cudaMallocHost(void **ptr, std::size_t bytes) { return tracked_alloc(ptr, bytes); }
cudaError_t cudaFree(void *ptr) { return tracked_free(ptr); }
cudaError_t cudaFreeHost(void *ptr) { return tracked_free(ptr); }

cudaError_t cudaMemcpy(void *dst, const void *src, std::size_t bytes, cudaMemcpyKind kind)
{
    if(kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault)
    {
        return record(cudaErrorInvalidMemcpyDirection);
    }
    if(bytes == 0)
    {
        return cudaSuccess;
    }
    if(dst == nullptr || src == nullptr)
    {
        return record(cudaErrorInvalidValue);
    }
    std::memmove(dst, src, bytes);
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t)
{
    return cudaMemcpy(dst, src, bytes, kind);
}

cudaError_t cudaMemset(void *dst, int value, std::size_t bytes)
{
    if(bytes == 0)
    {
        return cudaSuccess;
    }
    if(dst == nullptr)
    {
        return record(cudaErrorInvalidValue);
    }
    std::memset(dst, value, bytes);
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void *dst, int value, std::size_t bytes, cudaStream_t) { return cudaMemset(dst, value, bytes); }

cudaError_t cudaStreamCreate(cudaStream_t *stream) { return cudaStreamCreateWithFlags(stream, 0); }

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned flags)
{
    if(stream == nullptr)
    {
        return record(cudaErrorInvalidValue);
    }
    *stream          = new(std::nothrow) CUstream_st{};
    if(*stream == nullptr)
    {
        return record(cudaErrorMemoryAllocation);
    }
    (*stream)->flags = flags;
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    delete stream;
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return cudaSuccess; }

cudaError_t cudaEventCreate(cudaEvent_t *event)
{
    if(event == nullptr)
    {
        return record(cudaErrorInvalidValue);
    }
    *event = new(std::nothrow) CUevent_st{};
    return *event == nullptr ? record(cudaErrorMemoryAllocation) : cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    delete event;
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t)
{
    if(event == nullptr)
    {
        return record(cudaErrorInvalidResourceHandle);
    }
    event->stamp    = std::chrono::steady_clock::now();
    event->recorded = true;
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    return event == nullptr ? record(cudaErrorInvalidResourceHandle) : cudaSuccess;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    if(ms == nullptr || start == nullptr || end == nullptr)
    {
        return record(cudaErrorInvalidValue);
    }
    if(!start->recorded || !end->recorded)
    {
        return record(cudaErrorNotReady);
    }
    *ms = std::chrono::duration<float, std::milli>(end->stamp - start->stamp).count();
    return cudaSuccess;
}

// Tessera - device memory ownership.
//
// DeviceBuffer<T>  RAII owner of a typed device allocation.
// Arena            bump allocator for per-frame temporaries (sort scratch,
//                  scan carries, intersection lists). Allocations are 256-byte
//                  aligned and released all at once by reset().
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "tsr/device/backend.h"

namespace tsr
{
template<typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { resize(count); }
    DeviceBuffer(const DeviceBuffer &)            = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&other) noexcept { swap(other); }
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if(this != &other)
        {
            release();
            swap(other);
        }
        return *this;
    }
    ~DeviceBuffer() { release(); }

    // Reallocates only when growing; contents are not preserved.
    void resize(std::size_t count)
    {
        if(count <= capacity_)
        {
            size_ = count;
            return;
        }
        release();
        if(count > 0)
        {
            TSR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)));
        }
        size_     = count;
        capacity_ = count;
    }

    void release()
    {
        if(data_ != nullptr)
        {
            cudaFree(data_);
        }
        data_     = nullptr;
        size_     = 0;
        capacity_ = 0;
    }

    void zero(cudaStream_t stream = nullptr)
    {
        if(size_ > 0)
        {
            TSR_CUDA_CHECK(cudaMemsetAsync(data_, 0, size_ * sizeof(T), stream));
        }
    }

    void upload(const T *host, std::size_t count, cudaStream_t stream = nullptr)
    {
        resize(count);
        if(count > 0)
        {
            TSR_CUDA_CHECK(cudaMemcpyAsync(data_, host, count * sizeof(T), cudaMemcpyHostToDevice, stream));
        }
    }

    void upload(const std::vector<T> &host, cudaStream_t stream = nullptr) { upload(host.data(), host.size(), stream); }

    void download(T *host, std::size_t count, cudaStream_t stream = nullptr) const
    {
        TSR_CHECK(count <= size_, "download of " << count << " elements from a buffer of " << size_);
        if(count > 0)
        {
            TSR_CUDA_CHECK(cudaMemcpyAsync(host, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
            TSR_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    [[nodiscard]] std::vector<T> to_host(cudaStream_t stream = nullptr) const
    {
        std::vector<T> out(size_);
        download(out.data(), size_, stream);
        return out;
    }

    [[nodiscard]] T *data() noexcept { return data_; }
    [[nodiscard]] const T *data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return size_ * sizeof(T); }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    void swap(DeviceBuffer &other) noexcept
    {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }

private:
    T *data_              = nullptr;
    std::size_t size_     = 0;
    std::size_t capacity_ = 0;
};

class Arena
{
public:
    static constexpr std::size_t kAlignment = 256;

    Arena() = default;
    explicit Arena(std::size_t initial_bytes) { reserve(initial_bytes); }
    Arena(const Arena &)            = delete;
    Arena &operator=(const Arena &) = delete;

    // Discards every allocation. Memory is kept for reuse; if the previous
    // frame overflowed into extra chunks they are coalesced into one block.
    void reset();

    void reserve(std::size_t bytes);

    void *allocate_bytes(std::size_t bytes);

    template<typename T>
    T *allocate(std::size_t count)
    {
        return static_cast<T *>(allocate_bytes(count * sizeof(T)));
    }

    [[nodiscard]] std::size_t used() const noexcept { return used_; }
    [[nodiscard]] std::size_t high_water() const noexcept { return high_water_; }
    [[nodiscard]] std::size_t capacity() const noexcept;

private:
    static std::size_t align_up(std::size_t v) { return (v + kAlignment - 1) / kAlignment * kAlignment; }

    DeviceBuffer<std::uint8_t> main_;
    std::vector<DeviceBuffer<std::uint8_t>> overflow_;
    std::size_t offset_     = 0;
    std::size_t used_       = 0;
    std::size_t high_water_ = 0;
};

} // namespace tsr

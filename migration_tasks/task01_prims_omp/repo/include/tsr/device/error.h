// Tessera - device error handling. Included by tsr/device/backend.h after the
// backend's runtime API has been declared; do not include directly.
#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace tsr
{
class DeviceError : public std::runtime_error
{
public:
    DeviceError(cudaError_t code, const std::string &what) : std::runtime_error(what), code_(code) {}
    [[nodiscard]] cudaError_t code() const noexcept { return code_; }

private:
    cudaError_t code_;
};

class Error : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

namespace detail
{
[[noreturn]] inline void throw_device_error(cudaError_t code, const char *expr, const char *file, int line)
{
    std::ostringstream os;
    os << file << ":" << line << ": " << expr << " failed: " << cudaGetErrorName(code) << " ("
       << cudaGetErrorString(code) << ")";
    throw DeviceError(code, os.str());
}

[[noreturn]] inline void throw_error(const std::string &msg, const char *file, int line)
{
    std::ostringstream os;
    os << file << ":" << line << ": " << msg;
    throw Error(os.str());
}

bool sync_after_launch();
} // namespace detail

// Surfaces asynchronous launch errors. With TSR_SYNC_LAUNCHES=1 the stream is
// also synchronized so that faults are attributed to the right kernel.
void check_launch(cudaStream_t stream);

} // namespace tsr

#define TSR_CUDA_CHECK(expr)                                                     \
    do                                                                           \
    {                                                                            \
        const cudaError_t tsr_err_ = (expr);                                     \
        if(tsr_err_ != cudaSuccess)                                              \
        {                                                                        \
            ::tsr::detail::throw_device_error(tsr_err_, #expr, __FILE__, __LINE__); \
        }                                                                        \
    } while(false)

#define TSR_CHECK(cond, msg)                                                     \
    do                                                                           \
    {                                                                            \
        if(!(cond))                                                              \
        {                                                                        \
            std::ostringstream tsr_os_;                                          \
            tsr_os_ << "check failed: " #cond ": " << msg;                       \
            ::tsr::detail::throw_error(tsr_os_.str(), __FILE__, __LINE__);       \
        }                                                                        \
    } while(false)

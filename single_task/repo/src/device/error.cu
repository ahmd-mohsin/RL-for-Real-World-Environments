// Tessera - launch error propagation.
#include <cstdlib>
#include <cstring>

#include "tsr/device/backend.h"

namespace tsr
{
namespace detail
{
bool sync_after_launch()
{
    static const bool enabled = []
    {
        const char *v = std::getenv("TSR_SYNC_LAUNCHES");
        return v != nullptr && std::strcmp(v, "0") != 0 && *v != '\0';
    }();
    return enabled;
}
} // namespace detail

void check_launch(cudaStream_t stream)
{
    const cudaError_t launch_err = cudaGetLastError();
    if(launch_err != cudaSuccess)
    {
        detail::throw_device_error(launch_err, "kernel launch", __FILE__, __LINE__);
    }
    if(detail::sync_after_launch())
    {
        TSR_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}
} // namespace tsr

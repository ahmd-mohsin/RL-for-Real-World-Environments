// Tessera - parallel rendering of many views.
//
// Fans views out over a host thread pool. Each worker owns a Renderer (its
// own stream, scratch arena and copy of the scene), the way a multi-GPU
// renderer would own one context per device, so views never share mutable
// state. Results are returned in camera order regardless of completion order.
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "tsr/render/renderer.h"
#include "tsr/sched/thread_pool.h"

namespace tsr
{
class FrameScheduler
{
public:
    FrameScheduler(const Scene &scene, const RenderSettings &settings, std::size_t workers = 0);
    ~FrameScheduler();

    FrameScheduler(const FrameScheduler &)            = delete;
    FrameScheduler &operator=(const FrameScheduler &) = delete;

    std::vector<Frame> render_all(const std::vector<Camera> &cameras);

    // Calls on_frame(index, frame) from worker threads as views finish.
    // Blocks until every view is done; the first exception is rethrown.
    void render_each(const std::vector<Camera> &cameras, const std::function<void(std::size_t, Frame &&)> &on_frame);

    [[nodiscard]] std::size_t worker_count() const { return pool_.get_thread_count(); }

private:
    Renderer &renderer_for_this_thread();

    const Scene &scene_;
    RenderSettings settings_;
    sched::ThreadPool pool_;
    std::vector<std::unique_ptr<Renderer>> renderers_; // indexed by pool thread
    std::vector<std::once_flag> init_flags_;
};

} // namespace tsr

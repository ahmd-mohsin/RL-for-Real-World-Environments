// Tessera - parallel rendering of many views.
#include "tsr/sched/frame_scheduler.h"

namespace tsr
{
FrameScheduler::FrameScheduler(const Scene &scene, const RenderSettings &settings, std::size_t workers)
    : scene_(scene), settings_(settings), pool_(workers), renderers_(pool_.get_thread_count()), init_flags_(pool_.get_thread_count())
{
}

FrameScheduler::~FrameScheduler()
{
    pool_.wait();
}

Renderer &FrameScheduler::renderer_for_this_thread()
{
    const auto idx = sched::this_thread::get_index();
    TSR_CHECK(idx.has_value() && sched::this_thread::get_pool() == &pool_, "renderer requested outside the scheduler's pool");
    std::call_once(init_flags_[*idx],
                   [&]
                   {
                       auto r = std::make_unique<Renderer>(settings_);
                       r->set_scene(scene_);
                       renderers_[*idx] = std::move(r);
                   });
    return *renderers_[*idx];
}

void FrameScheduler::render_each(const std::vector<Camera> &cameras, const std::function<void(std::size_t, Frame &&)> &on_frame)
{
    sched::multi_future<void> done;
    done.reserve(cameras.size());
    for(std::size_t i = 0; i < cameras.size(); ++i)
    {
        done.push_back(pool_.submit_task(
            [this, &cameras, &on_frame, i]
            {
                Frame f = renderer_for_this_thread().render(cameras[i]);
                on_frame(i, std::move(f));
            }
        ));
    }
    // Wait for all views before propagating the first error, so no task
    // outlives the references it captured.
    done.wait();
    done.get();
}

std::vector<Frame> FrameScheduler::render_all(const std::vector<Camera> &cameras)
{
    std::vector<Frame> frames(cameras.size());
    std::mutex m;
    render_each(cameras,
                [&](std::size_t i, Frame &&f)
                {
                    const std::scoped_lock lock(m);
                    frames[i] = std::move(f);
                });
    return frames;
}
} // namespace tsr

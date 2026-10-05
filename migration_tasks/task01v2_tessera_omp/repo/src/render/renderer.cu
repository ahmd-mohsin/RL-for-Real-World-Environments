// Tessera - the hybrid splat + mesh renderer.
#include "tsr/prims/prims.h"
#include "tsr/render/renderer.h"

namespace tsr
{
namespace
{
constexpr std::size_t kInitialArenaBytes = std::size_t{4} << 20;

__global__ void visible_flags_kernel(std::uint32_t n, const int2 *__restrict__ radii, std::uint32_t *__restrict__ flags)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        const int2 r = radii[i];
        flags[i]     = (r.x > 0 && r.y > 0) ? 1u : 0u;
    }
}
} // namespace

Renderer::Renderer(const RenderSettings &settings) : settings_(settings), arena_(kInitialArenaBytes)
{
    TSR_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

Renderer::~Renderer()
{
    if(stream_ != nullptr)
    {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
}

void Renderer::set_scene(const Scene &scene)
{
    gaussians_.upload(scene.gaussians, stream_);
    mesh_.upload(scene.meshes, stream_);
    if(mesh_.n_faces > 0)
    {
        material_.initialize(scene.material, stream_);
    }
    background_ = scene.background;
    TSR_CUDA_CHECK(cudaStreamSynchronize(stream_));
    has_scene_ = true;
}

void Renderer::render_surface(const Camera &camera)
{
    const int W    = static_cast<int>(camera.width);
    const int H    = static_cast<int>(camera.height);
    mesh_overflow_ = false;
    if(mesh_.n_faces == 0)
    {
        clear_surface(W, H, background_, surface_, stream_);
        fragments_ = MeshFragments{};
        return;
    }
    transform_faces(mesh_, camera, faces_, stream_);
    bin_mesh_faces(faces_, W, H, settings_.mesh_binning, mesh_bins_, arena_, stream_);
    mesh_overflow_ = mesh_bins_.overflowed;
    rasterize_mesh_fine(faces_, mesh_bins_, W, H, settings_.mesh_raster, fragments_, stream_);
    shade_surface(material_, fragments_, faces_, camera, background_, surface_, arena_, stream_);
}

std::uint32_t Renderer::count_visible_splats()
{
    const std::uint32_t n = splats_.count;
    if(n == 0)
    {
        return 0;
    }
    std::uint32_t *flags = arena_.allocate<std::uint32_t>(n);
    std::uint32_t *total = arena_.allocate<std::uint32_t>(1);
    launch_linear(visible_flags_kernel, n, stream_, n, splats_.radii.data(), flags);
    prims::reduce(flags, n, total, prims::ReduceOp::Sum, arena_, stream_);
    std::uint32_t host_total = 0;
    TSR_CUDA_CHECK(cudaMemcpyAsync(&host_total, total, sizeof(host_total), cudaMemcpyDeviceToHost, stream_));
    TSR_CUDA_CHECK(cudaStreamSynchronize(stream_));
    return host_total;
}

Frame Renderer::render(const Camera &camera)
{
    TSR_CHECK(has_scene_, "Renderer::render called before set_scene");
    TSR_CHECK(camera.width > 0 && camera.height > 0, "camera has an empty image");
    arena_.reset();

    // opaque surface layer
    render_surface(camera);

    // splats
    project_gaussians(gaussians_, camera, settings_.projection, splats_, stream_);
    compute_splat_colors(gaussians_, camera, splats_, stream_);
    bin_splats(splats_, camera, settings_.binning, binning_, arena_, stream_);

    // composite
    rasterize_splats(splats_, binning_, surface_, camera, settings_.splat_raster, composite_, stream_);

    // post
    const std::size_t n_pixels = static_cast<std::size_t>(camera.width) * camera.height;
    ldr_.resize(n_pixels);
    Frame frame;
    frame.width  = static_cast<int>(camera.width);
    frame.height = static_cast<int>(camera.height);
    tonemap(composite_.color.data(), frame.width, frame.height, settings_.tonemap, ldr_.data(), &frame.stats.tonemap, arena_, stream_);

    frame.stats.visible_splats       = count_visible_splats();
    frame.stats.intersections        = binning_.n_isects;
    frame.stats.surface_pixels       = surface_.covered_pixels;
    frame.stats.mesh_bins_overflowed = mesh_overflow_;
    frame.linear                     = composite_.color.to_host(stream_);
    frame.alpha                      = composite_.alpha.to_host(stream_);
    frame.rgba8                      = ldr_.to_host(stream_);
    frame.stats.scratch_bytes        = arena_.high_water();
    return frame;
}
} // namespace tsr

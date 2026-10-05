// Offload probe: proves that every pipeline stage computes on the device.
//
// The probe interposes the libomptarget entry points that clang emits for
// target regions and data movement, plus the OpenMP device memory API, and
// counts, per stage call:
//   kernels  number of target regions launched
//   d2h      bytes copied from the device to the host
// A port that downloads its input, computes on the host and uploads the
// result is caught by the d2h budget; one that never offloads is caught by
// the kernel count. Output: one JSON line per primitive on stdout.
#include <dlfcn.h>
#include <omp.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "scenes.h"
#include "tsr/render/renderer.h"

namespace
{
std::atomic<long long> g_kernels{0};
std::atomic<long long> g_d2h{0};

constexpr std::int64_t kMapFrom = 0x002;

void count_from(std::int32_t n, const std::int64_t *sizes, const std::int64_t *types)
{
    if(sizes == nullptr || types == nullptr)
    {
        return;
    }
    for(std::int32_t i = 0; i < n; ++i)
    {
        if(types[i] & kMapFrom)
        {
            g_d2h += sizes[i];
        }
    }
}

template<typename F>
F real(const char *name)
{
    void *p = dlsym(RTLD_NEXT, name);
    if(p == nullptr)
    {
        std::fprintf(stderr, "offload_probe: cannot resolve %s\n", name);
        std::abort();
    }
    return reinterpret_cast<F>(p);
}

// Leading fields of libomptarget's KernelArgsTy (stable since LLVM 17).
struct KernelArgsPrefix
{
    std::uint32_t Version;
    std::uint32_t NumArgs;
    void **ArgBasePtrs;
    void **ArgPtrs;
    std::int64_t *ArgSizes;
    std::int64_t *ArgTypes;
};
} // namespace

extern "C" {
int __tgt_target_kernel(void *loc, std::int64_t dev, std::int32_t teams, std::int32_t threads, void *host_ptr, void *args)
{
    using Fn       = int (*)(void *, std::int64_t, std::int32_t, std::int32_t, void *, void *);
    static Fn next = real<Fn>("__tgt_target_kernel");
    ++g_kernels;
    const auto *a = static_cast<const KernelArgsPrefix *>(args);
    count_from(static_cast<std::int32_t>(a->NumArgs), a->ArgSizes, a->ArgTypes);
    return next(loc, dev, teams, threads, host_ptr, args);
}

#define TSR_PROBE_MAPPER(NAME)                                                                                       \
    void NAME(void *loc, std::int64_t dev, std::int32_t n, void **base, void **ptrs, std::int64_t *sizes,           \
              std::int64_t *types, void *names, void **mappers)                                                    \
    {                                                                                                                \
        using Fn       = void (*)(void *, std::int64_t, std::int32_t, void **, void **, std::int64_t *, std::int64_t *, void *, void **); \
        static Fn next = real<Fn>(#NAME);                                                                            \
        count_from(n, sizes, types);                                                                                 \
        next(loc, dev, n, base, ptrs, sizes, types, names, mappers);                                                 \
    }

TSR_PROBE_MAPPER(__tgt_target_data_end_mapper)
TSR_PROBE_MAPPER(__tgt_target_data_update_mapper)

#define TSR_PROBE_MAPPER_NOWAIT(NAME)                                                                                \
    void NAME(void *loc, std::int64_t dev, std::int32_t n, void **base, void **ptrs, std::int64_t *sizes,           \
              std::int64_t *types, void *names, void **mappers, std::int32_t dn, void *dl, std::int32_t ndn, void *ndl) \
    {                                                                                                                \
        using Fn       = void (*)(void *, std::int64_t, std::int32_t, void **, void **, std::int64_t *, std::int64_t *, void *, void **, std::int32_t, void *, std::int32_t, void *); \
        static Fn next = real<Fn>(#NAME);                                                                            \
        count_from(n, sizes, types);                                                                                 \
        next(loc, dev, n, base, ptrs, sizes, types, names, mappers, dn, dl, ndn, ndl);                               \
    }

TSR_PROBE_MAPPER_NOWAIT(__tgt_target_data_end_nowait_mapper)
TSR_PROBE_MAPPER_NOWAIT(__tgt_target_data_update_nowait_mapper)

int omp_target_memcpy(void *dst, const void *src, size_t length, size_t dst_off, size_t src_off, int dst_dev, int src_dev)
{
    using Fn       = int (*)(void *, const void *, size_t, size_t, size_t, int, int);
    static Fn next = real<Fn>("omp_target_memcpy");
    if(dst_dev == omp_get_initial_device() && src_dev != dst_dev)
    {
        g_d2h += static_cast<long long>(length);
    }
    return next(dst, src, length, dst_off, src_off, dst_dev, src_dev);
}

int omp_target_memcpy_rect(void *dst, const void *src, size_t elem, int ndims, const size_t *volume, const size_t *dst_off,
                           const size_t *src_off, const size_t *dst_dims, const size_t *src_dims, int dst_dev, int src_dev)
{
    using Fn       = int (*)(void *, const void *, size_t, int, const size_t *, const size_t *, const size_t *, const size_t *, const size_t *, int, int);
    static Fn next = real<Fn>("omp_target_memcpy_rect");
    if(dst != nullptr && volume != nullptr && dst_dev == omp_get_initial_device() && src_dev != dst_dev)
    {
        long long total = static_cast<long long>(elem);
        for(int d = 0; d < ndims; ++d)
        {
            total *= static_cast<long long>(volume[d]);
        }
        g_d2h += total;
    }
    return next(dst, src, elem, ndims, volume, dst_off, src_off, dst_dims, src_dims, dst_dev, src_dev);
}
} // extern "C"

namespace
{
template<typename F>
void section(const char *name, long long budget, F &&f)
{
    const long long k0 = g_kernels.load();
    const long long d0 = g_d2h.load();
    f();
    std::printf("{\"stage\": \"%s\", \"kernels\": %lld, \"d2h\": %lld, \"budget\": %lld}\n", name, g_kernels.load() - k0, g_d2h.load() - d0, budget);
    std::fflush(stdout);
}
} // namespace

int main()
{
    using namespace tsr;
    const std::uint32_t W = 640, H = 360;
    const Scene scene     = hidden::hybrid_scene(60000, 80, W, H, 4242);
    const Camera &cam     = scene.cameras[0];
    RenderSettings settings;
    settings.mesh_binning.bin_size = 32;
    Arena arena(std::size_t{32} << 20);

    DeviceGaussians gaussians;
    gaussians.upload(scene.gaussians, nullptr);
    DeviceMesh mesh;
    mesh.upload(scene.meshes, nullptr);
    NeuralMaterialModel material;
    material.initialize(scene.material, nullptr);
    ViewFaces faces;
    MeshBins bins;
    MeshFragments frags;
    SurfaceLayer surface;
    ProjectedSplats splats;
    TileBinning binning;
    SplatRasterOutput comp;
    DeviceBuffer<uchar4> ldr(static_cast<std::size_t>(W) * H);
    TonemapStats tstats;

    section("transform_faces", 64, [&] { transform_faces(mesh, cam, faces, nullptr); });
    section("bin_mesh_faces", 64, [&] { arena.reset(); bin_mesh_faces(faces, W, H, settings.mesh_binning, bins, arena, nullptr); });
    section("rasterize_mesh_fine", 64, [&] { rasterize_mesh_fine(faces, bins, W, H, settings.mesh_raster, frags, nullptr); });
    section("shade_surface", 64, [&] { arena.reset(); shade_surface(material, frags, faces, cam, scene.background, surface, arena, nullptr); });
    section("project_gaussians", 64, [&] { project_gaussians(gaussians, cam, settings.projection, splats, nullptr); });
    section("compute_splat_colors", 64, [&] { compute_splat_colors(gaussians, cam, splats, nullptr); });
    section("bin_splats", 64, [&] { arena.reset(); bin_splats(splats, cam, settings.binning, binning, arena, nullptr); });
    section("rasterize_splats", 64, [&] { rasterize_splats(splats, binning, surface, cam, settings.splat_raster, comp, nullptr); });
    section("tonemap", 4096, [&] { arena.reset(); tonemap(comp.color.data(), W, H, settings.tonemap, ldr.data(), &tstats, arena, nullptr); });
    {
        Renderer r(settings);
        r.set_scene(scene);
        (void)r.render(cam); // warm up
        // the frame's own outputs (linear, alpha, rgba8) plus a small allowance
        const long long frame_out = static_cast<long long>(W) * H * (sizeof(float3) + sizeof(float) + sizeof(uchar4));
        section("frame", frame_out + 8192, [&] { (void)r.render(cam); });
    }
    return 0;
}

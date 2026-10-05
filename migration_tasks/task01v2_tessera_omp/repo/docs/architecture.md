# Architecture

## Frame data flow

```
Scene ──► Renderer::set_scene ──► DeviceGaussians, DeviceMesh, NeuralMaterialModel
                                  (uploaded once per renderer)

Renderer::render(camera)
 │
 ├─ surface layer (meshes)
 │    transform_faces        world → camera → pixel → non-square NDC   geom/mesh.cu
 │    bin_mesh_faces         bounding boxes, shared-memory bit masks   bin/mesh_bins.cu
 │    rasterize_mesh_fine    top-K faces per pixel, barycentrics       raster/mesh_raster.cu
 │    shade_surface          compact hits → hash grid → MLP → shade    render/surface.cu
 │                           (prims::compact_indices, encode/*)
 │
 ├─ splats (Gaussians)
 │    project_gaussians      EWA projection, culling, conics           geom/projection.cu
 │    compute_splat_colors   spherical harmonics                       geom/projection.cu
 │    bin_splats             count → inclusive scan → emit keys →      bin/tile_intersect.cu
 │                           sort_pairs → tile offsets                 (prims::inclusive_scan,
 │                                                                       prims::sort_pairs)
 ├─ composite
 │    rasterize_splats       per-tile front-to-back blending, stops    raster/splat_raster.cu
 │                           at the surface depth
 └─ post
      tonemap                log-average (reduce), histogram + scan    post/tonemap.cu
                             white point, Reinhard, sRGB
```

`FrameScheduler` (sched/frame_scheduler.cu) renders many cameras on a thread
pool; each pool thread lazily builds its own `Renderer`, so views never share
streams, scratch memory or intermediate buffers.

## Coordinate systems

| Space | Convention | Used by |
|---|---|---|
| world | right-handed, +y up | scene files |
| camera | +x right, +y down, +z forward | projection, depths, mesh z |
| pixel | origin at the top-left corner, pixel centres at +0.5 | splats, tiles, images |
| raster NDC | +x **left**, +y **up**, shorter side spans [-1, 1] | mesh binning and rasterization |

`geom/mesh_ndc.cuh` converts pixel coordinates to raster NDC. The mesh
rasterizer works on raster indices (which grow towards +x/+y NDC) and flips
back to image rows/columns only when it writes its outputs, so mesh bins are
not aligned with splat tiles.

## Intersection keys

Splat/tile intersections are sorted as 64-bit keys

```
 63            32+tile_bits  32                0
 [   unused    |  tile id    |  depth (float bits) ]
```

Depths are positive (near-plane culling), so their IEEE-754 bit patterns sort
like the values. `prims::sort_pairs` only needs the key bits below
`32 + tile_bits`. Short lists (<= 1024 intersections) are sorted by a single
shared-memory bitonic network ordered by (key, splat index); longer lists use
the stable LSD radix sort, which yields the same order because splat indices
are emitted in increasing order.

## Scratch memory

All per-frame temporaries come from the renderer's `Arena`, a bump allocator
over device memory that is reset at the start of every frame. If a frame
overflows the arena's block, the extra chunks are coalesced into one larger
block on the next reset.

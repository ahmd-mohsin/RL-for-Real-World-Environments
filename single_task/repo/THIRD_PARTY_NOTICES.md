# Third-party notices

Tessera contains code derived from the projects below. Derived files say so in
their header comment and keep the original copyright notices where the
licenses require it. Full license texts are in `third_party/LICENSES/`.

| Project | License | Used for |
|---|---|---|
| [gsplat](https://github.com/nerfstudio-project/gsplat) (Nerfstudio Team and contributors; NVIDIA CORPORATION & AFFILIATES) | Apache-2.0 | Gaussian projection math, spherical harmonics, tile intersection (SnugBox/AccuTile), tile offsets, splat rasterization: `include/tsr/geom/gaussian_math.cuh`, `src/geom/projection.cu`, `src/bin/tile_intersect.cu`, `src/raster/splat_raster.cu` |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) (Meta Platforms, Inc. and affiliates) | BSD-3-Clause | Non-square NDC helpers, coarse mesh binning, fine mesh rasterization: `include/tsr/geom/mesh_ndc.cuh`, `src/bin/mesh_bins.cu`, `src/raster/mesh_raster.cu` |
| [cuda-samples](https://github.com/NVIDIA/cuda-samples) (NVIDIA CORPORATION) | BSD-3-Clause | Vector math (`helper_math.h`), scan, reduction, histogram, bitonic sort and tiled matrix multiply: `include/tsr/math/vec.h`, `include/tsr/prims/block.cuh`, `src/prims/*.cu`, `src/encode/mlp.cu` |
| [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) (NVIDIA CORPORATION) | BSD-3-Clause | Multiresolution hash-grid encoding: `src/encode/hashgrid.cu` |
| [BS::thread_pool](https://github.com/bshoshany/thread-pool) (Barak Shoshany) | MIT | Host thread pool: `include/tsr/sched/thread_pool.h` |
| [doctest](https://github.com/doctest/doctest) (Viktor Kirilov) | MIT | Unit test framework (vendored unmodified in `third_party/doctest/`) |

## BSD-3-Clause notices

Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. (cuda-samples)

Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved. (tiny-cuda-nn)

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved. (PyTorch3D)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer; redistributions in binary
form must reproduce the above copyright notice, this list of conditions and
the following disclaimer in the documentation and/or other materials provided
with the distribution; neither the name of the copyright holder nor the names
of its contributors may be used to endorse or promote products derived from
this software without specific prior written permission. See the full texts
in `third_party/LICENSES/` for the warranty disclaimer.

## Apache-2.0 notice (gsplat)

Copyright 2025 the Regents of the University of California, Nerfstudio Team and
contributors. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Licensed under
the Apache License, Version 2.0; see `third_party/LICENSES/gsplat.txt`. The
derived files were modified for Tessera as described in their headers.

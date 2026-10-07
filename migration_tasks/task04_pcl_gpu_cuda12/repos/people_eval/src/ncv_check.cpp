// ncv_check: exercises the NVIDIA NCV / NPP-staging image primitives shipped inside
// pcl::gpu::people (integral images, decimation, rect std-dev, separable filters,
// resize, frame interpolation, stream compaction, transpose) and the Haar cascade
// object detector (multi-scale pipeline on synthetic cascades and images).
//
//   ncv_check --out DIR [--seed S]
//
// --seed S (S != 0) draws different input data and image sizes; S = 0 is the default set.
// Writes DIR/<case>.bin with each output buffer and DIR/result.json with the
// status code of every call and, where the library provides a host reference
// implementation, the maximum difference against it.
#include <pcl/gpu/people/NCV.hpp>
#include <pcl/gpu/people/NPP_staging.hpp>
#include <pcl/gpu/people/NCVHaarObjectDetection.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string g_out;
std::uint32_t g_seed = 0;
std::ostringstream g_json;
bool g_first = true;

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

std::uint32_t hash32(std::uint32_t x) {
  x ^= g_seed * 0x9e3779b1U;
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}

template <class T>
void save(const std::string& name, const std::vector<T>& v) {
  std::ofstream f(g_out + "/" + name + ".bin", std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}

void record(const std::string& name, int status, double maxdiff = -1) {
  g_json << (g_first ? "" : ", ") << "\"" << name << "\": {\"status\": " << status << ", \"max_diff_vs_host\": " << maxdiff << "}";
  g_first = false;
  std::cout << name << " status " << status << " max_diff_vs_host " << maxdiff << std::endl;
}

template <class T>
T* devAlloc(size_t bytes) {
  void* p = nullptr;
  check(cudaMalloc(&p, bytes), "cudaMalloc");
  check(cudaMemset(p, 0, bytes), "cudaMemset");
  return static_cast<T*>(p);
}

// ---------------------------------------------------------------- integral images
void testIntegral(cudaDeviceProp& prop, int W, int H, int offset, const std::string& tag) {
  const Ncv32u srcStep = ((W + offset + 255) / 256) * 256;
  std::vector<Ncv8u> h_src(srcStep * H, 0);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) h_src[y * srcStep + x + offset] = (Ncv8u)(hash32(y * 7919 + x) & 0xff);
  Ncv8u* d_srcBase = devAlloc<Ncv8u>(h_src.size());
  check(cudaMemcpy(d_srcBase, h_src.data(), h_src.size(), cudaMemcpyHostToDevice), "upload");
  Ncv8u* d_src = d_srcBase + offset;  // offset != 0: not texture-aligned
  NcvSize32u roi = {(Ncv32u)W, (Ncv32u)H};

  const Ncv32u dstStep = ((W + 1) * 4 + 255) / 256 * 256;
  Ncv32u bufSize = 0;
  int st = nppiStIntegralGetSize_8u32u(roi, &bufSize, prop);
  Ncv8u* buf = devAlloc<Ncv8u>(std::max<Ncv32u>(bufSize, 1));
  Ncv32u* d_int = devAlloc<Ncv32u>(dstStep * (H + 1));
  if (st == NPPST_SUCCESS) st = nppiStIntegral_8u32u_C1R(d_src, srcStep, d_int, dstStep, roi, buf, bufSize, prop);
  check(cudaDeviceSynchronize(), "integral");
  std::vector<Ncv32u> got(dstStep / 4 * (H + 1)), ref(got.size());
  check(cudaMemcpy(got.data(), d_int, got.size() * 4, cudaMemcpyDeviceToHost), "download");
  nppiStIntegral_8u32u_C1R_host(h_src.data() + offset, srcStep, ref.data(), dstStep, roi);
  double md = 0;
  for (int y = 0; y <= H; ++y)
    for (int x = 0; x <= W; ++x) md = std::max(md, std::abs((double)got[y * dstStep / 4 + x] - ref[y * dstStep / 4 + x]));
  save("integral_" + tag, got);
  record("integral_" + tag, st, md);

  // squared integral
  const Ncv32u sqStep = ((W + 1) * 8 + 255) / 256 * 256;
  Ncv32u sqBuf = 0;
  int st2 = nppiStSqrIntegralGetSize_8u64u(roi, &sqBuf, prop);
  Ncv8u* buf2 = devAlloc<Ncv8u>(std::max<Ncv32u>(sqBuf, 1));
  Ncv64u* d_sq = devAlloc<Ncv64u>(sqStep * (H + 1));
  if (st2 == NPPST_SUCCESS) st2 = nppiStSqrIntegral_8u64u_C1R(d_src, srcStep, d_sq, sqStep, roi, buf2, sqBuf, prop);
  check(cudaDeviceSynchronize(), "sqintegral");
  std::vector<Ncv64u> got2(sqStep / 8 * (H + 1)), ref2(got2.size());
  check(cudaMemcpy(got2.data(), d_sq, got2.size() * 8, cudaMemcpyDeviceToHost), "download");
  nppiStSqrIntegral_8u64u_C1R_host(h_src.data() + offset, srcStep, ref2.data(), sqStep, roi);
  double md2 = 0;
  for (int y = 0; y <= H; ++y)
    for (int x = 0; x <= W; ++x) md2 = std::max(md2, std::abs((double)got2[y * sqStep / 8 + x] - (double)ref2[y * sqStep / 8 + x]));
  save("sqintegral_" + tag, got2);
  record("sqintegral_" + tag, st2, md2);

  // rect std-dev over the integral images, through the read-only cache and through plain loads
  if (offset == 0) {
    NcvRect32u rect = {1, 2, 9, 7};
    NcvSize32u nroi = {(Ncv32u)(W - rect.x - rect.width), (Ncv32u)(H - rect.y - rect.height)};
    const Ncv32u normStep = (nroi.width * 4 + 255) / 256 * 256;
    std::vector<Ncv32f> ref3(normStep / 4 * nroi.height);
    nppiStRectStdDev_32f_C1R_host(ref.data(), dstStep, ref2.data(), sqStep, ref3.data(), normStep, nroi, rect, 1.0f);
    for (int tex = 0; tex < 2; ++tex) {
      Ncv32f* d_norm = devAlloc<Ncv32f>(normStep * nroi.height);
      int st3 = nppiStRectStdDev_32f_C1R(d_int, dstStep, d_sq, sqStep, d_norm, normStep, nroi, rect, 1.0f, tex != 0);
      check(cudaDeviceSynchronize(), "rectstddev");
      std::vector<Ncv32f> got3(ref3.size());
      check(cudaMemcpy(got3.data(), d_norm, got3.size() * 4, cudaMemcpyDeviceToHost), "download");
      double md3 = 0;
      for (Ncv32u y = 0; y < nroi.height; ++y)
        for (Ncv32u x = 0; x < nroi.width; ++x)
          md3 = std::max(md3, (double)std::abs(got3[y * normStep / 4 + x] - ref3[y * normStep / 4 + x]));
      save(std::string("rectstddev_") + (tex ? "tex" : "notex"), got3);
      record(std::string("rectstddev_") + (tex ? "tex" : "notex"), st3, md3);
      cudaFree(d_norm);
    }
  }
  cudaFree(d_srcBase); cudaFree(buf); cudaFree(d_int); cudaFree(buf2); cudaFree(d_sq);
}

// ---------------------------------------------------------------- decimate / transpose / compact
void testDecimateTransposeCompact(cudaDeviceProp& prop) {
  const int W = 333, H = 211;
  const Ncv32u step = ((W * 4 + 255) / 256) * 256;
  std::vector<Ncv32u> h(step / 4 * H);
  for (size_t i = 0; i < h.size(); ++i) h[i] = hash32((Ncv32u)i * 2654435761U);
  Ncv32u* d = devAlloc<Ncv32u>(h.size() * 4);
  check(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "upload");
  for (Ncv32u scale : {2u, 3u}) {
    for (int tex = 0; tex < 2; ++tex) {
      NcvSize32u roi = {(Ncv32u)W, (Ncv32u)H};
      const Ncv32u dstep = ((W / scale) * 4 + 255) / 256 * 256;
      Ncv32u* dd = devAlloc<Ncv32u>(dstep * (H / scale));
      int st = nppiStDecimate_32u_C1R(d, step, dd, dstep, roi, scale, tex != 0);
      check(cudaDeviceSynchronize(), "decimate");
      std::vector<Ncv32u> got(dstep / 4 * (H / scale)), ref(got.size());
      check(cudaMemcpy(got.data(), dd, got.size() * 4, cudaMemcpyDeviceToHost), "download");
      nppiStDecimate_32u_C1R_host(h.data(), step, ref.data(), dstep, roi, scale);
      double md = 0;
      for (Ncv32u y = 0; y < H / scale; ++y)
        for (Ncv32u x = 0; x < W / scale; ++x) md = std::max(md, std::abs((double)got[y * dstep / 4 + x] - ref[y * dstep / 4 + x]));
      std::string name = "decimate_s" + std::to_string(scale) + (tex ? "_tex" : "_notex");
      save(name, got);
      record(name, st, md);
      cudaFree(dd);
    }
  }
  // transpose
  {
    const Ncv32u tstep = ((H * 4 + 255) / 256) * 256;
    Ncv32u* dt = devAlloc<Ncv32u>(tstep * W);
    NcvSize32u roi = {(Ncv32u)W, (Ncv32u)H};
    int st = nppiStTranspose_32u_C1R(d, step, dt, tstep, roi);
    check(cudaDeviceSynchronize(), "transpose");
    std::vector<Ncv32u> got(tstep / 4 * W);
    check(cudaMemcpy(got.data(), dt, got.size() * 4, cudaMemcpyDeviceToHost), "download");
    double md = 0;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) md = std::max(md, std::abs((double)got[x * tstep / 4 + y] - h[y * step / 4 + x]));
    save("transpose", got);
    record("transpose", st, md);
    cudaFree(dt);
  }
  // compact: remove every element equal to 7 (about a quarter of them)
  {
    const Ncv32u n = 100003;
    std::vector<Ncv32u> v(n);
    for (Ncv32u i = 0; i < n; ++i) v[i] = (hash32(i) & 3) == 0 ? 7u : hash32(i + 77);
    Ncv32u* dv = devAlloc<Ncv32u>(n * 4);
    Ncv32u* dc = devAlloc<Ncv32u>(n * 4);
    check(cudaMemcpy(dv, v.data(), n * 4, cudaMemcpyHostToDevice), "upload");
    Ncv32u bs = 0;
    int st = nppsStCompactGetSize_32u(n, &bs, prop);
    Ncv8u* buf = devAlloc<Ncv8u>(std::max<Ncv32u>(bs, 1));
    Ncv32u* h_len = nullptr;
    check(cudaMallocHost(&h_len, sizeof(Ncv32u)), "pinned");
    *h_len = 0;
    if (st == NPPST_SUCCESS) st = nppsStCompact_32u(dv, n, dc, h_len, 7u, buf, bs, prop);
    check(cudaDeviceSynchronize(), "compact");
    std::vector<Ncv32u> got(*h_len), ref(n);
    if (*h_len) check(cudaMemcpy(got.data(), dc, *h_len * 4, cudaMemcpyDeviceToHost), "download");
    Ncv32u rlen = 0;
    nppsStCompact_32u_host(v.data(), n, ref.data(), &rlen, 7u);
    ref.resize(rlen);
    double md = got.size() == ref.size() ? 0 : 1e9;
    for (size_t i = 0; i < std::min(got.size(), ref.size()); ++i) md = std::max(md, std::abs((double)got[i] - ref[i]));
    save("compact", got);
    record("compact", st, md);
    cudaFree(dv); cudaFree(dc); cudaFree(buf); cudaFreeHost(h_len);
  }
  cudaFree(d);
}

// ---------------------------------------------------------------- float filters, resize, interpolation
std::vector<Ncv32f> floatImage(int W, int H, Ncv32u stepElems, float phase) {
  std::vector<Ncv32f> v(stepElems * H, 0.f);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      v[y * stepElems + x] = std::sin(0.071f * x + phase) * std::cos(0.053f * y - phase) + 0.25f * ((hash32(y * 4099 + x) & 255) / 255.f);
  return v;
}

void testFloat(cudaDeviceProp&) {
  const int W = 301, H = 177;
  const Ncv32u step = ((W * 4 + 255) / 256) * 256;
  std::vector<Ncv32f> img = floatImage(W, H, step / 4, 0.3f);
  Ncv32f* d_img = devAlloc<Ncv32f>(img.size() * 4);
  check(cudaMemcpy(d_img, img.data(), img.size() * 4, cudaMemcpyHostToDevice), "upload");
  NcvSize32u size = {(Ncv32u)W, (Ncv32u)H};

  // separable filters with mirror borders; large anchors push mirror indices out of range
  std::vector<Ncv32f> k = {0.05f, 0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f, 0.02f, 0.01f};
  Ncv32f* d_k = devAlloc<Ncv32f>(k.size() * 4);
  check(cudaMemcpy(d_k, k.data(), k.size() * 4, cudaMemcpyHostToDevice), "upload");
  for (int anchor : {0, 4, 8}) {
    for (int col = 0; col < 2; ++col) {
      Ncv32f* d_out = devAlloc<Ncv32f>(img.size() * 4);
      NcvRect32u roi = {0, 0, (Ncv32u)W, (Ncv32u)H};
      int st = col ? nppiStFilterColumnBorder_32f_C1R(d_img, size, step, d_out, size, step, roi, nppStBorderMirror, d_k, (Ncv32s)k.size(), anchor, 1.5f)
                   : nppiStFilterRowBorder_32f_C1R(d_img, size, step, d_out, size, step, roi, nppStBorderMirror, d_k, (Ncv32s)k.size(), anchor, 1.5f);
      check(cudaDeviceSynchronize(), "filter");
      std::vector<Ncv32f> got(img.size());
      check(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost), "download");
      std::string name = std::string(col ? "filter_col" : "filter_row") + "_a" + std::to_string(anchor);
      save(name, got);
      record(name, st);
      cudaFree(d_out);
    }
  }

  // separable filters on tiny ROIs at the start of the source buffer, where mirrored indices leave the
  // image: once inside guard bands (garbage around the image) and once in an allocation of exactly the image
  for (int guarded = 1; guarded >= 0; --guarded) {
    const size_t n = img.size(), G = guarded ? 256 : 0;  // 1 KiB: keeps the texture alignment
    std::vector<Ncv32f> host(n + 2 * G, 1.0e6f);
    std::copy(img.begin(), img.end(), host.begin() + G);
    Ncv32f* base = nullptr;
    check(cudaMalloc(&base, host.size() * 4), "cudaMalloc");
    check(cudaMemcpy(base, host.data(), host.size() * 4, cudaMemcpyHostToDevice), "upload");
    for (int anchor : {0, 4, 8})
      for (int col = 0; col < 2; ++col)
        for (NcvRect32u roi : {NcvRect32u{0, 0, 4, 3}, NcvRect32u{0, 0, 2, 5}}) {
          Ncv32f* d_out = devAlloc<Ncv32f>(n * 4);
          int st = col ? nppiStFilterColumnBorder_32f_C1R(base + G, size, step, d_out, size, step, roi, nppStBorderMirror, d_k, (Ncv32s)k.size(), anchor, 1.0f)
                       : nppiStFilterRowBorder_32f_C1R(base + G, size, step, d_out, size, step, roi, nppStBorderMirror, d_k, (Ncv32s)k.size(), anchor, 1.0f);
          check(cudaDeviceSynchronize(), "filter");
          std::vector<Ncv32f> got(n);
          check(cudaMemcpy(got.data(), d_out, n * 4, cudaMemcpyDeviceToHost), "download");
          std::string name = std::string(col ? "filter_col" : "filter_row") + (guarded ? "_guard" : "_exact") + "_w" +
                             std::to_string(roi.width) + "_a" + std::to_string(anchor);
          save(name, got);
          record(name, st);
          cudaFree(d_out);
        }
    cudaFree(base);
  }

  // resize: supersampling (shrink) and bicubic (enlarge, mirror addressing on normalized coordinates), on the
  // whole image and on ROIs (offset ROIs sample outside [0, 1] in normalized coordinates; the last one ends at
  // the image's last row and column)
  struct R { const char* name; float f; NppStInterpMode mode; NcvRect32u sroi; };
  for (R r : {R{"resize_super", 0.43f, nppStSupersample, {0, 0, (Ncv32u)W, (Ncv32u)H}},
              R{"resize_bicubic", 1.37f, nppStBicubic, {0, 0, (Ncv32u)W, (Ncv32u)H}},
              R{"resize_bicubic_roi", 1.37f, nppStBicubic, {37, 21, 150, 90}},
              R{"resize_bicubic_roi_shrink", 0.71f, nppStBicubic, {113, 64, 160, 101}},
              R{"resize_bicubic_roi_end", 1.9f, nppStBicubic, {(Ncv32u)W - 61, (Ncv32u)H - 43, 61, 43}},
              R{"resize_super_roi_end", 0.43f, nppStSupersample, {(Ncv32u)W - 128, (Ncv32u)H - 76, 128, 76}},
              R{"resize_super_roi", 0.29f, nppStSupersample, {19, 7, 200, 150}}}) {
    NcvRect32u sroi = r.sroi;
    NcvSize32u dsize = {(Ncv32u)std::floor(sroi.width * r.f), (Ncv32u)std::floor(sroi.height * r.f)};
    const Ncv32u dstep = ((dsize.width * 4 + 255) / 256) * 256;
    Ncv32f* d_out = devAlloc<Ncv32f>(dstep * dsize.height);
    NcvRect32u droi = {0, 0, dsize.width, dsize.height};
    int st = nppiStResize_32f_C1R(d_img, size, step, sroi, d_out, dsize, dstep, droi, r.f, r.f, r.mode);
    check(cudaDeviceSynchronize(), "resize");
    std::vector<Ncv32f> got(dstep / 4 * dsize.height);
    check(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost), "download");
    save(r.name, got);
    record(r.name, st);
    cudaFree(d_out);
  }

  // frame interpolation (bilinear texture fetches with fractional flows)
  {
    std::vector<Ncv32f> f1 = floatImage(W, H, step / 4, 0.9f);
    std::vector<Ncv32f> fu(img.size()), fv(img.size()), bu(img.size()), bv(img.size());
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        size_t i = y * step / 4 + x;
        fu[i] = 1.37f + 0.6f * std::sin(0.05f * y);
        fv[i] = -0.71f + 0.4f * std::cos(0.04f * x);
        bu[i] = -fu[i] + 0.13f;
        bv[i] = -fv[i] - 0.07f;
      }
    Ncv32f *d_f1 = devAlloc<Ncv32f>(img.size() * 4), *d_fu = devAlloc<Ncv32f>(img.size() * 4), *d_fv = devAlloc<Ncv32f>(img.size() * 4),
           *d_bu = devAlloc<Ncv32f>(img.size() * 4), *d_bv = devAlloc<Ncv32f>(img.size() * 4), *d_new = devAlloc<Ncv32f>(img.size() * 4);
    check(cudaMemcpy(d_f1, f1.data(), img.size() * 4, cudaMemcpyHostToDevice), "upload");
    check(cudaMemcpy(d_fu, fu.data(), img.size() * 4, cudaMemcpyHostToDevice), "upload");
    check(cudaMemcpy(d_fv, fv.data(), img.size() * 4, cudaMemcpyHostToDevice), "upload");
    check(cudaMemcpy(d_bu, bu.data(), img.size() * 4, cudaMemcpyHostToDevice), "upload");
    check(cudaMemcpy(d_bv, bv.data(), img.size() * 4, cudaMemcpyHostToDevice), "upload");
    Ncv32u bufSize = 0;
    int st = nppiStGetInterpolationBufferSize(size, step, &bufSize);
    NppStInterpolationState s;
    s.size = size; s.nStep = step; s.pos = 0.37f;
    s.pSrcFrame0 = d_img; s.pSrcFrame1 = d_f1;
    s.pFU = d_fu; s.pFV = d_fv; s.pBU = d_bu; s.pBV = d_bv; s.pNewFrame = d_new;
    std::vector<Ncv32f*> bufs;
    for (int b = 0; b < 6; ++b) { bufs.push_back(devAlloc<Ncv32f>(bufSize)); s.ppBuffers[b] = bufs.back(); }
    if (st == NPPST_SUCCESS) st = nppiStInterpolateFrames(&s);
    check(cudaDeviceSynchronize(), "interpolate");
    std::vector<Ncv32f> got(img.size());
    check(cudaMemcpy(got.data(), d_new, got.size() * 4, cudaMemcpyDeviceToHost), "download");
    save("interpolate", got);
    record("interpolate", st);
    for (auto* b : bufs) cudaFree(b);
    cudaFree(d_f1); cudaFree(d_fu); cudaFree(d_fv); cudaFree(d_bu); cudaFree(d_bv); cudaFree(d_new);
  }
  cudaFree(d_img); cudaFree(d_k);
}


// ---------------------------------------------------------------- Haar cascade detection
// A synthetic cascade (stumps and two-level trees, two- and three-rectangle features; the late stages are
// large enough for the classifier-parallel kernels) run through the multi-scale detection pipeline under
// several configurations, including the code paths that the library selects for older devices.
struct Cascade {
  HaarClassifierCascadeDescriptor desc;
  std::vector<HaarStage64> stages;
  std::vector<HaarClassifierNode128> nodes;
  std::vector<HaarFeature64> features;
};

float u01(std::uint32_t k, std::uint32_t salt) { return (hash32(k * 2654435761U ^ salt) & 0xffffffU) / 16777215.f; }

Cascade makeCascade(std::uint32_t salt) {
  const Ncv32u CW = 24, CH = 24;
  const float areaNorm = (CW - 2 * HAAR_STDDEV_BORDER) * (CH - 2 * HAAR_STDDEV_BORDER);
  const std::vector<Ncv32u> stageTrees = {3, 4, 5, 6, 8, 10, 12, 16, 24, 40, 70, 96};
  Cascade c;
  std::vector<HaarClassifierNode128> roots, children;
  std::uint32_t k = salt * 7919u + 1;
  // one node with a 2- or 3-rectangle feature; leaves (when leaf) +/- values
  auto makeNode = [&](bool leftLeaf, bool rightLeaf, float leftVal, Ncv32u rightOffsetOrNone, float rightVal) {
    ++k;
    Ncv32u fx = (Ncv32u)(u01(k, 1) * 14), fy = (Ncv32u)(u01(k, 2) * 14);
    Ncv32u fw = 4 + 2 * (Ncv32u)(u01(k, 3) * 4), fh = 4 + (Ncv32u)(u01(k, 4) * 6);
    fw = std::min(fw, CW - fx); fh = std::min(fh, CH - fy);
    const bool three = u01(k, 5) < 0.35f;
    const Ncv32u off = (Ncv32u)c.features.size();
    HaarFeature64 r;
    if (!three) {  // left half minus right half
      Ncv32u hw = std::max<Ncv32u>(fw / 2, 1);
      r.setRect(fx, fy, hw, fh, CW, CH); r.setWeight(1.0f); c.features.push_back(r);
      r.setRect(fx + hw, fy, hw, fh, CW, CH); r.setWeight(-1.0f); c.features.push_back(r);
    } else {  // whole minus 3x middle third (vertical)
      Ncv32u th = std::max<Ncv32u>(fh / 3, 1);
      r.setRect(fx, fy, fw, 3 * th, CW, CH); r.setWeight(-1.0f); c.features.push_back(r);
      r.setRect(fx, fy + th, fw, th, CW, CH); r.setWeight(3.0f); c.features.push_back(r);
    }
    const float rectArea = (float)fw * fh;
    HaarFeatureDescriptor32 fd;
    fd.create(false, leftLeaf, rightLeaf, (Ncv32u)c.features.size() - off, off);
    HaarClassifierNode128 n;
    n.setFeatureDesc(fd);
    n.setThreshold((u01(k, 6) - 0.5f) * 0.6f * rectArea / areaNorm);
    HaarClassifierNodeDescriptor32 l, rr;
    l.create(leftVal);
    if (rightLeaf) rr.create(rightVal); else rr.create(rightOffsetOrNone);
    n.setLeftNodeDesc(l);
    n.setRightNodeDesc(rr);
    return n;
  };
  Ncv32u numStages = (Ncv32u)stageTrees.size(), totalRoots = 0;
  for (Ncv32u t : stageTrees) totalRoots += t;
  for (Ncv32u s = 0; s < numStages; ++s) {
    HaarStage64 st;
    st.setStartClassifierRootNodeOffset((Ncv32u)roots.size());
    st.setNumClassifierRootNodes(stageTrees[s]);
    st.setStageThreshold(-0.05f * std::sqrt((float)stageTrees[s]));
    c.stages.push_back(st);
    for (Ncv32u t = 0; t < stageTrees[s]; ++t) {
      const float a = 0.6f + 0.8f * u01(k, 7), b = -(0.6f + 0.8f * u01(k, 8));
      if ((t + s) % 5 == 3) {  // a two-level tree: the right branch is another node
        Ncv32u childIndex = totalRoots + (Ncv32u)children.size();
        roots.push_back(makeNode(true, false, a, childIndex, 0.f));
        children.push_back(makeNode(true, true, 0.5f * a, 0, b));
      } else {
        roots.push_back(makeNode(true, true, a, 0, b));
      }
    }
  }
  c.nodes = roots;
  c.nodes.insert(c.nodes.end(), children.begin(), children.end());
  c.desc.NumStages = numStages;
  c.desc.NumClassifierRootNodes = totalRoots;
  c.desc.NumClassifierTotalNodes = (Ncv32u)c.nodes.size();
  c.desc.NumFeatures = (Ncv32u)c.features.size();
  c.desc.ClassifierSize = {CW, CH};
  c.desc.bNeedsTiltedII = false;
  c.desc.bHasStumpsOnly = children.empty();
  return c;
}

std::vector<Ncv8u> haarImage(int W, int H, std::uint32_t salt) {
  std::vector<Ncv8u> img((size_t)W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      float v = 120.f + 55.f * std::sin(0.13f * x + 0.7f * salt) * std::cos(0.09f * y - 0.3f * salt) +
                45.f * ((hash32(y * 7919 + x + salt * 104729u) & 255) / 255.f - 0.5f);
      // a few bright and dark blocks of different sizes
      for (int b = 0; b < 6; ++b) {
        int bx = (int)(hash32(salt * 31 + b) % (W - 40)), by = (int)(hash32(salt * 37 + b) % (H - 40));
        int bs = 12 + (int)(hash32(salt * 41 + b) % 28);
        if (x >= bx && x < bx + bs && y >= by && y < by + bs) v += (b & 1) ? 60.f : -60.f;
      }
      img[(size_t)y * W + x] = (Ncv8u)std::min(255.f, std::max(0.f, v));
    }
  return img;
}

void testHaar(cudaDeviceProp& prop) {
  NCVMemNativeAllocator gpuAlloc(NCVMemoryTypeDevice, static_cast<Ncv32u>(prop.textureAlignment));
  NCVMemNativeAllocator cpuAlloc(NCVMemoryTypeHostPinned, static_cast<Ncv32u>(prop.textureAlignment));
  for (int ci = 0; ci < 2; ++ci) {
    Cascade c = makeCascade(11 + ci);
    NCVVectorAlloc<HaarStage64> h_stages(cpuAlloc, (Ncv32u)c.stages.size());
    NCVVectorAlloc<HaarStage64> d_stages(gpuAlloc, (Ncv32u)c.stages.size());
    NCVVectorAlloc<HaarClassifierNode128> d_nodes(gpuAlloc, (Ncv32u)c.nodes.size());
    NCVVectorAlloc<HaarFeature64> d_features(gpuAlloc, (Ncv32u)c.features.size());
    std::copy(c.stages.begin(), c.stages.end(), h_stages.ptr());
    check(cudaMemcpy(d_stages.ptr(), c.stages.data(), c.stages.size() * sizeof(HaarStage64), cudaMemcpyHostToDevice), "upload");
    check(cudaMemcpy(d_nodes.ptr(), c.nodes.data(), c.nodes.size() * sizeof(HaarClassifierNode128), cudaMemcpyHostToDevice), "upload");
    check(cudaMemcpy(d_features.ptr(), c.features.data(), c.features.size() * sizeof(HaarFeature64), cudaMemcpyHostToDevice), "upload");
    const int W = ci ? 352 : 320, H = ci ? 288 : 240;
    std::vector<Ncv8u> img = haarImage(W, H, 3 + ci);
    struct Cfg { const char* name; int major, minor; Ncv32u pixelStep, flags, minNeighbors; float scaleStep; };
    for (Cfg cfg : {Cfg{"real", 0, 0, 1, NCVPipeObjDet_Default, 0, 1.2f}, Cfg{"step2", 0, 0, 2, NCVPipeObjDet_Default, 0, 1.2f},
                    Cfg{"fair", 0, 0, 1, NCVPipeObjDet_UseFairImageScaling, 0, 1.25f},
                    Cfg{"sm13", 1, 3, 1, NCVPipeObjDet_Default, 0, 1.2f}, Cfg{"sm10", 1, 0, 1, NCVPipeObjDet_Default, 0, 1.2f},
                    Cfg{"sm10_group", 1, 0, 1, NCVPipeObjDet_Default, 2, 1.15f},
                    Cfg{"sm10_largest", 1, 0, 1, NCVPipeObjDet_FindLargestObject, 1, 1.2f}}) {
      cudaDeviceProp p = prop;
      if (cfg.major) { p.major = cfg.major; p.minor = cfg.minor; }
      NCVMatrixAlloc<Ncv8u> d_img(gpuAlloc, (Ncv32u)W, (Ncv32u)H);
      check(cudaMemcpy2D(d_img.ptr(), d_img.pitch(), img.data(), W, W, H, cudaMemcpyHostToDevice), "upload");
      const Ncv32u maxRects = 1u << 19;  // above the number of candidate windows over all scales: no truncation
      NCVVectorAlloc<NcvRect32u> d_rects(gpuAlloc, maxRects);
      Ncv32u num = 0;
      int st = ncvDetectObjectsMultiScale_device(d_img, NcvSize32u{(Ncv32u)W, (Ncv32u)H}, d_rects, num, c.desc, h_stages, d_stages,
                                                 d_nodes, d_features, NcvSize32u{24, 24}, cfg.minNeighbors, cfg.scaleStep,
                                                 cfg.pixelStep, cfg.flags, gpuAlloc, cpuAlloc, p, 0);
      check(cudaDeviceSynchronize(), "haar");
      std::vector<NcvRect32u> rects(std::min(num, maxRects));
      if (!rects.empty())
        check(cudaMemcpy(rects.data(), d_rects.ptr(), rects.size() * sizeof(NcvRect32u), cudaMemcpyDeviceToHost), "download");
      // detections are compacted with atomics on newer devices: compare them as a set
      std::vector<std::uint32_t> out;
      for (auto& r : rects) out.insert(out.end(), {r.x, r.y, r.width, r.height});
      std::vector<std::array<std::uint32_t, 4>> q;
      for (size_t i = 0; i + 3 < out.size(); i += 4) q.push_back({out[i], out[i + 1], out[i + 2], out[i + 3]});
      std::sort(q.begin(), q.end());
      out.clear();
      out.push_back(num);
      for (auto& a : q) out.insert(out.end(), a.begin(), a.end());
      std::string name = std::string("haar") + std::to_string(ci) + "_" + cfg.name;
      save(name, out);
      record(name, st, num);
    }
  }
}

}  // namespace

int main(int argc, char** argv) try {
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--out" && i + 1 < argc) g_out = argv[++i];
    else if (std::string(argv[i]) == "--seed" && i + 1 < argc) g_seed = (std::uint32_t)std::stoul(argv[++i]);
  if (g_out.empty()) throw std::runtime_error("usage: ncv_check --out DIR");
  int dev = 0;
  check(cudaGetDevice(&dev), "device");
  cudaDeviceProp prop;
  check(cudaGetDeviceProperties(&prop, dev), "props");
  testIntegral(prop, 640, 480, 0, "aligned");
  if (g_seed == 0) testIntegral(prop, 637, 471, 1, "unaligned");
  else testIntegral(prop, 600 + (int)(g_seed % 41), 440 + (int)(g_seed / 41 % 37), 1 + (int)(g_seed / 1517 % 7), "unaligned");
  testDecimateTransposeCompact(prop);
  testFloat(prop);
  testHaar(prop);
  std::ofstream(g_out + "/result.json") << "{" << g_json.str() << "}\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "ncv_check: " << e.what() << std::endl;
  return 1;
}

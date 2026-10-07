// people_eval: headless body-part segmentation with pcl::gpu::people on organized RGB-D frames.
//
//   people_eval --pcd FRAME.pcd --trees DIR --out DIR [--variants all|V1,V2,...] [--repeat N] [--threads K]
//
// Variants: base, mirror, farther, noise, holes, and jitter-<seed> (seeded noise, hole and shift).
//
// For every deterministic variant of the input frame writes DIR/<variant>.labels
// (640x480 uint8 body-part labels) and a line in DIR/result.json with label
// histograms, the detector return code and timing.
#include <pcl/gpu/people/people_detector.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <thread>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cuda_runtime.h>

namespace {

using Cloud = pcl::PointCloud<pcl::PointXYZRGBA>;

std::uint32_t hash32(std::uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');)
    if (!item.empty()) out.push_back(item);
  return out;
}

Cloud::Ptr variant(const Cloud& base, const std::string& v) {
  Cloud::Ptr c(new Cloud(base));
  const int W = (int)base.width, H = (int)base.height;
  if (v == "base") return c;
  if (v == "mirror") {
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        auto p = base.at(W - 1 - x, y);
        p.x = -p.x;
        c->at(x, y) = p;
      }
  } else if (v == "farther") {
    for (auto& p : c->points) if (std::isfinite(p.z)) { p.x *= 1.25f; p.y *= 1.25f; p.z *= 1.25f; p.z += 0.0f; }
  } else if (v == "noise") {
    for (size_t i = 0; i < c->points.size(); ++i) {
      auto& p = c->points[i];
      if (!std::isfinite(p.z)) continue;
      float u = (hash32((std::uint32_t)i * 2654435761U) + 0.5f) / 4294967296.f - 0.5f;
      float s = 1.f + 0.012f * u;
      p.x *= s; p.y *= s; p.z *= s;
    }
  } else if (v == "holes") {
    for (int y = 120; y < 300; ++y)
      for (int x = 250; x < 330; ++x) {
        auto& p = c->at(x, y);
        p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
      }
  } else if (v.rfind("jitter-", 0) == 0) {
    // jitter-<seed>: seeded per-point scale noise, a seeded hole and a one-pixel shift
    const std::uint32_t seed = (std::uint32_t)std::stoul(v.substr(7));
    for (size_t i = 0; i < c->points.size(); ++i) {
      auto& p = c->points[i];
      if (!std::isfinite(p.z)) continue;
      float u = (hash32((std::uint32_t)i * 2654435761U ^ seed * 0x9e3779b1U) + 0.5f) / 4294967296.f - 0.5f;
      float s = 1.f + 0.02f * u;
      p.x *= s; p.y *= s; p.z *= s;
    }
    const int hx = 40 + (int)(hash32(seed) % 480), hy = 40 + (int)(hash32(seed + 1) % 320);
    for (int y = hy; y < hy + 60; ++y)
      for (int x = hx; x < hx + 70; ++x) {
        auto& p = c->at(x, y);
        p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
      }
    if (seed & 1) {
      Cloud tmp(*c);
      for (int y = 0; y < H; ++y)
        for (int x = 1; x < W; ++x) c->at(x, y) = tmp.at(x - 1, y);
    }
  } else {
    throw std::runtime_error("unknown variant " + v);
  }
  return c;
}

}  // namespace

// Runs every variant of the frame through one detector; writes OUT/<variant>.labels and OUT/result.json.
int run_detector(const Cloud& base, const std::string& trees, const std::string& out, const std::vector<std::string>& vars, int repeat) {
  std::vector<std::string> names = {trees + "/tree_20_1.txt", trees + "/tree_20_2.txt", trees + "/tree_20_3.txt"};
  pcl::gpu::people::RDFBodyPartsDetector::Ptr rdf(new pcl::gpu::people::RDFBodyPartsDetector(names));
  pcl::gpu::people::PeopleDetector det;
  det.rdf_detector_ = rdf;
  std::ofstream js(out + "/result.json");
  js << "{";
  for (size_t vi = 0; vi < vars.size(); ++vi) {
    Cloud::Ptr c = variant(base, vars[vi]);
    int ret = -1;
    std::vector<double> ms;
    for (int r = 0; r < repeat; ++r) {
      cudaDeviceSynchronize();
      auto t0 = std::chrono::steady_clock::now();
      ret = det.process(c);
      cudaDeviceSynchronize();
      ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    std::vector<unsigned char> labels;
    int cols;
    rdf->getLabels().download(labels, cols);
    std::ofstream lf(out + "/" + vars[vi] + ".labels", std::ios::binary);
    lf.write(reinterpret_cast<const char*>(labels.data()), labels.size());
    std::map<int, long> hist;
    for (auto l : labels) hist[l]++;
    js << (vi ? ", " : "") << "\"" << vars[vi] << "\": {\"return\": " << ret << ", \"ms_median\": " << ms[ms.size() / 2]
       << ", \"histogram\": {";
    bool first = true;
    for (auto& kv : hist) {
      js << (first ? "" : ", ") << "\"" << kv.first << "\": " << kv.second;
      first = false;
    }
    js << "}}";
  }
  js << "}\n";
  return 0;
}

int main(int argc, char** argv) try {
  std::string pcd, trees, out = ".", which = "all";
  int repeat = 1, threads = 0;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
    std::string v = argv[++i];
    if (k == "--pcd") pcd = v;
    else if (k == "--trees") trees = v;
    else if (k == "--out") out = v;
    else if (k == "--variants") which = v;
    else if (k == "--repeat") repeat = std::stoi(v);
    else if (k == "--threads") threads = std::stoi(v);
    else throw std::runtime_error("unknown argument " + k);
  }
  Cloud base;
  if (pcl::io::loadPCDFile(pcd, base) != 0 || base.width != 640 || base.height != 480)
    throw std::runtime_error("cannot read an organized 640x480 cloud from " + pcd);
  std::vector<std::string> vars = which == "all" ? std::vector<std::string>{"base", "mirror", "farther", "noise", "holes"}
                                                 : split(which);
  if (threads <= 0) return run_detector(base, trees, out, vars, repeat);
  // --threads K: K independent detectors at the same time, each on the variants in a
  // different order; thread t writes to OUT/thread<t>/
  std::vector<std::thread> th;
  std::vector<int> rc(threads, 1);
  for (int t = 0; t < threads; ++t) {
    std::vector<std::string> mine = vars;
    std::rotate(mine.begin(), mine.begin() + (t % mine.size()), mine.end());
    std::string dir = out + "/thread" + std::to_string(t);
    std::filesystem::create_directories(dir);
    th.emplace_back([&, t, mine, dir] {
      try { rc[t] = run_detector(base, trees, dir, mine, repeat); }
      catch (const std::exception& e) { std::cerr << "thread " << t << ": " << e.what() << std::endl; rc[t] = 1; }
    });
  }
  for (auto& x : th) x.join();
  for (int r : rc)
    if (r) return 1;
  return 0;
} catch (const std::exception& e) {
  std::cerr << "people_eval: " << e.what() << std::endl;
  return 1;
}

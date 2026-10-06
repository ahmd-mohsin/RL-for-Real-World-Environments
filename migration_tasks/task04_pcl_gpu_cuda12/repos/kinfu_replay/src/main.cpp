// kinfu_replay: headless KinectFusion runs on synthetic depth sequences.
//
//   kinfu_replay --mode kinfu|kinfuls --frames N --out DIR [--noise SEED] [--icp-trunc M] [--skip-mesh]
//
// Writes DIR/result.json (poses, ground truth, errors, TSDF statistics, mesh
// quality, timing) and DIR/tsdf_slices.bin (three axis-aligned TSDF slices).
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
#include <pcl/gpu/kinfu/raycaster.h>
#include <pcl/gpu/kinfu_large_scale/kinfu.h>
#include <pcl/gpu/kinfu_large_scale/marching_cubes.h>
#include <pcl/gpu/kinfu_large_scale/raycaster.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "scene.h"

using namespace replay;

namespace {

struct Args {
  std::string mode = "kinfu", out = ".";
  int frames = 120, noise = -1;
  float icp_trunc = 0.f;
  bool skip_mesh = false;
  std::string traj, scene;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--mode") a.mode = val();
    else if (k == "--frames") a.frames = std::stoi(val());
    else if (k == "--out") a.out = val();
    else if (k == "--noise") a.noise = std::stoi(val());
    else if (k == "--icp-trunc") a.icp_trunc = std::stof(val());
    else if (k == "--skip-mesh") a.skip_mesh = true;
    else if (k == "--traj") a.traj = val();
    else if (k == "--scene") a.scene = val();
    else throw std::runtime_error("unknown argument " + k);
  }
  return a;
}

std::string mat(const Eigen::Affine3f& p) {
  std::ostringstream o;
  o.precision(9);
  o << "[";
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) o << (r || c ? "," : "") << p.matrix()(r, c);
  o << "]";
  return o.str();
}

float rotErrDeg(const Eigen::Matrix3f& a, const Eigen::Matrix3f& b) {
  Eigen::AngleAxisf aa(a.transpose() * b);
  return std::abs(aa.angle()) * 57.29578f;
}

// TSDF voxels are packed short2 (tsdf * 32767, weight) in an int
struct TsdfStats {
  long long observed = 0, near_surface = 0;
  double weight_sum = 0, tsdf_abs_sum = 0;
};

TsdfStats tsdfStats(const std::vector<int>& vox, std::vector<float>& slices, int res) {
  TsdfStats s;
  slices.assign(3 * (size_t)res * res, 0.f);
  for (size_t i = 0; i < vox.size(); ++i) {
    short t = (short)(vox[i] & 0xffff);
    short w = (short)((unsigned)vox[i] >> 16);
    if (w == 0) continue;
    s.observed++;
    s.weight_sum += w;
    float tf = t / 32767.f;
    s.tsdf_abs_sum += std::abs(tf);
    if (std::abs(tf) < 0.2f) s.near_surface++;
  }
  // slices through the volume centre: x = res/2 (y,z), y = res/2 (x,z), z = res/2 (x,y); layout x fastest, then y, then z
  auto at = [&](int x, int y, int z) {
    int v = vox[((size_t)z * res + y) * res + x];
    short w = (short)((unsigned)v >> 16);
    return w ? (short)(v & 0xffff) / 32767.f : 2.f;
  };
  int h = res / 2;
  for (int a = 0; a < res; ++a)
    for (int b = 0; b < res; ++b) {
      slices[(size_t)a * res + b] = at(h, a, b);
      slices[(size_t)res * res + (size_t)a * res + b] = at(a, h, b);
      slices[2 * (size_t)res * res + (size_t)a * res + b] = at(a, b, h);
    }
  return s;
}

template <class Point>
void meshQuality(const std::vector<Point>& tri, const Scene& scene, const Eigen::Affine3f& world_from_volume, std::ostream& o) {
  double area = 0, err_sum = 0, err_max = 0;
  Eigen::Vector3f lo(1e9f, 1e9f, 1e9f), hi(-1e9f, -1e9f, -1e9f);
  for (size_t k = 0; k + 2 < tri.size(); k += 3) {
    Eigen::Vector3f a(tri[k].x, tri[k].y, tri[k].z), b(tri[k + 1].x, tri[k + 1].y, tri[k + 1].z), c(tri[k + 2].x, tri[k + 2].y, tri[k + 2].z);
    area += 0.5 * ((b - a).cross(c - a)).norm();
    for (auto* p : {&a, &b, &c}) {
      Eigen::Vector3f w = world_from_volume * (*p);
      double e = std::abs(scene.sdf(w));
      err_sum += e;
      err_max = std::max(err_max, e);
      lo = lo.cwiseMin(w);
      hi = hi.cwiseMax(w);
    }
  }
  size_t nv = tri.size();
  o << "\"mesh\": {\"triangles\": " << nv / 3 << ", \"area\": " << area << ", \"surface_err_mean\": " << (nv ? err_sum / nv : 0)
    << ", \"surface_err_max\": " << err_max << ", \"bbox\": [" << lo.x() << "," << lo.y() << "," << lo.z() << "," << hi.x() << ","
    << hi.y() << "," << hi.z() << "]}";
}

}  // namespace

int main(int argc, char** argv) try {
  Args a = parse(argc, argv);
  Intrinsics in;
  const bool ls = a.mode == "kinfuls";
  if (!ls && a.mode != "kinfu") throw std::runtime_error("unknown mode " + a.mode);
  Scene scene(!a.scene.empty() ? a.scene : (ls ? "corridor" : "room"));
  const std::string traj = !a.traj.empty() ? a.traj : (ls ? "walk" : "orbit");
  const Eigen::Vector3f vsize(3.f, 3.f, 3.f);

  std::vector<Eigen::Affine3f> gt, est;
  std::vector<double> frame_ms;
  std::vector<std::uint16_t> depth;
  pcl::gpu::DeviceArray2D<unsigned short> d_depth;
  std::ostringstream extra;
  std::vector<float> slices;
  TsdfStats ts;
  int res = 512;
  double raycast_err = -1, raycast_valid = 0;
  Eigen::Affine3f world_from_volume = Eigen::Affine3f::Identity();

  auto run_frames = [&](auto&& kf, auto&& pose_of) {
    for (int k = 0; k < a.frames; ++k) {
      Eigen::Affine3f p = trajectoryPose(traj, k, a.frames);
      gt.push_back(p);
      renderDepth(scene, in, p, a.noise < 0 ? -1 : a.noise * 100003 + k, depth);
      d_depth.upload(depth.data(), in.cols * sizeof(std::uint16_t), in.rows, in.cols);
      cudaDeviceSynchronize();
      auto t0 = std::chrono::steady_clock::now();
      kf(d_depth);
      cudaError_t err = cudaDeviceSynchronize();
      if (std::getenv("REPLAY_DEBUG")) {
        size_t nz = 0;
        for (auto d : depth) nz += d != 0;
        std::cerr << "frame " << k << " valid_depth " << nz << " cuda " << cudaGetErrorString(err) << " last "
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
      }
      frame_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      est.push_back(pose_of(k));
    }
  };

  if (!ls) {
    pcl::gpu::KinfuTracker kf(in.rows, in.cols);
    kf.setDepthIntrinsics(in.fx, in.fy, in.cx, in.cy);
    kf.volume().setSize(vsize);
    kf.setInitalCameraPose(trajectoryPose(traj, 0, a.frames));
    kf.setDepthTruncationForICP(a.icp_trunc);
    kf.setCameraMovementThreshold(0.f);
    run_frames([&](auto& d) { return kf(d); }, [&](int) { return kf.getCameraPose(); });
    pcl::gpu::DeviceArray2D<int> vol = kf.volume().data();
    std::vector<int> host;
    int cols;
    vol.download(host, cols);
    res = kf.volume().getResolution().x();
    ts = tsdfStats(host, slices, res);
    if (!a.skip_mesh) {
      pcl::gpu::MarchingCubes mc;
      pcl::gpu::DeviceArray<pcl::PointXYZ> buf;
      pcl::gpu::DeviceArray<pcl::PointXYZ> tris = mc.run(kf.volume(), buf);
      std::vector<pcl::PointXYZ> h;
      tris.download(h);
      meshQuality(h, scene, world_from_volume, extra);
      extra << ", ";
    }
    // ray-cast the model at the last pose and compare with the rendered ground truth
    pcl::gpu::RayCaster rc(in.rows, in.cols, in.fx, in.fy, in.cx, in.cy);
    rc.run(kf.volume(), kf.getCameraPose());
    pcl::gpu::DeviceArray2D<unsigned short> d_rc;
    rc.generateDepthImage(d_rc);
    std::vector<unsigned short> h_rc;
    int c2;
    d_rc.download(h_rc, c2);
    renderDepth(scene, in, gt.back(), -1, depth);
    double s = 0; long long n = 0;
    for (size_t i = 0; i < depth.size(); ++i)
      if (depth[i] && h_rc[i]) { s += std::abs((double)depth[i] - h_rc[i]); ++n; }
    raycast_err = n ? s / n : -1;
    raycast_valid = (double)n / depth.size();
  } else {
    pcl::gpu::kinfuLS::KinfuTracker kf(vsize, 0.8f, in.rows, in.cols);
    kf.setDepthIntrinsics(in.fx, in.fy, in.cx, in.cy);
    kf.setInitialCameraPose(trajectoryPose(traj, 0, a.frames));
    kf.setDepthTruncationForICP(a.icp_trunc);
    kf.setCameraMovementThreshold(0.f);
    run_frames([&](auto& d) { return kf(d); }, [&](int) { return kf.getCameraPose(); });
    // the cyclical buffer's origin gives the volume's current world placement
    pcl::gpu::kinfuLS::tsdf_buffer* buf = kf.getCyclicalBufferStructure();
    world_from_volume.translation() = Eigen::Vector3f(buf->origin_metric.x, buf->origin_metric.y, buf->origin_metric.z);
    extra << "\"shifts_origin\": [" << buf->origin_GRID_global.x << "," << buf->origin_GRID_global.y << "," << buf->origin_GRID_global.z
          << "], ";
    pcl::gpu::DeviceArray2D<int> vol = kf.volume().data();
    std::vector<int> host;
    int cols;
    vol.download(host, cols);
    res = kf.volume().getResolution().x();
    ts = tsdfStats(host, slices, res);
    if (!a.skip_mesh) {
      pcl::gpu::kinfuLS::MarchingCubes mc;
      pcl::gpu::DeviceArray<pcl::PointXYZ> mbuf;
      pcl::gpu::DeviceArray<pcl::PointXYZ> tris = mc.run(kf.volume(), mbuf);
      std::vector<pcl::PointXYZ> h;
      tris.download(h);
      // the tracker's world frame is its own (default initial pose); map it onto the scene with frame 0
      Eigen::Affine3f scene_from_tracker = gt[0] * est[0].inverse();
      meshQuality(h, scene, scene_from_tracker * world_from_volume, extra);
      extra << ", ";
    }
  }

  // errors of the motion relative to frame 0 (the trackers may place their own world origin)
  double te_sum = 0, te_max = 0, re_max = 0;
  const Eigen::Affine3f g0i = gt[0].inverse(), e0i = est[0].inverse();
  for (size_t k = 0; k < gt.size(); ++k) {
    Eigen::Affine3f g = g0i * gt[k], e = e0i * est[k];
    double te = (g.translation() - e.translation()).norm();
    te_sum += te;
    te_max = std::max(te_max, te);
    re_max = std::max(re_max, (double)rotErrDeg(g.linear(), e.linear()));
  }
  std::vector<double> sorted = frame_ms;
  std::sort(sorted.begin(), sorted.end());

  std::ofstream f(a.out + "/result.json");
  f.precision(10);
  f << "{\"mode\": \"" << a.mode << "\", \"frames\": " << a.frames << ", \"noise\": " << a.noise << ", ";
  f << "\"trans_err_mean\": " << te_sum / gt.size() << ", \"trans_err_max\": " << te_max << ", \"rot_err_max_deg\": " << re_max << ", ";
  f << "\"tsdf\": {\"resolution\": " << res << ", \"observed\": " << ts.observed << ", \"near_surface\": " << ts.near_surface
    << ", \"weight_sum\": " << ts.weight_sum << ", \"tsdf_abs_mean\": " << (ts.observed ? ts.tsdf_abs_sum / ts.observed : 0) << "}, ";
  f << extra.str();
  f << "\"raycast_depth_err_mm\": " << raycast_err << ", \"raycast_valid_frac\": " << raycast_valid << ", ";
  f << "\"frame_ms_median\": " << sorted[sorted.size() / 2] << ", \"frame_ms_total\": " << std::accumulate(frame_ms.begin(), frame_ms.end(), 0.0) << ", ";
  f << "\"poses\": [";
  for (size_t k = 0; k < est.size(); ++k) f << (k ? "," : "") << mat(est[k]);
  f << "], \"gt_poses\": [";
  for (size_t k = 0; k < gt.size(); ++k) f << (k ? "," : "") << mat(gt[k]);
  f << "]}\n";
  std::ofstream sl(a.out + "/tsdf_slices.bin", std::ios::binary);
  sl.write(reinterpret_cast<const char*>(&res), 4);
  sl.write(reinterpret_cast<const char*>(slices.data()), slices.size() * sizeof(float));
  std::cout << "ok " << a.mode << " frames " << a.frames << " trans_err_mean " << te_sum / gt.size() << " median_ms "
            << sorted[sorted.size() / 2] << std::endl;
  return 0;
} catch (const std::exception& e) {
  std::cerr << "kinfu_replay: " << e.what() << std::endl;
  return 1;
}

// kinfu_replay: headless KinectFusion runs on synthetic depth sequences.
//
//   kinfu_replay --mode kinfu|kinfuls --frames N --out DIR [--noise SEED] [--icp-trunc M] [--skip-mesh]
//                [--traj orbit|walk|static] [--scene room|corridor] [--stop K] [--shift M]
//                [--extract-every K] [--color] [--api]
//                [--concurrent "mode:traj:noise,..."]
//
// Writes DIR/result.json (poses, ground truth, errors, TSDF statistics, mesh
// quality, timing) and DIR/tsdf_slices.bin (three axis-aligned TSDF slices).
// --color feeds RGB frames (procedural scene colours) to the trackers' colour volumes;
// --api also exercises normals, colours, rendered views, ray-cast maps, the last-frame
// buffers and the host-side extraction, writing DIR/view.bin and DIR/scene_view.bin.
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/kinfu/color_volume.h>
#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
#include <pcl/gpu/kinfu/raycaster.h>
#include <pcl/gpu/kinfu_large_scale/color_volume.h>
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
#include <filesystem>
#include <thread>

#include "scene.h"

using namespace replay;

namespace {

struct Args {
  std::string mode = "kinfu", out = ".";
  int frames = 120, noise = -1;
  int stop = 0;           // > 0: run only the first `stop` frames of the trajectory
  int extract_every = 0;  // > 0: extract a mesh (and a cloud) every K frames and record checksums
  float icp_trunc = 0.f, shift = 0.8f;  // shift: kinfuLS volume shifting distance (m)
  bool skip_mesh = false, color = false, api = false;
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
    else if (k == "--stop") a.stop = std::stoi(val());
    else if (k == "--shift") a.shift = std::stof(val());
    else if (k == "--extract-every") a.extract_every = std::stoi(val());
    else if (k == "--out") a.out = val();
    else if (k == "--noise") a.noise = std::stoi(val());
    else if (k == "--icp-trunc") a.icp_trunc = std::stof(val());
    else if (k == "--skip-mesh") a.skip_mesh = true;
    else if (k == "--color") a.color = true;
    else if (k == "--api") a.api = true;
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

// exact, order-independent checksum of a point set (sum of the coordinates' bit patterns)
template <class Point>
std::uint64_t pointChecksum(const std::vector<Point>& pts) {
  std::uint64_t s = 0;
  for (const auto& p : pts) {
    std::uint32_t b[3];
    std::memcpy(&b[0], &p.x, 4);
    std::memcpy(&b[1], &p.y, 4);
    std::memcpy(&b[2], &p.z, 4);
    s += b[0] + 3ull * b[1] + 7ull * b[2];
  }
  return s;
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


// --api statistics -----------------------------------------------------------------------------

std::string num(double v) {
  std::ostringstream o;
  o.precision(9);
  o << (std::isfinite(v) ? v : -1.0);
  return o.str();
}

bool finite3(float x, float y, float z) { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }

// oriented points against the scene: count, mean |sdf| and mean normal angle (degrees, sign-free)
struct Oriented {
  long long valid = 0;
  double sdf = 0, angle = 0, absn[3] = {0, 0, 0};
  void add(const Scene& scene, const Eigen::Affine3f& T, float px, float py, float pz, float nx, float ny, float nz, bool gt) {
    if (!finite3(px, py, pz) || !finite3(nx, ny, nz)) return;
    Eigen::Vector3f n(nx, ny, nz);
    if (n.norm() < 0.5f) return;
    ++valid;
    absn[0] += std::abs(nx); absn[1] += std::abs(ny); absn[2] += std::abs(nz);
    if (!gt) return;
    Eigen::Vector3f p = T * Eigen::Vector3f(px, py, pz);
    sdf += std::abs(scene.sdf(p));
    float c = std::abs((T.linear() * n.normalized()).dot(sceneNormal(scene, p)));
    angle += std::acos(std::min(1.f, c)) * 57.29578;
  }
  std::string json(bool gt) const {
    double k = valid ? 1.0 / valid : 0;
    std::ostringstream o;
    o << "{\"valid\": " << valid << ", \"abs_normal_mean\": [" << num(absn[0] * k) << "," << num(absn[1] * k) << "," << num(absn[2] * k) << "]";
    if (gt) o << ", \"sdf_mean\": " << num(sdf * k) << ", \"angle_deg_mean\": " << num(angle * k);
    o << "}";
    return o.str();
  }
};

template <class Map>
std::string mapStats(const Scene& scene, const Eigen::Affine3f& T, const Map& vmap, const Map& nmap, bool gt) {
  std::vector<float> v, n;
  int cv, cn;
  vmap.download(v, cv);
  nmap.download(n, cn);
  const size_t H = v.size() / cv / 3;
  Oriented s;
  for (size_t y = 0; y < H; ++y)
    for (int x = 0; x < cv; ++x)
      s.add(scene, T, v[y * cv + x], v[(y + H) * cv + x], v[(y + 2 * H) * cv + x], n[y * cn + x], n[(y + H) * cn + x],
            n[(y + 2 * H) * cn + x], gt);
  return s.json(gt);
}

template <class Pixel>
std::string saveView(const std::string& path, const pcl::gpu::DeviceArray2D<Pixel>& view) {
  std::vector<Pixel> h;
  int cols;
  view.download(h, cols);
  const int rows = cols ? (int)(h.size() / cols) : 0;
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(&rows), 4);
  f.write(reinterpret_cast<const char*>(&cols), 4);
  double sum[3] = {0, 0, 0};
  for (const auto& p : h) {
    const unsigned char c[3] = {p.r, p.g, p.b};
    f.write(reinterpret_cast<const char*>(c), 3);
    sum[0] += c[0]; sum[1] += c[1]; sum[2] += c[2];
  }
  const double k = h.empty() ? 0 : 1.0 / h.size();
  return "{\"rows\": " + std::to_string(rows) + ", \"cols\": " + std::to_string(cols) + ", \"mean\": [" + num(sum[0] * k) + "," +
         num(sum[1] * k) + "," + num(sum[2] * k) + "]}";
}

std::string colorStats(const Eigen::Affine3f& T, const std::vector<pcl::PointXYZ>& pts, const std::vector<pcl::RGB>& cols, bool gt) {
  double err = 0, mean[3] = {0, 0, 0};
  const size_t n = std::min(pts.size(), cols.size());
  for (size_t i = 0; i < n; ++i) {
    mean[0] += cols[i].r; mean[1] += cols[i].g; mean[2] += cols[i].b;
    if (gt) {
      std::uint8_t ref[3];
      sceneColor(T * Eigen::Vector3f(pts[i].x, pts[i].y, pts[i].z), ref);
      err += (std::abs((int)cols[i].r - ref[0]) + std::abs((int)cols[i].g - ref[1]) + std::abs((int)cols[i].b - ref[2])) / 3.0;
    }
  }
  const double k = n ? 1.0 / n : 0;
  std::string s = "{\"points\": " + std::to_string(n) + ", \"mean\": [" + num(mean[0] * k) + "," + num(mean[1] * k) + "," + num(mean[2] * k) + "]";
  if (gt) s += ", \"abs_err_mean\": " + num(err * k);
  return s + "}";
}

template <class Tracker>
std::string lastFrameStats(const Scene& scene, const Eigen::Affine3f& T, const Tracker& kf) {
  pcl::gpu::DeviceArray2D<pcl::PointXYZ> lc;
  pcl::gpu::DeviceArray2D<pcl::Normal> ln;
  kf.getLastFrameCloud(lc);
  kf.getLastFrameNormals(ln);
  std::vector<pcl::PointXYZ> hc;
  std::vector<pcl::Normal> hn;
  int c1, c2;
  lc.download(hc, c1);
  ln.download(hn, c2);
  Oriented s;
  for (size_t i = 0; i < std::min(hc.size(), hn.size()); ++i)
    s.add(scene, T, hc[i].x, hc[i].y, hc[i].z, hn[i].normal_x, hn[i].normal_y, hn[i].normal_z, true);
  return s.json(true);
}

}  // namespace

int run_case(const Args& a) {
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

  std::vector<std::uint8_t> rgb;  // --color: the current RGB frame
  std::ostringstream series;  // --extract-every: [frame, triangles, checksum, cloud points, checksum]
  auto run_frames = [&](auto&& kf, auto&& pose_of, auto&& extract) {
    const int n = a.stop > 0 ? std::min(a.stop, a.frames) : a.frames;
    for (int k = 0; k < n; ++k) {
      Eigen::Affine3f p = trajectoryPose(traj, k, a.frames);
      gt.push_back(p);
      renderDepth(scene, in, p, a.noise < 0 ? -1 : a.noise * 100003 + k, depth);
      d_depth.upload(depth.data(), in.cols * sizeof(std::uint16_t), in.rows, in.cols);
      if (a.color) renderColor(scene, in, p, rgb);
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
      if (a.extract_every > 0 && (k + 1) % a.extract_every == 0) extract(k);
    }
  };

  if (!ls) {
    pcl::gpu::KinfuTracker kf(in.rows, in.cols);
    kf.setDepthIntrinsics(in.fx, in.fy, in.cx, in.cy);
    kf.volume().setSize(vsize);
    kf.setInitalCameraPose(trajectoryPose(traj, 0, a.frames));
    kf.setDepthTruncationForICP(a.icp_trunc);
    kf.setCameraMovementThreshold(0.f);
    if (a.color) kf.initColorIntegration();
    pcl::gpu::KinfuTracker::View d_color;
    pcl::gpu::MarchingCubes smc;
    pcl::gpu::DeviceArray<pcl::PointXYZ> sbuf, scbuf;
    run_frames([&](auto& d) {
      if (!a.color) return kf(d);
      d_color.upload(rgb.data(), in.cols * 3, in.rows, in.cols);
      return kf(d, d_color);
    }, [&](int) { return kf.getCameraPose(); }, [&](int k) {
      std::vector<pcl::PointXYZ> h, hc;
      smc.run(kf.volume(), sbuf).download(h);
      kf.volume().fetchCloud(scbuf).download(hc);
      series << (series.tellp() > 0 ? "," : "") << "[" << k << "," << h.size() / 3 << "," << pointChecksum(h) << "," << hc.size()
             << "," << pointChecksum(hc) << "]";
    });
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
      {
        std::vector<double> t;
        for (int r = 0; r < 5; ++r) {
          cudaDeviceSynchronize();
          auto t0 = std::chrono::steady_clock::now();
          tris = mc.run(kf.volume(), buf);
          cudaDeviceSynchronize();
          t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(t.begin(), t.end());
        extra << "\"mc_ms\": " << t[2] << ", ";
      }
      std::vector<pcl::PointXYZ> h;
      tris.download(h);
      meshQuality(h, scene, world_from_volume, extra);
      extra << ", ";
    }
    {
      // zero-crossing cloud extraction
      pcl::gpu::DeviceArray<pcl::PointXYZ> cbuf;
      pcl::gpu::DeviceArray<pcl::PointXYZ> cloud = kf.volume().fetchCloud(cbuf);
      {
        std::vector<double> t;
        for (int r = 0; r < 5; ++r) {
          cudaDeviceSynchronize();
          auto t0 = std::chrono::steady_clock::now();
          cloud = kf.volume().fetchCloud(cbuf);
          cudaDeviceSynchronize();
          t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(t.begin(), t.end());
        extra << "\"cloud_ms\": " << t[2] << ", ";
      }
      std::vector<pcl::PointXYZ> hc;
      cloud.download(hc);
      double cx = 0, cy = 0, cz = 0, err = 0;
      for (const auto& p : hc) {
        cx += p.x; cy += p.y; cz += p.z;
        err += std::abs(scene.sdf(Eigen::Vector3f(p.x, p.y, p.z)));
      }
      size_t n = hc.size();
      extra << "\"cloud\": {\"points\": " << n << ", \"centroid\": [" << (n ? cx / n : 0) << "," << (n ? cy / n : 0) << ","
            << (n ? cz / n : 0) << "], \"surface_err_mean\": " << (n ? err / n : 0) << "}, ";
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
    if (a.api) {
      const Eigen::Affine3f I = Eigen::Affine3f::Identity();
      std::ostringstream o;
      pcl::gpu::DeviceArray<pcl::PointXYZ> cb;
      pcl::gpu::DeviceArray<pcl::PointXYZ> cloud = kf.volume().fetchCloud(cb);
      pcl::gpu::DeviceArray<pcl::Normal> nb;
      kf.volume().fetchNormals(cloud, nb);
      std::vector<pcl::PointXYZ> hc;
      std::vector<pcl::Normal> hn;
      cloud.download(hc);
      nb.download(hn);
      Oriented cn;
      for (size_t i = 0; i < std::min(hc.size(), hn.size()); ++i)
        cn.add(scene, I, hc[i].x, hc[i].y, hc[i].z, hn[i].normal_x, hn[i].normal_y, hn[i].normal_z, true);
      o << "\"api\": {\"cloud_normals\": " << cn.json(true) << ", ";
      if (a.color) {
        pcl::gpu::DeviceArray<pcl::RGB> colb;
        kf.colorVolume().fetchColors(cloud, colb);
        std::vector<pcl::RGB> hcol;
        colb.download(hcol);
        o << "\"cloud_colors\": " << colorStats(I, hc, hcol, true) << ", ";
      }
      pcl::gpu::KinfuTracker::View view, sview;
      kf.getImage(view);
      rc.generateSceneView(sview, Eigen::Vector3f(1.0f, 0.6f, 0.4f));
      o << "\"view\": " << saveView(a.out + "/view.bin", view) << ", \"scene_view\": " << saveView(a.out + "/scene_view.bin", sview)
        << ", \"raycast_maps\": " << mapStats(scene, I, rc.getVertexMap(), rc.getNormalMap(), true)
        << ", \"last_frame\": " << lastFrameStats(scene, I, kf) << ", ";
      pcl::PointCloud<pcl::PointXYZ> hostc;
      kf.volume().fetchCloudHost(hostc, true);
      std::vector<float> tsdf;
      std::vector<short> w;
      kf.volume().downloadTsdfAndWeighs(tsdf, w);
      long long nw = 0;
      double sabs = 0;
      for (size_t i = 0; i < std::min(tsdf.size(), w.size()); ++i)
        if (w[i] > 0) { ++nw; sabs += std::abs(tsdf[i]); }
      o << "\"host_cloud_points\": " << hostc.size() << ", \"tsdf_download\": {\"weighted\": " << nw << ", \"abs_sum\": " << num(sabs)
        << "}}, ";
      extra << o.str();
    }
  } else {
    pcl::gpu::kinfuLS::KinfuTracker kf(vsize, a.shift, in.rows, in.cols);
    kf.setDepthIntrinsics(in.fx, in.fy, in.cx, in.cy);
    kf.setInitialCameraPose(trajectoryPose(traj, 0, a.frames));
    kf.setDepthTruncationForICP(a.icp_trunc);
    kf.setCameraMovementThreshold(0.f);
    if (a.color) kf.initColorIntegration();
    pcl::gpu::kinfuLS::KinfuTracker::View d_color;
    pcl::gpu::kinfuLS::MarchingCubes smc;
    pcl::gpu::DeviceArray<pcl::PointXYZ> sbuf;
    run_frames([&](auto& d) {
      if (!a.color) return kf(d);
      d_color.upload(rgb.data(), in.cols * 3, in.rows, in.cols);
      return kf(d, d_color);
    }, [&](int) { return kf.getCameraPose(); }, [&](int k) {
      std::vector<pcl::PointXYZ> h;
      smc.run(kf.volume(), sbuf).download(h);
      series << (series.tellp() > 0 ? "," : "") << "[" << k << "," << h.size() / 3 << "," << pointChecksum(h) << "]";
    });
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
    if (a.api) {
      // the cyclical volume's cloud and ray-cast maps are volume-local: their statistics are compared with the
      // original's; the last frame is in the tracker's world frame and is checked against the scene
      const Eigen::Affine3f scene_from_tracker = gt[0] * est[0].inverse();
      std::ostringstream o;
      pcl::gpu::DeviceArray<pcl::PointXYZ> cb;
      pcl::gpu::DeviceArray<pcl::PointXYZ> cloud = kf.volume().fetchCloud(cb);
      pcl::gpu::DeviceArray<pcl::Normal> nb;
      kf.volume().fetchNormals(cloud, nb);
      std::vector<pcl::PointXYZ> hc;
      std::vector<pcl::Normal> hn;
      cloud.download(hc);
      nb.download(hn);
      Oriented cn;
      for (size_t i = 0; i < std::min(hc.size(), hn.size()); ++i)
        cn.add(scene, scene_from_tracker, hc[i].x, hc[i].y, hc[i].z, hn[i].normal_x, hn[i].normal_y, hn[i].normal_z, false);
      o << "\"api\": {\"cloud_normals\": " << cn.json(false) << ", ";
      if (a.color) {
        pcl::gpu::DeviceArray<pcl::RGB> colb;
        kf.colorVolume().fetchColors(cloud, colb);
        std::vector<pcl::RGB> hcol;
        colb.download(hcol);
        o << "\"cloud_colors\": " << colorStats(scene_from_tracker, hc, hcol, false) << ", ";
      }
      pcl::gpu::kinfuLS::RayCaster rc(in.rows, in.cols, in.fx, in.fy, in.cx, in.cy);
      rc.run(kf.volume(), kf.getCameraPose(), kf.getCyclicalBufferStructure());
      pcl::gpu::kinfuLS::KinfuTracker::View view, sview;
      kf.getImage(view);
      rc.generateSceneView(sview, Eigen::Vector3f(1.0f, 0.6f, 0.4f));
      o << "\"view\": " << saveView(a.out + "/view.bin", view) << ", \"scene_view\": " << saveView(a.out + "/scene_view.bin", sview)
        << ", \"raycast_maps\": " << mapStats(scene, scene_from_tracker, rc.getVertexMap(), rc.getNormalMap(), false)
        << ", \"last_frame\": " << lastFrameStats(scene, scene_from_tracker, kf) << "}, ";
      extra << o.str();
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
  if (a.extract_every > 0) f << "\"extract_series\": [" << series.str() << "], ";
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
}

// --concurrent "mode:traj:noise,mode:traj:noise,...": run independent trackers, one per host
// thread, at the same time; case k writes to OUT/case<k>/.
int run_concurrent(const Args& base, const std::string& spec) {
  std::vector<Args> cases;
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    Args c = base;
    std::stringstream is(item);
    std::string mode, traj, noise;
    std::getline(is, mode, ':');
    std::getline(is, traj, ':');
    std::getline(is, noise, ':');
    c.mode = mode;
    c.traj = traj;
    c.noise = noise.empty() ? -1 : std::stoi(noise);
    c.out = base.out + "/case" + std::to_string(cases.size());
    std::filesystem::create_directories(c.out);
    cases.push_back(c);
  }
  std::vector<int> rc(cases.size(), 1);
  std::vector<std::thread> th;
  for (size_t k = 0; k < cases.size(); ++k)
    th.emplace_back([&, k] {
      try { rc[k] = run_case(cases[k]); }
      catch (const std::exception& e) { std::cerr << "case " << k << ": " << e.what() << std::endl; rc[k] = 1; }
    });
  for (auto& t : th) t.join();
  for (int r : rc)
    if (r) return 1;
  return 0;
}

int main(int argc, char** argv) try {
  std::string concurrent;
  std::vector<char*> rest = {argv[0]};
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--concurrent" && i + 1 < argc) concurrent = argv[++i];
    else rest.push_back(argv[i]);
  }
  Args a = parse((int)rest.size(), rest.data());
  return concurrent.empty() ? run_case(a) : run_concurrent(a, concurrent);
} catch (const std::exception& e) {
  std::cerr << "kinfu_replay: " << e.what() << std::endl;
  return 1;
}

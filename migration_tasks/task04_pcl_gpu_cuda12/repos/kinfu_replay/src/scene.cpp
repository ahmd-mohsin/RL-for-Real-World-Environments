#include "scene.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace replay {
namespace {

float sdBox(const Eigen::Vector3f& p, const Eigen::Vector3f& half) {
  Eigen::Vector3f q = p.cwiseAbs() - half;
  return q.cwiseMax(0.f).norm() + std::min(std::max(q.x(), std::max(q.y(), q.z())), 0.f);
}

float sdSphere(const Eigen::Vector3f& p, const Eigen::Vector3f& c, float r) { return (p - c).norm() - r; }

float sdTorus(const Eigen::Vector3f& p, float R, float r) {
  float qx = std::sqrt(p.x() * p.x() + p.z() * p.z()) - R;
  return std::sqrt(qx * qx + p.y() * p.y()) - r;
}

float sdCylinderY(const Eigen::Vector3f& p, const Eigen::Vector3f& c, float r, float h) {
  Eigen::Vector3f d = p - c;
  float dr = std::sqrt(d.x() * d.x() + d.z() * d.z()) - r;
  float dy = std::abs(d.y()) - h;
  return std::min(std::max(dr, dy), 0.f) + std::sqrt(std::max(dr, 0.f) * std::max(dr, 0.f) + std::max(dy, 0.f) * std::max(dy, 0.f));
}

Eigen::Vector3f rotY(const Eigen::Vector3f& p, float a) {
  float c = std::cos(a), s = std::sin(a);
  return {c * p.x() + s * p.z(), p.y(), -s * p.x() + c * p.z()};
}

// counter-based hash noise (deterministic across platforms)
std::uint32_t hash32(std::uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}
float gauss(std::uint32_t a, std::uint32_t b) {
  float u1 = (hash32(a * 2654435761U ^ b) + 0.5f) / 4294967296.f;
  float u2 = (hash32(b * 2246822519U ^ a ^ 0x9e3779b9U) + 0.5f) / 4294967296.f;
  return std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * u2);
}

}  // namespace

Scene::Scene(const std::string& name) : name_(name) {
  if (name != "room" && name != "corridor") throw std::runtime_error("unknown scene " + name);
}

float Scene::sdf(const Eigen::Vector3f& p) const {
  if (name_ == "room") {
    // inside of a box (walls), plus objects
    float walls = -sdBox(p - Eigen::Vector3f(1.5f, 1.4f, 1.6f), Eigen::Vector3f(1.45f, 0.95f, 1.35f));
    float d = walls;
    d = std::min(d, sdSphere(p, {1.05f, 1.75f, 2.05f}, 0.32f));
    d = std::min(d, sdSphere(p, {2.05f, 1.95f, 1.55f}, 0.22f));
    d = std::min(d, sdBox(rotY(p - Eigen::Vector3f(1.6f, 2.05f, 2.25f), 0.55f), {0.28f, 0.3f, 0.18f}) - 0.02f);
    d = std::min(d, sdTorus(rotY(p - Eigen::Vector3f(1.55f, 1.25f, 2.5f), 0.3f), 0.28f, 0.07f));
    d = std::min(d, sdCylinderY(p, {0.65f, 1.9f, 1.5f}, 0.12f, 0.45f));
    return d;
  }
  // gallery along +x (9 m): floor and ceiling, a back wall with relief and, in front of it and on the
  // floor, irregularly placed objects, so that ICP is constrained along the walking direction
  auto u = [](std::uint32_t k, std::uint32_t salt) { return (hash32(k * 2654435761U ^ salt) & 0xffffU) / 65535.f; };
  float relief_floor = 0.03f * std::sin(4.1f * p.x()) * std::cos(3.3f * p.z());
  float relief_wall = 0.04f * std::sin(5.3f * p.x() + 1.1f) * std::cos(4.7f * p.y() + 0.4f);
  float d = std::min(2.25f + relief_floor - p.y(), p.y() - 0.35f);
  d = std::min(d, 3.0f - relief_wall - p.z());
  d = std::min(d, p.z() + 0.6f);
  for (std::uint32_t k = 0; k < 30; ++k) {
    Eigen::Vector3f c(0.15f + 0.3f * k + 0.12f * u(k, 1), 0.6f + 1.35f * u(k, 2), 2.55f + 0.3f * u(k, 3));
    float a = 0.08f + 0.12f * u(k, 4);
    if (k % 3 == 0) d = std::min(d, sdSphere(p, c, a));
    else if (k % 3 == 1) d = std::min(d, sdBox(rotY(p - c, 3.f * u(k, 5)), {a, 0.6f * a + 0.05f, 0.7f * a}));
    else d = std::min(d, sdCylinderY(p, c, 0.5f * a + 0.03f, a + 0.1f));
  }
  for (std::uint32_t k = 0; k < 11; ++k) {
    float h = 0.15f + 0.25f * u(k, 6);
    Eigen::Vector3f c(0.4f + 0.85f * k + 0.2f * u(k, 7), 2.25f - h, 1.7f + 0.8f * u(k, 8));
    d = std::min(d, sdBox(rotY(p - c, 2.f * u(k, 9)), {0.12f + 0.1f * u(k, 10), h, 0.1f + 0.08f * u(k, 11)}) - 0.01f);
  }
  return d;
}

Eigen::Affine3f trajectoryPose(const std::string& traj, int k, int n) {
  float t = n > 1 ? float(k) / float(n - 1) : 0.f;
  Eigen::Affine3f pose = Eigen::Affine3f::Identity();
  if (traj == "orbit") {
    // a slow sweep in front of the objects (small inter-frame motion, as KinFu needs)
    float a = 0.35f * std::sin(6.2831853f * t);
    float b = 0.12f * std::sin(12.566371f * t);
    pose.linear() = (Eigen::AngleAxisf(a, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(b, Eigen::Vector3f::UnitX())).toRotationMatrix();
    pose.translation() = Eigen::Vector3f(1.5f + 0.45f * std::sin(6.2831853f * t), 1.35f + 0.1f * std::sin(18.849556f * t),
                                         0.35f + 0.25f * (1.f - std::cos(6.2831853f * t)));
  } else if (traj == "static") {
    pose.translation() = Eigen::Vector3f(1.5f, 1.35f, 0.35f);
  } else if (traj == "walk") {
    // walk along the gallery (x from 0.9 to 6.9 m), facing the back wall, with a slight wobble
    float yaw = 0.15f * std::sin(12.566371f * t);
    pose.linear() = (Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitY()) *
                     Eigen::AngleAxisf(0.06f * std::sin(18.849556f * t), Eigen::Vector3f::UnitX())).toRotationMatrix();
    pose.translation() = Eigen::Vector3f(0.9f + 6.0f * t, 1.3f + 0.04f * std::sin(31.4159f * t), 0.55f + 0.15f * std::sin(6.2831853f * t));
  } else {
    throw std::runtime_error("unknown trajectory " + traj);
  }
  return pose;
}

void renderDepth(const Scene& scene, const Intrinsics& in, const Eigen::Affine3f& cam_to_world, int noise_seed,
                 std::vector<std::uint16_t>& depth) {
  depth.assign((size_t)in.rows * in.cols, 0);
  const Eigen::Matrix3f R = cam_to_world.linear();
  const Eigen::Vector3f o = cam_to_world.translation();
#pragma omp parallel for schedule(dynamic, 4)
  for (int v = 0; v < in.rows; ++v) {
    for (int u = 0; u < in.cols; ++u) {
      Eigen::Vector3f dc((u - in.cx) / in.fx, (v - in.cy) / in.fy, 1.f);
      Eigen::Vector3f dir = R * dc.normalized();
      float t = 0.f;
      bool hit = false;
      for (int it = 0; it < 256 && t < 6.f; ++it) {
        float d = scene.sdf(o + t * dir);
        if (d < 1e-4f) { hit = true; break; }
        t += 0.8f * d;  // the relief makes the corridor SDF non-exact
      }
      if (!hit) continue;
      float z = t * dir.dot(R.col(2));  // depth along the camera z axis
      if (noise_seed >= 0) {
        float sigma = 0.0012f + 0.0019f * (z - 0.4f) * (z - 0.4f);  // Kinect-like axial noise
        z += sigma * gauss((std::uint32_t)(v * in.cols + u), (std::uint32_t)noise_seed);
      }
      if (z > 0.3f && z < 4.5f) depth[(size_t)v * in.cols + u] = (std::uint16_t)std::lround(z * 1000.f);
    }
  }
}

}  // namespace replay

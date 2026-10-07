// Analytic test scenes (signed distance functions) and a deterministic depth renderer.
//
// Coordinates are KinFu's volume frame: metres, x right, y down, z forward,
// the volume spans [0, size]^3. Depth images are 16-bit millimetres along the
// camera z axis, 0 where the ray hits nothing within range.
#pragma once

#include <Eigen/Geometry>
#include <cstdint>
#include <string>
#include <vector>

namespace replay {

struct Intrinsics {
  int rows = 480, cols = 640;
  float fx = 525.f, fy = 525.f, cx = 319.5f, cy = 239.5f;
};

class Scene {
 public:
  // "room": a 3 m cube with objects (KinFu); "corridor": a 9 m long gallery with objects (KinFu large scale)
  explicit Scene(const std::string& name);
  float sdf(const Eigen::Vector3f& p) const;
  const std::string& name() const { return name_; }

 private:
  std::string name_;
};

// camera-to-world pose of frame k for a named trajectory
Eigen::Affine3f trajectoryPose(const std::string& traj, int k, int nframes);

// procedural surface color (RGB) of the scenes at a world point
void sceneColor(const Eigen::Vector3f& p, std::uint8_t rgb[3]);

// outward unit normal of the scene at p (normalized SDF gradient)
Eigen::Vector3f sceneNormal(const Scene& scene, const Eigen::Vector3f& p);

// render one RGB frame (3 bytes per pixel, black where the ray hits nothing within range)
void renderColor(const Scene& scene, const Intrinsics& in, const Eigen::Affine3f& cam_to_world, std::vector<std::uint8_t>& rgb);

// render one depth frame (mm); noise_seed < 0 means noise free
void renderDepth(const Scene& scene, const Intrinsics& in, const Eigen::Affine3f& cam_to_world, int noise_seed,
                 std::vector<std::uint16_t>& depth);

}  // namespace replay

// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace kimera_pgmo {
class DeltaCompression;
}

namespace hydra {

class VolumetricMap;
class ProjectiveIntegrator;
class MeshIntegrator;
class MeshSegmenter;
class CameraLidarFusion;

/**
 * @brief Config for the HydraMeshMapper facade.
 */
struct HydraMeshMapperConfig {
  // --- Volumetric map ---
  float voxel_size = 0.1f;
  float truncation_distance = 0.3f;
  int voxels_per_side = 16;

  // --- Camera intrinsics (used by CameraLidarFusion sensor) ---
  int width = -1;
  int height = -1;
  float fx = -1.f;
  float fy = -1.f;
  float cx = -1.f;
  float cy = -1.f;

  // --- Camera extrinsics: cam2body_{rotation, translation}
  //     encodes body_T_sensor (i.e. sensor→body transform).
  Eigen::Quaterniond cam2body_rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d cam2body_translation = Eigen::Vector3d::Zero();

  // --- TSDF integrator ---
  float max_weight = 10000.f;
  int integration_threads = 4;

  // --- Mesh integrator ---
  float mesh_min_weight = 1e-4f;

  // --- PGMO mesh compression resolution (metres).
  //     Used by DeltaCompression to reduce vertex count. ---
  double pgmo_mesh_resolution = 0.1;

  // --- MeshSegmenter ---
  float cluster_tolerance = 0.25f;
  size_t min_cluster_size = 40;
  size_t max_cluster_size = 100000;
  // Semantic labels to segment objects from. Default: {0} = all unlabelled geometry.
  std::set<uint32_t> object_labels = {0};
};

/**
 * @brief Per-object cluster returned by extractObjects().
 */
struct ObjectCluster {
  uint32_t label = 0;
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  Eigen::MatrixXd points;   // (N, 3) float64 world frame
  Eigen::MatrixXi colors;   // (N, 3) int32 RGB
  Eigen::VectorXf semantic_feature;  // CLIP embedding; empty if none
};

/**
 * @brief Offline-mode wrapper around Hydra's metric-semantic reconstruction.
 *
 * Pipeline:
 *   1. Call addFrame() once per sensor timestep.
 *   2. Call finalizeMesh() after the last frame.
 *   3. Call extractObjects() to run MeshSegmenter on the complete mesh and
 *      retrieve object clusters.
 *
 * The resulting ObjectCluster list maps directly to ROMAN SegmentMinimalData.
 */
class HydraMeshMapper {
 public:
  explicit HydraMeshMapper(const HydraMeshMapperConfig& config);
  ~HydraMeshMapper();

  /**
   * @brief Integrate one sensor frame into the volumetric map.
   *
   * @param timestamp_ns     Frame timestamp in nanoseconds.
   * @param world_T_body     4x4 world-from-body pose (SE3).
   * @param color_image      H x W x 3 uint8 OpenCV BGR image.
   * @param lidar_pts        N x 3 float32 LiDAR points in camera optical frame.
   * @param features_mask    Optional H x W uint16: pixel → semantic label
   *                         (0 = background).  Pass empty Mat to skip.
   * @param semantic_features Optional label → CLIP embedding map. Pass empty
   *                          map to skip.
   */
  void addFrame(
      uint64_t timestamp_ns,
      const Eigen::Isometry3d& world_T_body,
      const cv::Mat& color_image,
      const Eigen::MatrixXf& lidar_pts,
      const cv::Mat& features_mask,
      const std::unordered_map<uint16_t, Eigen::VectorXf>& semantic_features);

  /**
   * @brief Run a final full mesh generation over all accumulated TSDF blocks.
   * Call once after the last addFrame().
   */
  void finalizeMesh();

  /**
   * @brief Run MeshSegmenter on the complete accumulated mesh and return
   *        per-object clusters.
   *
   * @param final_timestamp_ns  Timestamp to stamp the extraction (nanoseconds).
   * @return List of ObjectCluster, one per detected object.
   */
  std::vector<ObjectCluster> extractObjects(uint64_t final_timestamp_ns);

 private:
  HydraMeshMapperConfig config_;
  std::unique_ptr<VolumetricMap> map_;
  std::unique_ptr<ProjectiveIntegrator> tsdf_integrator_;
  std::unique_ptr<MeshIntegrator> mesh_integrator_;
  std::unique_ptr<MeshSegmenter> segmenter_;
  std::unique_ptr<kimera_pgmo::DeltaCompression> mesh_compression_;
  std::shared_ptr<CameraLidarFusion> sensor_;
};

}  // namespace hydra

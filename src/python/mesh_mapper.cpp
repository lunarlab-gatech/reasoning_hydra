// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "hydra/python/mesh_mapper.h"

#define PCL_NO_PRECOMPILE
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#undef PCL_NO_PRECOMPILE

#include <glog/logging.h>
#include <kimera_pgmo/compression/delta_compression.h>
#include <kimera_pgmo/hashing.h>

#include "hydra/frontend/mesh_segmenter.h"
#include "hydra/input/camera_lidar_fusion.h"
#include "hydra/input/input_data.h"
#include "hydra/reconstruction/mesh_integrator.h"
#include "hydra/reconstruction/mesh_integrator_config.h"
#include "hydra/reconstruction/projective_integrator.h"
#include "hydra/reconstruction/projective_integrator_config.h"
#include "hydra/reconstruction/volumetric_map.h"
#include "hydra/reconstruction/voxel_types.h"
#include "hydra/utils/pgmo_mesh_interface.h"
#include "hydra/utils/pgmo_mesh_traits.h"

namespace hydra {

namespace {

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

CameraLidarFusion::Config buildSensorConfig(const HydraMeshMapperConfig& cfg) {
  CameraLidarFusion::Config sc;
  sc.width = cfg.width;
  sc.height = cfg.height;
  sc.fx = cfg.fx;
  sc.fy = cfg.fy;
  sc.cx = cfg.cx;
  sc.cy = cfg.cy;
  sc.cam2body_rotation = cfg.cam2body_rotation;
  sc.cam2body_translation = cfg.cam2body_translation;
  return sc;
}

ProjectiveIntegratorConfig buildTsdfConfig(const HydraMeshMapperConfig& cfg) {
  ProjectiveIntegratorConfig tc;
  tc.max_weight = cfg.max_weight;
  tc.num_threads = cfg.integration_threads;
  return tc;
}

MeshIntegratorConfig buildMeshConfig(const HydraMeshMapperConfig& cfg) {
  MeshIntegratorConfig mc;
  mc.min_weight = cfg.mesh_min_weight;
  return mc;
}

VolumetricMap::Config buildMapConfig(const HydraMeshMapperConfig& cfg) {
  VolumetricMap::Config mc;
  mc.voxel_size = cfg.voxel_size;
  mc.voxels_per_side = cfg.voxels_per_side;
  mc.truncation_distance = cfg.truncation_distance;
  return mc;
}

MeshSegmenter::Config buildSegmenterConfig(const HydraMeshMapperConfig& cfg) {
  MeshSegmenter::Config sc;
  sc.cluster_tolerance = cfg.cluster_tolerance;
  sc.min_cluster_size = cfg.min_cluster_size;
  sc.max_cluster_size = cfg.max_cluster_size;
  sc.labels = cfg.object_labels;
  return sc;
}

}  // namespace

// ---------------------------------------------------------------------------
// HydraMeshMapper
// ---------------------------------------------------------------------------

HydraMeshMapper::HydraMeshMapper(const HydraMeshMapperConfig& config)
    : config_(config) {
  // Create the volumetric map.  Semantics are needed so per-vertex CLIP
  // features end up in the SemanticLayer and flow through to the mesh blocks.
  map_ = std::make_unique<VolumetricMap>(buildMapConfig(config_),
                                         /*with_semantics=*/true,
                                         /*with_tracking=*/false);

  tsdf_integrator_ =
      std::make_unique<ProjectiveIntegrator>(buildTsdfConfig(config_));
  mesh_integrator_ = std::make_unique<MeshIntegrator>(buildMeshConfig(config_));

  segmenter_ = std::make_unique<MeshSegmenter>(buildSegmenterConfig(config_));

  // DeltaCompression flattens the block-based MeshLayer into the PCL-based
  // MeshDelta that MeshSegmenter::detect() expects.
  mesh_compression_ = std::make_unique<kimera_pgmo::DeltaCompression>(
      config_.pgmo_mesh_resolution);

  sensor_ = std::make_shared<CameraLidarFusion>(buildSensorConfig(config_));
}

HydraMeshMapper::~HydraMeshMapper() = default;

// ---------------------------------------------------------------------------
// addFrame
// ---------------------------------------------------------------------------

void HydraMeshMapper::addFrame(
    uint64_t timestamp_ns,
    const Eigen::Isometry3d& world_T_body,
    const cv::Mat& color_image,
    const Eigen::MatrixXf& lidar_pts,
    const cv::Mat& features_mask,
    const std::unordered_map<uint16_t, Eigen::VectorXf>& semantic_features) {

  // Build InputData.
  auto input = std::make_shared<InputData>(sensor_);
  input->timestamp_ns = timestamp_ns;
  input->world_T_body = world_T_body;

  // Color image (H x W x 3 uint8; CameraLidarFusion expects BGR).
  input->color_image = color_image.clone();

  // Build PCL PointXYZRGBL cloud from N x 3 LiDAR matrix (camera frame).
  // CameraLidarFusion::finalizeRepresentations() will project each point to
  // the image plane to assign colors and build the vertex_map.
  auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBL>>();
  cloud->reserve(static_cast<size_t>(lidar_pts.rows()));
  for (int i = 0; i < lidar_pts.rows(); ++i) {
    pcl::PointXYZRGBL pt;
    pt.x = lidar_pts(i, 0);
    pt.y = lidar_pts(i, 1);
    pt.z = lidar_pts(i, 2);
    pt.r = pt.g = pt.b = 0;  // filled by finalizeRepresentations
    pt.label = 0;
    cloud->push_back(pt);
  }
  input->pointcloud = cloud;

  // Semantic features mask (pixel → label) and per-label CLIP embeddings.
  if (!features_mask.empty()) {
    input->features_mask = features_mask.clone();
  }
  if (!semantic_features.empty()) {
    input->semantic_features = semantic_features;
  }

  // Project LiDAR points to image plane → fills vertex_map, range_image,
  // and assigns per-point color from the RGB image.
  if (!sensor_->finalizeRepresentations(*input)) {
    LOG(WARNING) << "[HydraMeshMapper] finalizeRepresentations() failed at t="
                 << timestamp_ns << "; skipping frame.";
    return;
  }

  // TSDF integration: fuse the new observation into the volumetric map.
  const BlockIndices updated = tsdf_integrator_->updateMap(*input, *map_);

  // Incremental mesh update: run marching cubes only on updated blocks.
  if (!updated.empty()) {
    mesh_integrator_->generateMesh(*map_, /*only_updated=*/true,
                                   /*clear_updated_flag=*/true);
  }
}

// ---------------------------------------------------------------------------
// finalizeMesh
// ---------------------------------------------------------------------------

void HydraMeshMapper::finalizeMesh() {
  // Full mesh pass over ALL TSDF blocks (catches any remaining updates).
  mesh_integrator_->generateMesh(*map_, /*only_updated=*/false,
                                 /*clear_updated_flag=*/true);
}

// ---------------------------------------------------------------------------
// extractObjects
// ---------------------------------------------------------------------------

std::vector<ObjectCluster> HydraMeshMapper::extractObjects(
    uint64_t final_timestamp_ns) {

  // Step 1: Compress the full MeshLayer into a kimera_pgmo::MeshDelta.
  //   PgmoMeshLayerInterface wraps the entire MeshLayer (all blocks).
  //   DeltaCompression flattens it into a PCL vertex cloud with semantic
  //   labels and feature vectors, which MeshSegmenter::detect() expects.
  PgmoMeshLayerInterface interface(map_->getMeshLayer());
  kimera_pgmo::HashedIndexMapping remapping;
  auto mesh_delta = mesh_compression_->update(interface, final_timestamp_ns,
                                               &remapping);

  if (!mesh_delta || mesh_delta->vertex_updates->empty()) {
    LOG(WARNING) << "[HydraMeshMapper] extractObjects: mesh delta is empty.";
    return {};
  }

  // Step 2: Run MeshSegmenter::detect() to obtain Euclidean clusters grouped
  //   by semantic label.  The second argument (robot position) is optional;
  //   we pass nullopt for batch/offline use.
  const auto label_clusters = segmenter_->detect(
      final_timestamp_ns, *mesh_delta, /*robot_pos=*/std::nullopt);

  // Step 3: Convert LabelClusters → ObjectCluster (Python-friendly).
  std::vector<ObjectCluster> result;
  result.reserve(32);  // reasonable initial capacity

  for (const auto& [label, clusters] : label_clusters) {
    for (const auto& cluster : clusters) {
      ObjectCluster oc;
      oc.label = label;
      oc.centroid = cluster.centroid;
      if (cluster.semantic_feature) {
        oc.semantic_feature = *cluster.semantic_feature;
      }

      // Extract the world-frame 3D positions and colors for each vertex in
      // the cluster.  cluster.indices stores *global* vertex indices;
      // delta.getLocalIndex() maps global → local PCL cloud index.
      const size_t n = cluster.indices.size();
      oc.points.resize(n, 3);
      oc.colors.resize(n, 3);
      for (size_t i = 0; i < n; ++i) {
        const auto local_idx = mesh_delta->getLocalIndex(cluster.indices[i]);
        const auto& p = mesh_delta->vertex_updates->at(local_idx);
        oc.points(i, 0) = static_cast<double>(p.x);
        oc.points(i, 1) = static_cast<double>(p.y);
        oc.points(i, 2) = static_cast<double>(p.z);
        oc.colors(i, 0) = static_cast<int>(p.r);
        oc.colors(i, 1) = static_cast<int>(p.g);
        oc.colors(i, 2) = static_cast<int>(p.b);
      }

      result.push_back(std::move(oc));
    }
  }

  return result;
}

}  // namespace hydra

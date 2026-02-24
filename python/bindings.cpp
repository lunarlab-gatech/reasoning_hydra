// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Python bindings for HydraMeshMapper.
// Import as:  import hydra_python as hydra
//
// Typical usage:
//   cfg = hydra.HydraMeshMapperConfig()
//   cfg.voxel_size = 0.05
//   cfg.width = 752; cfg.height = 480
//   cfg.fx = 376.0; cfg.fy = 376.0; cfg.cx = 376.0; cfg.cy = 240.0
//
//   mapper = hydra.HydraMeshMapper(cfg)
//   for t, pose, color, lidar, fmask, sfeats in frames:
//       mapper.add_frame(t, pose, color, lidar, fmask, sfeats)
//   mapper.finalize_mesh()
//
//   clusters = mapper.extract_objects(final_timestamp_ns)
//   for c in clusters:
//       print(c.centroid, c.points.shape, c.semantic_feature.shape)

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <set>
#include <unordered_map>

#include <opencv2/core/mat.hpp>

#include "hydra/python/mesh_mapper.h"

namespace py = pybind11;
using namespace hydra;

// ---------------------------------------------------------------------------
// Numpy → cv::Mat helpers (zero-copy; caller owns the numpy array)
// ---------------------------------------------------------------------------

static cv::Mat numpy_uint8_3ch_to_mat(const py::array_t<uint8_t>& arr) {
  py::buffer_info buf = arr.request();
  if (buf.ndim != 3 || buf.shape[2] != 3) {
    throw std::invalid_argument(
        "color_image must be a (H, W, 3) uint8 numpy array");
  }
  return cv::Mat(static_cast<int>(buf.shape[0]),
                 static_cast<int>(buf.shape[1]),
                 CV_8UC3, buf.ptr);
}

static cv::Mat numpy_uint16_to_mat(const py::array_t<uint16_t>& arr) {
  py::buffer_info buf = arr.request();
  if (buf.ndim != 2) {
    throw std::invalid_argument(
        "features_mask must be a (H, W) uint16 numpy array");
  }
  return cv::Mat(static_cast<int>(buf.shape[0]),
                 static_cast<int>(buf.shape[1]),
                 CV_16UC1, buf.ptr);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(hydra_python, m) {
  m.doc() =
      "Python bindings for Hydra's metric-semantic reconstruction pipeline.\n"
      "Provides HydraMeshMapper: a batch-mode TSDF + mesh + object segmentation "
      "pipeline that produces object clusters compatible with ROMAN SegmentMinimalData.";

  // ---- HydraMeshMapperConfig ------------------------------------------------
  py::class_<HydraMeshMapperConfig>(m, "HydraMeshMapperConfig",
      "Configuration for HydraMeshMapper.")
      .def(py::init<>())
      // Volumetric map
      .def_readwrite("voxel_size", &HydraMeshMapperConfig::voxel_size,
                     "TSDF voxel size in metres.")
      .def_readwrite("truncation_distance",
                     &HydraMeshMapperConfig::truncation_distance,
                     "TSDF truncation distance in metres.")
      .def_readwrite("voxels_per_side", &HydraMeshMapperConfig::voxels_per_side,
                     "Number of voxels per TSDF block side.")
      // Camera intrinsics
      .def_readwrite("width", &HydraMeshMapperConfig::width)
      .def_readwrite("height", &HydraMeshMapperConfig::height)
      .def_readwrite("fx", &HydraMeshMapperConfig::fx)
      .def_readwrite("fy", &HydraMeshMapperConfig::fy)
      .def_readwrite("cx", &HydraMeshMapperConfig::cx)
      .def_readwrite("cy", &HydraMeshMapperConfig::cy)
      // Camera extrinsics as flat quaternion [x,y,z,w] and translation [x,y,z]
      .def_property(
          "cam2body_rotation",
          [](const HydraMeshMapperConfig& c) -> Eigen::Vector4d {
            return c.cam2body_rotation.coeffs();  // [x, y, z, w]
          },
          [](HydraMeshMapperConfig& c, const Eigen::Vector4d& xyzw) {
            c.cam2body_rotation =
                Eigen::Quaterniond(xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
          },
          "Camera-to-body rotation as [x, y, z, w] quaternion.")
      .def_property(
          "cam2body_translation",
          [](const HydraMeshMapperConfig& c) -> Eigen::Vector3d {
            return c.cam2body_translation;
          },
          [](HydraMeshMapperConfig& c, const Eigen::Vector3d& t) {
            c.cam2body_translation = t;
          },
          "Camera-to-body translation vector [x, y, z] in metres.")
      // Integrator
      .def_readwrite("max_weight", &HydraMeshMapperConfig::max_weight)
      .def_readwrite("integration_threads",
                     &HydraMeshMapperConfig::integration_threads)
      .def_readwrite("mesh_min_weight", &HydraMeshMapperConfig::mesh_min_weight)
      // PGMO / DeltaCompression
      .def_readwrite("pgmo_mesh_resolution",
                     &HydraMeshMapperConfig::pgmo_mesh_resolution,
                     "Vertex decimation resolution used by DeltaCompression.")
      // MeshSegmenter
      .def_readwrite("cluster_tolerance",
                     &HydraMeshMapperConfig::cluster_tolerance,
                     "Euclidean cluster tolerance in metres.")
      .def_readwrite("min_cluster_size", &HydraMeshMapperConfig::min_cluster_size)
      .def_readwrite("max_cluster_size", &HydraMeshMapperConfig::max_cluster_size)
      .def_property(
          "object_labels",
          [](const HydraMeshMapperConfig& c) -> std::vector<uint32_t> {
            return std::vector<uint32_t>(c.object_labels.begin(),
                                         c.object_labels.end());
          },
          [](HydraMeshMapperConfig& c, const std::vector<uint32_t>& v) {
            c.object_labels = std::set<uint32_t>(v.begin(), v.end());
          },
          "List of semantic labels to cluster as objects. "
          "Use [0] to cluster all unlabelled geometry.");

  // ---- ObjectCluster --------------------------------------------------------
  py::class_<ObjectCluster>(m, "ObjectCluster",
      "A single object cluster extracted from the 3D mesh.")
      .def_readonly("label", &ObjectCluster::label,
                    "Semantic label of this cluster.")
      .def_readonly("centroid", &ObjectCluster::centroid,
                    "Cluster centroid in world frame, shape (3,) float64.")
      .def_readonly("points", &ObjectCluster::points,
                    "Cluster vertex positions in world frame, shape (N, 3) float64.")
      .def_readonly("colors", &ObjectCluster::colors,
                    "Per-vertex RGB colors, shape (N, 3) int32.")
      .def_readonly("semantic_feature", &ObjectCluster::semantic_feature,
                    "Averaged CLIP embedding for this cluster, shape (D,) float32. "
                    "Empty (size 0) if no semantic features were integrated.");

  // ---- HydraMeshMapper -------------------------------------------------------
  py::class_<HydraMeshMapper>(m, "HydraMeshMapper",
      "Offline metric-semantic TSDF + mesh + object segmentation pipeline.")
      .def(py::init<const HydraMeshMapperConfig&>(), py::arg("config"))

      .def(
          "add_frame",
          [](HydraMeshMapper& self,
             uint64_t timestamp_ns,
             const Eigen::Matrix4d& world_T_body_mat,  // (4,4) float64
             const py::array_t<uint8_t>& color_np,    // (H,W,3) uint8
             const Eigen::MatrixXf& lidar_pts,         // (N,3) float32
             const py::array_t<uint16_t>& features_mask_np,  // (H,W) uint16
             const std::unordered_map<uint16_t, Eigen::VectorXf>&
                 semantic_features) {
            Eigen::Isometry3d world_T_body;
            world_T_body.matrix() = world_T_body_mat;

            cv::Mat color_mat = numpy_uint8_3ch_to_mat(color_np);

            cv::Mat features_mask_mat;
            if (features_mask_np.size() > 0) {
              features_mask_mat = numpy_uint16_to_mat(features_mask_np);
            }

            self.addFrame(timestamp_ns, world_T_body, color_mat, lidar_pts,
                          features_mask_mat, semantic_features);
          },
          py::arg("timestamp_ns"),
          py::arg("world_T_body"),
          py::arg("color_image"),
          py::arg("lidar_pts"),
          py::arg("features_mask"),
          py::arg("semantic_features"),
          "Integrate one sensor frame into the volumetric map.\n\n"
          "Args:\n"
          "  timestamp_ns:      Frame timestamp in nanoseconds (int).\n"
          "  world_T_body:      (4, 4) float64 world-from-body SE3 matrix.\n"
          "  color_image:       (H, W, 3) uint8 numpy BGR image.\n"
          "  lidar_pts:         (N, 3) float32 LiDAR points in camera optical frame.\n"
          "  features_mask:     (H, W) uint16 numpy array; pixel→semantic label.\n"
          "                     Pass np.zeros((H,W), dtype=np.uint16) to skip.\n"
          "  semantic_features: dict[int, np.ndarray] label→CLIP embedding.\n"
          "                     Pass {} to skip.")

      .def("finalize_mesh", &HydraMeshMapper::finalizeMesh,
           "Run a final full mesh generation over all TSDF blocks.\n"
           "Call once after the last add_frame() before extract_objects().")

      .def("extract_objects", &HydraMeshMapper::extractObjects,
           py::arg("final_timestamp_ns"),
           "Run MeshSegmenter on the full accumulated mesh.\n\n"
           "Args:\n"
           "  final_timestamp_ns: Timestamp used to stamp the extraction (int).\n\n"
           "Returns:\n"
           "  list[ObjectCluster]: one cluster per detected object.");
}

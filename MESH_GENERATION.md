Here's a breakdown of where each major stage happens:                                                                                              
                                
  ---                                                                                                                                                
  Metric-Semantic 3D Mesh Construction                                                                                                               
                                                                                                                                                     
  The pipeline fuses CLIP features with RGB-D data through these stages:                                                                             

  1. Input Data                                                                                                                                      
                                                                                                                                                     
  src/reasoning_hydra/include/hydra/input/input_data.h (lines 101-104)
  The InputData struct holds all modalities: color image, depth image, features_mask, and semantic_features (a map of uint16_t → Eigen::VectorXf,
  i.e., CLIP embeddings).

  2. Feature Interpolation (RGB-D + CLIP Fusion)

  src/reasoning_hydra/src/reconstruction/projection_interpolators.cpp (lines 136-150)
  InterpolatorNearest::interpolateFeatures() maps the 2D CLIP features onto 3D points using depth-derived interpolation weights.

  3. Semantic Voxel Integration

  src/reasoning_hydra/src/reconstruction/projective_integrator.cpp
  - getVoxelMeasurement() (lines ~229–272): computes per-voxel measurements combining depth, RGB, label, and CLIP feature vector.
  - updateVoxel() (lines ~274–321): fuses measurements into TSDF voxels.

  4. Feature Averaging into Voxels

  src/reasoning_hydra/src/reconstruction/semantic_integrator.cpp (lines 95-122)
  MLESemanticIntegrator::updateLikelihoods() performs running-average fusion of CLIP feature vectors into each SemanticVoxel.feature_vector across
  observations.

  5. Mesh Generation

  src/reasoning_hydra/include/hydra/reconstruction/mesh_integrator.h
  MeshIntegrator runs marching cubes over the semantic voxel grid to produce the final 3D mesh with per-vertex semantic features.

  ---
  Object Extraction with Semantic Classes

  Objects are extracted from the completed mesh in:

  src/reasoning_hydra/src/frontend/mesh_segmenter.cpp

  ┌─────────────────────────┬──────────┬────────────────────────────────────────────────────────────────────────────────────────┐
  │        Function         │  Lines   │                                          Role                                          │
  ├─────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ MeshSegmenter::detect() │ ~286–325 │ Top-level detection entry point                                                        │
  ├─────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ getLabelIndices()       │ ~164–194 │ Groups mesh vertices by semantic label                                                 │
  ├─────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ findClusters()          │ ~196–274 │ Euclidean clustering per label; aggregates CLIP features per cluster                   │
  ├─────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ updateGraph()           │ ~367–535 │ Inserts/updates object nodes in the scene graph                                        │
  ├─────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ addNodeToGraph()        │ ~616–659 │ Creates a new object node, assigns semantic label + CLIP feature vector + bounding box │
  ├─────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ mergeActiveNodes()      │ ~537–591 │ Merges overlapping object instances, fusing their feature vectors                      │
  └─────────────────────────┴──────────┴────────────────────────────────────────────────────────────────────────────────────────┘

  The per-object semantic class comes from the label field assigned during getLabelIndices(), while the open-vocabulary CLIP features are aggregated
  in findClusters() and stored via attrs->semantic_feature in addNodeToGraph().
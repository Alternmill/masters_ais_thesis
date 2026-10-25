#!/usr/bin/env bash
set -euo pipefail
D="$1"
MATCHER="${2:-sequential}"
DB="$D/database.db"
mkdir -p "$D/sparse"
rm -f "$DB"

GPU=1
log() { echo "[colmap] $*"; }

log "feature extraction (OPENCV_FISHEYE, single shared camera, refine principal point)"
colmap feature_extractor \
  --database_path "$DB" --image_path "$D/images" \
  --ImageReader.camera_model OPENCV_FISHEYE \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu $GPU \
  --SiftExtraction.max_image_size 1600 \
  --SiftExtraction.estimate_affine_shape 1 \
  --SiftExtraction.domain_size_pooling 1 2>&1 | tail -3 || {
    log "GPU SIFT failed, retrying on CPU"; GPU=0
    colmap feature_extractor --database_path "$DB" --image_path "$D/images" \
      --ImageReader.camera_model OPENCV_FISHEYE --ImageReader.single_camera 1 \
      --SiftExtraction.use_gpu 0 --SiftExtraction.estimate_affine_shape 1 \
      --SiftExtraction.domain_size_pooling 1 2>&1 | tail -3; }

if [ "$MATCHER" = "exhaustive" ]; then
  log "exhaustive matching (all pairs — merges fragments on long/repetitive scans)"
  colmap exhaustive_matcher --database_path "$DB" --SiftMatching.use_gpu $GPU 2>&1 | tail -3
else
  log "sequential matching (video-ordered) with loop closure"
  colmap sequential_matcher \
    --database_path "$DB" \
    --SiftMatching.use_gpu $GPU \
    --SequentialMatching.overlap 15 \
    --SequentialMatching.quadratic_overlap 1 2>&1 | tail -3
fi

log "mapping (refine principal point)"
colmap mapper \
  --database_path "$DB" --image_path "$D/images" \
  --output_path "$D/sparse" \
  --Mapper.ba_refine_principal_point 1 2>&1 | tail -5

log "done. models:"; ls "$D/sparse"

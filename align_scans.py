"""Rigid alignment of one .lcc scan onto another (same room, different session).

Each LCC scan has its own origin, so poses from different scans of the same room are
not comparable. This solves the SE(3) transform T that maps SRC scan coordinates into
REF scan coordinates, so a camera pose chosen in one scan can be reused in the others.

Both scans come out of the scanner gravity-aligned, so the search is effectively
yaw + translation: we sweep yaw, run coarse point-to-point ICP from each start, keep
the best, then refine with point-to-plane ICP.

Usage:
    python align_scans.py --ref scans/bathroom_1/lcc-result \
        --src scans/bathroom_2/lcc-result --out align/T_b2_to_b1.json

A pose transforms as w2c_src = w2c_ref @ T (T maps src->ref).
"""
import argparse
import json

import numpy as np
from lcc import LccScan


def scan_cloud(path, lod=0, margin=3.0, voxel=0.04, max_pts=400000, seed=0):
    """Decode a scan to an open3d cloud, cropped to its trajectory bbox + margin."""
    import open3d as o3d
    scan = LccScan.open(path)
    sp = scan.decode(lod=lod)
    centers = scan.pose_centers().astype(np.float64)
    lo, hi = centers.min(0) - margin, centers.max(0) + margin
    sp = sp.crop_box((lo + hi) / 2, (hi - lo) / 2)
    pts = np.asarray(sp.means, dtype=np.float64)
    if len(pts) > max_pts:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_pts, replace=False)]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=30))
    return pc, centers


def yaw_T(deg, t):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T = np.eye(4)
    T[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T[:3, 3] = t
    return T


def fpfh(pc, voxel):
    import open3d as o3d
    return o3d.pipelines.registration.compute_fpfh_feature(
        pc, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 12, max_nn=100))


def global_register(ref_pc, src_pc, voxel):
    """Feature-based RANSAC: no yaw prior, recovers gross rotation reliably."""
    import open3d as o3d
    r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pc, ref_pc, fpfh(src_pc, voxel), fpfh(ref_pc, voxel), True, voxel * 4,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 4)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 0.999))
    print(f"  ransac: fitness {r.fitness:.3f}  rmse {r.inlier_rmse * 100:.1f} cm")
    return np.asarray(r.transformation), r.fitness


def multiscale_icp(ref_pc, src_pc, T, voxel):
    """Coarse-to-fine point-to-plane ICP."""
    import open3d as o3d
    for mult, iters in ((8, 100), (4, 100), (2, 200), (1, 200)):
        r = o3d.pipelines.registration.registration_icp(
            src_pc, ref_pc, voxel * mult, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
        T = r.transformation
    print(f"  icp:    fitness {r.fitness:.3f}  rmse {r.inlier_rmse * 100:.1f} cm")
    return np.asarray(T), float(r.fitness), float(r.inlier_rmse)


def align(ref_pc, src_pc, ref_c, src_c, voxel=0.04, yaw_step=15, use_ransac=True):
    import open3d as o3d
    if use_ransac:
        T_g, fit_g = global_register(ref_pc, src_pc, voxel)
        T_ms, fit, rmse = multiscale_icp(ref_pc, src_pc, T_g, voxel)
        if fit >= 0.85:
            return T_ms, fit, rmse
        print("  ransac start weak — falling back to yaw sweep")
    t0 = ref_c.mean(0) - src_c.mean(0)
    best = None
    for yaw in range(0, 360, yaw_step):
        R = yaw_T(yaw, np.zeros(3))
        T = np.eye(4)
        T[:3, :3] = R[:3, :3]
        T[:3, 3] = ref_c.mean(0) - R[:3, :3] @ src_c.mean(0)
        r = o3d.pipelines.registration.registration_icp(
            src_pc, ref_pc, voxel * 8, T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        score = (r.fitness, -r.inlier_rmse)
        if best is None or score > best[0]:
            best = (score, r.transformation, yaw, r.fitness, r.inlier_rmse)
    _, T_coarse, yaw, fit, rmse = best
    print(f"  coarse: yaw {yaw}deg  fitness {fit:.3f}  rmse {rmse * 100:.1f} cm")
    return multiscale_icp(ref_pc, src_pc, T_coarse, voxel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference scan lcc-result dir")
    ap.add_argument("--src", required=True, help="scan to align onto ref")
    ap.add_argument("--voxel", type=float, default=0.04)
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--yaw-step", type=int, default=15)
    ap.add_argument("--no-ransac", action="store_true", help="skip feature-based init")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    print(f"ref  {a.ref}")
    ref_pc, ref_c = scan_cloud(a.ref, margin=a.margin, voxel=a.voxel)
    print(f"src  {a.src}")
    src_pc, src_c = scan_cloud(a.src, margin=a.margin, voxel=a.voxel)
    print(f"  ref {len(ref_pc.points)} pts | src {len(src_pc.points)} pts")

    T, fit, rmse = align(ref_pc, src_pc, ref_c, src_c, a.voxel, a.yaw_step,
                         use_ransac=not a.no_ransac)
    json.dump({"ref": a.ref, "src": a.src, "T_src_to_ref": T.tolist(),
               "fitness": fit, "inlier_rmse_m": rmse}, open(a.out, "w"), indent=2)
    print("wrote", a.out)
    if fit < 0.5 or rmse > 0.15:
        print("  !! WEAK alignment — check visually before trusting poses")


if __name__ == "__main__":
    main()

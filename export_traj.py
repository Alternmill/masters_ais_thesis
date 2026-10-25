"""
Register a COLMAP OPENCV_FISHEYE reconstruction of the real frames to the metric splat
trajectory: recover the similarity transform by ICP on the camera centres, then export
each frame's pose, the recovered intrinsics, and its splat-frame centre.

Usage:
    python export_traj.py --sparse <colmap>/sparse/0 --scan <lcc-result> --out traj_data.json
"""
import argparse
import json
import re
import numpy as np
import pycolmap
import open3d as o3d

_FISHEYE_PARAMS = ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]


def camera_intrinsics(cam) -> dict:
    """Extract OPENCV_FISHEYE intrinsics from a COLMAP camera into our KI dict."""
    p = list(cam.params)
    model = str(cam.model).split(".")[-1]
    if "FISHEYE" not in model:
        raise ValueError(f"expected an OPENCV_FISHEYE camera, got {model} "
                         f"(re-run COLMAP with --Mapper.ba_refine_principal_point 1)")
    d = dict(zip(_FISHEYE_PARAMS, p))
    return dict(model="OPENCV_FISHEYE", w=int(cam.width), h=int(cam.height),
                fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"],
                k=[d["k1"], d["k2"], d["k3"], d["k4"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", required=True, help="COLMAP sparse/<n> reconstruction dir")
    ap.add_argument("--scan", required=True, help="lcc-result dir (for assets/poses.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inlier-pct", type=float, default=92,
                    help="drop trajectory-centre outliers beyond this percentile before ICP")
    ap.add_argument("--time-sync", type=int, default=0, metavar="N_FRAMES",
                    help="total decoded frames in img/ (f0000..fN-1). Seeds the similarity from time correspondences "
                         "(frame index/N ~ pose index/P; video and poses are both uniform in time) instead of the "
                         "8-flip global ICP — needed when COLMAP reconstructs only a FRAGMENT of a long walk")
    args = ap.parse_args()

    rec = pycolmap.Reconstruction(args.sparse)
    intr = camera_intrinsics(next(iter(rec.cameras.values())))
    print(f"[traj] intrinsics fx={intr['fx']:.1f} cx={intr['cx']:.1f} cy={intr['cy']:.1f} "
          f"k={np.round(intr['k'],3).tolist()}")

    rows = []
    for im in rec.images.values():
        m = re.search(r"f(\d+)", im.name)
        if m is None:
            continue
        rows.append((int(m.group(1)), np.asarray(im.projection_center()), im))
    rows.sort()
    Ccol = np.array([r[1] for r in rows])
    imgs = [r[2] for r in rows]
    print(f"[traj] {len(imgs)} registered frames")

    med = np.median(Ccol, 0)
    dist = np.linalg.norm(Ccol - med, axis=1)
    keep = dist < np.percentile(dist, args.inlier_pct)

    from lcc import LccScan
    Tgt = LccScan.open(args.scan).pose_centers()

    def pc(P):
        q = o3d.geometry.PointCloud()
        q.points = o3d.utility.Vector3dVector(P)
        return q

    sc_s = Ccol[keep].std(0).mean()
    sc_t = Tgt.std(0).mean()
    best = None
    flips = [np.diag(v) for v in [[1, 1, 1.], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
                                  [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]]]
    for flip in flips:
        init = np.eye(4)
        init[:3, :3] = (sc_t / sc_s) * flip
        init[:3, 3] = Tgt.mean(0) - init[:3, :3] @ Ccol[keep].mean(0)
        r = o3d.pipelines.registration.registration_icp(
            pc(Ccol[keep]), pc(Tgt), 0.6, init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))
        if best is None or r.fitness > best[0]:
            best = (r.fitness, np.asarray(r.transformation))
    if args.time_sync:
        idx = np.array([r[0] for r in rows], float)
        k = np.clip(np.round(idx / (args.time_sync - 1) * (len(Tgt) - 1)).astype(int), 0, len(Tgt) - 1)
        src, dst = Ccol, Tgt[k]
        use = np.ones(len(src), bool)
        for _ in range(4):
            mu_s, mu_d = src[use].mean(0), dst[use].mean(0)
            H = (src[use] - mu_s).T @ (dst[use] - mu_d) / use.sum()
            U, D, Vt = np.linalg.svd(H)
            S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(U @ Vt))
            R = (U @ S @ Vt).T
            sc = np.trace(np.diag(D) @ S) / ((src[use] - mu_s) ** 2).sum(1).mean()
            res = np.linalg.norm((sc * (R @ src.T)).T + (mu_d - sc * R @ mu_s) - dst, axis=1)
            use = res < np.percentile(res, 80)
        init = np.eye(4); init[:3, :3] = sc * R; init[:3, 3] = mu_d - sc * R @ mu_s
        lo, hi = max(0, k.min() - 50), min(len(Tgt), k.max() + 50)
        r = o3d.pipelines.registration.registration_icp(
            pc(Ccol[keep]), pc(Tgt[lo:hi]), 0.3, init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))
        print(f"[traj] time-sync Umeyama: median residual {np.median(res):.3f} m (scale {sc:.4f}); "
              f"ICP on poses {lo}-{hi} fitness {r.fitness:.3f}")
        s_icp = float(np.cbrt(np.linalg.det(np.asarray(r.transformation)[:3, :3])))
        if abs(s_icp / sc - 1) > 0.25:
            print(f"[traj] ICP scale {s_icp:.4f} deviates >25 % from time-sync scale -> keeping the time-sync similarity")
            best = (float(np.mean(res < 0.5)), init)
        else:
            best = (r.fitness, np.asarray(r.transformation))
    Tr = best[1]
    s = float(np.cbrt(np.linalg.det(Tr[:3, :3])))
    print(f"[traj] ICP fitness {best[0]:.3f}  scale {s:.4f}")

    frames = []
    for im in imgs:
        Twc = im.cam_from_world().matrix()
        Rwc, twc = Twc[:3, :3], Twc[:3, 3]
        c_col = -Rwc.T @ twc
        c_spl = Tr[:3, :3] @ c_col + Tr[:3, 3]
        frames.append(dict(name=im.name, c_col=c_col.tolist(), R_col=Rwc.tolist(),
                           c_spl=c_spl.tolist()))

    json.dump(dict(scale=s, intrinsics=intr, frames=frames), open(args.out, "w"))
    print(f"[traj] exported {len(frames)} frames (+ intrinsics) to {args.out}")


if __name__ == "__main__":
    main()

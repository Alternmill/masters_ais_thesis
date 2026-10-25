"""
Per-frame pose refinement to pixel accuracy, in three stages:

  1. LightGlue render<->real matches -> fisheye-aware PnP-RANSAC.
  2. solvePnPRefineLM on the RANSAC inliers.
  3. Direct edge-alignment polish: optimize the 6-DoF pose to align the render's
     Canny edges with the real photo's (distance-transform score).

Each frame gets an `edge` score (center edge-match %); gen_realgt gates on it.

Usage:
    python refine_poses.py --scan <lcc-result> --poses realgt_poses.json \
        --frames <img_dir> --out realgt_poses_refined.json [--test]
"""
import argparse
import json
import time
import numpy as np
import cv2
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

from lcc import LccScan
from render import Renderer, Intrinsics

SC = 0.5
DEV = "cuda"


def rodrigues(omega):
    th = np.linalg.norm(omega)
    if th < 1e-9:
        return np.eye(3)
    k = omega / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-inl", type=int, default=20, help="min PnP inliers to attempt edge polish")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--test-frames", default="")
    ap.add_argument("--sequential", action="store_true",
                    help="seed each frame's pose from the previous refined frame (temporal init), "
                         "falling back to the global anchor init; pose is still solved against the "
                         "splat each frame, so no drift. Helps low-texture frames (tile).")
    args = ap.parse_args()

    P = json.load(open(args.poses))
    intr = Intrinsics.from_dict(P["intrinsics"]).scaled(SC)
    Kcv = intr.K()
    Dcv = np.array(intr.k, float).reshape(4, 1)

    scan = LccScan.open(args.scan)
    splats = scan.decode(lod=0)
    splats.quat[:] = 0.0
    splats.quat[:, 0] = 1.0
    renderer = Renderer(splats)

    extractor = SuperPoint(max_num_keypoints=4096).eval().to(DEV)
    matcher = LightGlue(features="superpoint").eval().to(DEV)

    def feats(img):
        t = torch.tensor(img, device=DEV).float().permute(2, 0, 1)[None] / 255
        with torch.no_grad():
            return extractor.extract(t)

    def match(render_rgb, real_rgb):
        with torch.no_grad():
            f0, f1 = feats(render_rgb), feats(real_rgb)
            m = matcher({"image0": f0, "image1": f1})
        f0, f1, m = [rbd(x) for x in (f0, f1, m)]
        mi = m["matches"]
        return (f0["keypoints"][mi[:, 0]].cpu().numpy(),
                f1["keypoints"][mi[:, 1]].cpu().numpy())

    def undist_norm(pts):
        return cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), Kcv, Dcv).reshape(-1, 2)

    def render_depth(w2c):
        return renderer.render(w2c, intr, depth=True)

    def edge_score(rend_rgb, real_edges, real_dt_center, real_dt_periph):
        """Fraction of real edges within 3px of a render edge (center-weighted)."""
        ce = cv2.Canny(cv2.cvtColor(rend_rgb, cv2.COLOR_RGB2GRAY), 80, 160)
        dt = distance_transform_edt(~(ce > 0))
        c = 100 * np.mean(dt[real_dt_center] < 3) if real_dt_center.any() else 0.0
        p = 100 * np.mean(dt[real_dt_periph] < 3) if real_dt_periph.any() else 0.0
        return 0.7 * c + 0.3 * p, c

    def pnp(w2c, real_rgb):
        ninl = 0
        for _ in range(3):
            rgb, depth = render_depth(w2c)
            k0, k1 = match(rgb, real_rgb)
            if len(k0) < 8:
                break
            n0 = undist_norm(k0)
            c2w = np.linalg.inv(w2c)
            Rcw, cc = c2w[:3, :3], c2w[:3, 3]
            P3, P2 = [], []
            for (u, v), (xr, yr), (ur, vr) in zip(k0, n0, k1):
                iu, iv = int(round(u)), int(round(v))
                if not (0 <= iu < intr.w and 0 <= iv < intr.h):
                    continue
                dd = depth[iv, iu]
                if dd <= 0.1 or dd > 40:
                    continue
                P3.append(Rcw @ (dd * np.array([xr, yr, 1.0])) + cc)
                P2.append([ur, vr])
            if len(P3) < 8:
                break
            P3 = np.array(P3)
            P2u = undist_norm(np.array(P2, float))
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                P3, P2u, np.eye(3), None, reprojectionError=0.008,
                iterationsCount=500, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or inl is None or len(inl) < 8:
                break
            rvec, tvec = cv2.solvePnPRefineLM(P3[inl[:, 0]], P2u[inl[:, 0]], np.eye(3), None, rvec, tvec)
            R, _ = cv2.Rodrigues(rvec)
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = tvec[:, 0]
            ninl = len(inl)
        return w2c, ninl

    def edge_polish(w2c, real_rgb):
        real_e = cv2.Canny(cv2.cvtColor(real_rgb, cv2.COLOR_RGB2GRAY), 80, 160)
        yy, xx = np.mgrid[0:intr.h, 0:intr.w]
        rad = np.sqrt(((xx - intr.w / 2) / (intr.w / 2)) ** 2 + ((yy - intr.h / 2) / (intr.h / 2)) ** 2)
        cmask = (real_e > 0) & (rad < 0.6)
        pmask = (real_e > 0) & (rad >= 0.6)
        c2w0 = np.linalg.inv(w2c)

        def score_of(x):
            dR = rodrigues(x[:3] * 0.02)
            c2w = c2w0.copy()
            c2w[:3, :3] = dR @ c2w0[:3, :3]
            c2w[:3, 3] = c2w0[:3, 3] + x[3:] * 0.03
            rgb, _ = renderer.render(np.linalg.inv(c2w), intr, depth=False)
            s, _ = edge_score(rgb, real_e, cmask, pmask)
            return -s

        res = minimize(score_of, np.zeros(6), method="Nelder-Mead",
                       options=dict(maxiter=80, xatol=0.02, fatol=0.3))
        x = res.x
        dR = rodrigues(x[:3] * 0.02)
        c2w = c2w0.copy()
        c2w[:3, :3] = dR @ c2w0[:3, :3]
        c2w[:3, 3] = c2w0[:3, 3] + x[3:] * 0.03
        w2c_new = np.linalg.inv(c2w)
        rgb, _ = renderer.render(w2c_new, intr, depth=False)
        _, center = edge_score(rgb, real_e, cmask, pmask)
        return w2c_new, float(center)

    def edge_of(w2c, real_rgb):
        real_e = cv2.Canny(cv2.cvtColor(real_rgb, cv2.COLOR_RGB2GRAY), 80, 160)
        yy, xx = np.mgrid[0:intr.h, 0:intr.w]
        rad = np.sqrt(((xx - intr.w/2)/(intr.w/2))**2 + ((yy-intr.h/2)/(intr.h/2))**2)
        cm_, pm_ = (real_e > 0) & (rad < 0.6), (real_e > 0) & (rad >= 0.6)
        _, c = edge_score(renderer.render(w2c, intr, depth=False)[0], real_e, cm_, pm_)
        return c

    def refine(w2c, real_rgb):
        w2c, ninl = pnp(w2c, real_rgb)
        edge = 0.0
        if ninl >= args.min_inl:
            edge = edge_of(w2c, real_rgb)
            if edge < 90:
                w2c, edge = edge_polish(w2c, real_rgb)
        return w2c, ninl, edge

    def real_rgb(name):
        return np.array(Image.open(f"{args.frames}/{name}").convert("RGB").resize((intr.w, intr.h)))

    frames = P["frames"]
    if args.test:
        names = args.test_frames.split(",") if args.test_frames else [f["name"] for f in frames[:: max(1, len(frames) // 6)][:6]]
        for fr in [f for f in frames if f["name"] in names]:
            w2c0 = np.array(fr["w2c"])
            real = real_rgb(fr["name"])
            re = cv2.Canny(cv2.cvtColor(real, cv2.COLOR_RGB2GRAY), 80, 160)
            yy, xx = np.mgrid[0:intr.h, 0:intr.w]
            radm = np.sqrt(((xx - intr.w/2)/(intr.w/2))**2 + ((yy-intr.h/2)/(intr.h/2))**2)
            cm_, pm_ = (re > 0) & (radm < 0.6), (re > 0) & (radm >= 0.6)
            _, e0 = edge_score(render_depth(w2c0)[0], re, cm_, pm_)
            w2cp, ninl = pnp(w2c0, real)
            _, e1 = edge_score(render_depth(w2cp)[0], re, cm_, pm_)
            w2ce, e2 = edge_polish(w2cp, real) if ninl >= args.min_inl else (w2cp, e1)
            print(f"{fr['name']}: inl {ninl} | edge% init {e0:.0f} -> pnp {e1:.0f} -> polished {e2:.0f}")
        return

    out = {k: P[k] for k in ("scale", "intrinsics", "anchor", "anchor_inliers") if k in P}
    if args.sequential:
        frames = sorted(frames, key=lambda f: f["name"])
    out["frames"] = []
    nref, edges, t0, prev, nprev = 0, [], time.time(), None, 0
    for i, fr in enumerate(frames):
        real = real_rgb(fr["name"])
        cands = [("global", np.array(fr["w2c"]))]
        if args.sequential and prev is not None:
            cands = [("prev", prev)] + cands
        best = None
        for tag, init in cands:
            w2c, ninl, edge = refine(init, real)
            if best is None or edge > best[2]:
                best = (w2c, ninl, edge, tag)
            if edge >= 85:
                break
        w2c, ninl, edge, tag = best
        used = ninl >= args.min_inl
        if args.sequential and edge >= 60:
            prev = w2c
            nprev += (tag == "prev")
        out["frames"].append(dict(name=fr["name"], w2c=w2c.tolist(),
                                  inl=int(ninl), edge=round(edge, 1), refined=bool(used)))
        if used:
            nref += 1
            edges.append(edge)
        if i % 40 == 0:
            print(f"[refine] frame {i}/{len(frames)} inl {ninl} edge {edge:.0f}% via {tag} ({time.time()-t0:.0f}s)", flush=True)
    if args.sequential:
        print(f"[refine] sequential: {nprev}/{len(frames)} frames locked from the previous pose")
    json.dump(out, open(args.out, "w"))
    print(f"[refine] refined {nref}/{len(frames)}; median edge-match "
          f"{int(np.median(edges)) if edges else 0}%")


if __name__ == "__main__":
    main()

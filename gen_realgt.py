"""
Build the real-GT set: at each refined pose, render the splat's metric depth with the
same fisheye model used for localization (so depth and image agree to the periphery) and
pair it with the real photo. Only frames that passed refinement (--min-inl) are kept.
Writes a manifest and a QC overlay for the best frame.

Usage:
    python gen_realgt.py --scan <lcc-result> --refined realgt_poses_refined.json \
        --frames <img_dir> --out <real_gt_dir> [--min-inl 30]
"""
import argparse
import json
import os
import numpy as np
import cv2
from PIL import Image

from lcc import LccScan
from render import Renderer, Intrinsics

RW, RH = 640, 480


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--refined", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-inl", type=int, default=30,
                    help="keep only frames refined with at least this many PnP inliers")
    ap.add_argument("--min-edge", type=float, default=0.0,
                    help="keep only frames whose render/photo center edge-match %% >= this "
                         "(the direct alignment gate; needs 'edge' in the refined poses)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    P = json.load(open(args.refined))
    sc = RW / P["intrinsics"]["w"]
    intr = Intrinsics.from_dict(P["intrinsics"]).scaled(sc)
    intr.model = "fisheye"

    scan = LccScan.open(args.scan)
    splats = scan.decode(lod=0)
    splats.quat[:] = 0.0
    splats.quat[:, 0] = 1.0
    renderer = Renderer(splats)

    frames = [f for f in P["frames"] if f.get("refined") and f.get("inl", 0) >= args.min_inl
              and f.get("edge", 100) >= args.min_edge]
    print(f"[realgt] {len(frames)}/{len(P['frames'])} frames pass min-inl {args.min_inl} "
          f"& min-edge {args.min_edge}%")

    def real_img(name):
        return np.array(Image.open(f"{args.frames}/{name}").convert("RGB").resize((RW, RH)))

    manifest = []
    for i, fr in enumerate(frames):
        rgb, depth = renderer.render(np.array(fr["w2c"]), intr, depth=True)
        stem = fr["name"].replace(".jpg", "")
        np.save(f"{args.out}/{stem}_depth.npy", depth.astype(np.float32))
        Image.fromarray(real_img(fr["name"])).save(f"{args.out}/{stem}_rgb.jpg", quality=92)
        m = (depth > 0.1) & (depth < 20)
        manifest.append(dict(name=fr["name"], inl=fr["inl"],
                             valid=float(m.mean()), median_depth=float(np.median(depth[m])) if m.any() else 0))
        if i % 50 == 0:
            print(f"[realgt] {i}/{len(frames)} {fr['name']}", flush=True)

    json.dump(dict(intrinsics=P["intrinsics"], frames=manifest), open(f"{args.out}/manifest.json", "w"), indent=2)
    print(f"[realgt] wrote {len(manifest)} (real-RGB, metric-depth) pairs to {args.out}")

    best = max(frames, key=lambda f: f["inl"])
    rgb, depth = renderer.render(np.array(best["w2c"]), intr, depth=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    m = depth > 0
    lo, hi = np.percentile(depth[m], 2), np.percentile(depth[m], 98)
    dn = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    dcol = (cm.turbo(dn)[..., :3] * 255).astype(np.uint8)
    real = real_img(best["name"])
    edges = cv2.Canny(cv2.cvtColor(real, cv2.COLOR_RGB2GRAY), 80, 160)
    ov = dcol.copy()
    ov[edges > 0] = (255, 255, 255)
    Image.fromarray(np.concatenate([real, dcol, ov], 1)).save(f"{args.out}/overlay_best.jpg", quality=92)
    print(f"[realgt] QC overlay: {args.out}/overlay_best.jpg (frame {best['name']}, inl {best['inl']})")


if __name__ == "__main__":
    main()

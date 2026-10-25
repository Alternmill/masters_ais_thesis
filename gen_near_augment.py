"""
Build a real-GT training set with optional near-pose render augmentation.

For each refined real frame: copy the real (RGB, depth) pair, then optionally add N
near-pose renders (<=radius m, <=rot deg away) to regularise a small real set. --naug 0
gives the real-only set with the same split, for a fair comparison.

Usage:
    python gen_near_augment.py --scan <lcc-result> --refined realgt_poses_refined.json \
        --realgt <real_gt_dir> --out <dataset_dir> --naug 4 --radius 0.10 --rot 8
"""
import argparse
import math
import os
import numpy as np
from PIL import Image

from lcc import LccScan
from render import Renderer, Intrinsics

RW, RH = 640, 480


def rand_rot(rng, max_deg):
    ax = rng.standard_normal(3)
    ax /= np.linalg.norm(ax)
    th = math.radians(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--refined", required=True)
    ap.add_argument("--realgt", required=True, help="dir with <stem>_rgb.jpg/_depth.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--naug", type=int, default=4, help="near-pose renders per train frame (0 = real only)")
    ap.add_argument("--radius", type=float, default=0.10, help="max translation jitter (m)")
    ap.add_argument("--rot", type=float, default=8.0, help="max rotation jitter (deg)")
    ap.add_argument("--min-inl", type=int, default=20)
    ap.add_argument("--val", type=float, nargs=2, default=[0.4, 0.6], help="held-out middle slice")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for s in ("train", "val"):
        os.makedirs(f"{args.out}/{s}", exist_ok=True)

    import json
    P = json.load(open(args.refined))
    intr = Intrinsics.from_dict(P["intrinsics"]).scaled(RW / P["intrinsics"]["w"])
    intr.model = "fisheye"
    ref = sorted([f for f in P["frames"] if f.get("refined") and f.get("inl", 0) >= args.min_inl],
                 key=lambda f: f["name"])
    n = len(ref)
    vlo, vhi = int(args.val[0] * n), int(args.val[1] * n)
    split = {f["name"]: ("val" if vlo <= i < vhi else "train") for i, f in enumerate(ref)}
    print(f"[aug] {n} refined frames -> {sum(v=='train' for v in split.values())} train / "
          f"{sum(v=='val' for v in split.values())} val; naug={args.naug} r={args.radius} rot={args.rot}")

    need_render = args.naug > 0
    renderer = None
    if need_render:
        scan = LccScan.open(args.scan)
        sp = scan.decode(lod=0)
        sp.quat[:] = 0.0
        sp.quat[:, 0] = 1.0
        renderer = Renderer(sp)

    rng = np.random.default_rng(args.seed)
    counts = {"train": 0, "val": 0}

    def save(sub, rgb, depth):
        i = counts[sub]
        Image.fromarray(rgb).save(f"{args.out}/{sub}/rgb_{i:04d}.jpg", quality=92)
        np.save(f"{args.out}/{sub}/depth_{i:04d}.npy", depth.astype(np.float32))
        counts[sub] += 1

    for f in ref:
        sub = split[f["name"]]
        stem = f["name"].replace(".jpg", "")
        rp = f"{args.realgt}/{stem}_rgb.jpg"
        dp = f"{args.realgt}/{stem}_depth.npy"
        rgb = np.array(Image.open(rp).convert("RGB").resize((RW, RH)))
        save(sub, rgb, np.load(dp))
        if need_render and sub == "train":
            c2w = np.linalg.inv(np.array(f["w2c"]))
            for _ in range(args.naug):
                c2w2 = c2w.copy()
                c2w2[:3, 3] = c2w[:3, 3] + np.clip(rng.normal(0, args.radius * 0.6, 3), -args.radius, args.radius)
                c2w2[:3, :3] = c2w[:3, :3] @ rand_rot(rng, args.rot)
                r_rgb, r_d = renderer.render(np.linalg.inv(c2w2), intr, depth=True)
                save("train", r_rgb, r_d)

    print(f"[aug] wrote train={counts['train']} val={counts['val']} to {args.out}")


if __name__ == "__main__":
    main()

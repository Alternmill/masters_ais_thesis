"""
Render an (RGB, metric-depth) training set from a .lcc scan's capture trajectory.

Held-out val is a contiguous slice of the trajectory, so train/val are spatially disjoint.

Usage:
    python gen_dataset.py --scan scans/office/lcc-result --out out/run_office
    python gen_dataset.py --scan <bathroom_scan> --out <dir> --jitter 2 --val 0.72 0.92
"""
import argparse
import os
import numpy as np
from PIL import Image

from lcc import LccScan, forward_of_pose, gravity_upright_c2w
from render import Renderer, Intrinsics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True, help="path to the lcc-result dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wh", type=int, nargs=2, default=[640, 480])
    ap.add_argument("--jitter", type=int, default=2, help="near-pose augmentations per train frame")
    ap.add_argument("--val", type=float, nargs=2, default=[0.72, 0.92],
                    help="contiguous held-out trajectory fraction [lo, hi]")
    ap.add_argument("--crop", type=float, nargs=3, default=None,
                    help="fixed half-extent box (around pose centroid) to keep splats; "
                         "default adapts to the trajectory extent (see --margin)")
    ap.add_argument("--margin", type=float, default=4.0,
                    help="when --crop is unset, keep splats within the pose bounding box "
                         "expanded by this many metres (robust across room- and corridor-scale scans)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(f"{args.out}/train", exist_ok=True)
    os.makedirs(f"{args.out}/val", exist_ok=True)

    scan = LccScan.open(args.scan)
    splats = scan.decode(lod=0)
    poses = scan.load_poses()
    centers = scan.pose_centers()
    n0 = len(splats)
    if args.crop is not None:
        centroid = centers.mean(0)
        splats = splats.crop_box(centroid, args.crop)
        desc = f"fixed box {args.crop} around {centroid.round(2)}"
    else:
        center = (centers.min(0) + centers.max(0)) / 2
        half = (centers.max(0) - centers.min(0)) / 2 + args.margin
        splats = splats.crop_box(center, half)
        desc = f"traj bbox + {args.margin}m (half-extent {half.round(1).tolist()})"
    print(f"[gen] {len(splats)}/{n0} splats kept ({desc}), {len(poses)} poses")

    splats.quat[:] = 0.0
    splats.quat[:, 0] = 1.0

    W, H = args.wh
    intr = Intrinsics.pinhole(W, H)
    renderer = Renderer(splats)
    rng = np.random.default_rng(args.seed)

    def save(prefix, i, fwd, t):
        c2w = gravity_upright_c2w(fwd, t)
        rgb, d = renderer.render(np.linalg.inv(c2w), intr, depth=True)
        Image.fromarray(rgb).save(f"{args.out}/{prefix}/rgb_{i:04d}.jpg", quality=92)
        np.save(f"{args.out}/{prefix}/depth_{i:04d}.npy", d)

    n = len(poses)
    val_lo, val_hi = int(args.val[0] * n), int(args.val[1] * n)
    ti = vi = 0
    for i, p in enumerate(poses):
        t = np.array(p["T"], float)
        fwd = forward_of_pose(p)
        if val_lo <= i < val_hi:
            save("val", vi, fwd, t); vi += 1
        else:
            save("train", ti, fwd, t); ti += 1
            for _ in range(args.jitter):
                save("train", ti, fwd + rng.normal(0, 0.05, 3), t + rng.normal(0, 0.15, 3))
                ti += 1
        if i % 150 == 0:
            print(f"[gen] pose {i}/{n}  train {ti}  val {vi}", flush=True)

    print(f"[gen] DONE train={ti} val={vi} (val poses {val_lo}-{val_hi} held out)")


if __name__ == "__main__":
    main()

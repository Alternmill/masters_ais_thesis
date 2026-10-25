#!/usr/bin/env python
"""Export an LCC splat's centers as a colored PLY point cloud.

binary_little_endian, float32 x/y/z (metres) + uchar red/green/blue: the widely
supported fast-path PLY layout. One point per splat mean (or Gaussian-sampled).

  python export_ply.py --scan scans/bathroom_2/lcc-result --out exports/bathroom_2.ply [--lod 0]
"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcc import LccScan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="scans/bathroom_2/lcc-result")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lod", type=int, default=0)
    ap.add_argument("--sample", action="store_true",
                    help="sample points across each Gaussian (axis-aligned, std=scales) "
                         "instead of emitting bare centers")
    ap.add_argument("--spacing", type=float, default=0.01,
                    help="[--sample] target point spacing (m): splats get ~area/spacing^2 samples")
    ap.add_argument("--cap", type=int, default=32, help="[--sample] max samples per splat")
    ap.add_argument("--sigma-clip", type=float, default=2.0,
                    help="[--sample] truncate Gaussian samples beyond this many sigma per axis")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sc = LccScan.open(a.scan)
    sp = sc.decode(lod=a.lod)

    if a.sample:
        rng = np.random.default_rng(a.seed)
        ss = np.sort(sp.scales, axis=1)
        area = ss[:, 1] * ss[:, 2]
        counts = np.clip(np.round(area / (a.spacing ** 2)), 1, a.cap).astype(np.int64)
        idx = np.repeat(np.arange(len(sp)), counts)
        noise = rng.standard_normal((idx.size, 3))
        noise = np.clip(noise, -a.sigma_clip, a.sigma_clip)
        first = np.concatenate(([0], np.cumsum(counts)[:-1]))
        noise[first] = 0.0
        pts = sp.means[idx] + noise * sp.scales[idx]
        xyz = pts.astype("<f4")
        rgb = np.clip(sp.color[idx] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        xyz = sp.means.astype("<f4")
        rgb = np.clip(sp.color * 255.0 + 0.5, 0, 255).astype(np.uint8)
    n = len(xyz)

    rec = np.empty(n, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                      ("red", "u1"), ("green", "u1"), ("blue", "u1")]))
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment exported from LCC splat %s (lod %d) by export_ply.py\n"
        "element vertex %d\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n" % (a.scan, a.lod, n)
    ).encode("ascii")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())
    mb = os.path.getsize(a.out) / 1e6
    lo, hi = xyz.min(0), xyz.max(0)
    print(f"[ply] wrote {n} points -> {a.out} ({mb:.1f} MB)")
    print(f"[ply] bbox (m) min {lo.round(2)} max {hi.round(2)}  extent {(hi-lo).round(2)}")


if __name__ == "__main__":
    main()

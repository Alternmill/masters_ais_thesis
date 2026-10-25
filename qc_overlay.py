"""
Localization QC: render the splat at each refined pose over the real frame with a Canny
edge overlay, tile worst/median/best into a montage, and print the center/periphery
edge-match stats used to accept frames. Run right after refinement, before finetuning.

Usage:
    python qc_overlay.py --scan <lcc-result> --refined realgt_poses_fish_refined.json \
        --frames <real_frames/img> --out qc_overlay.jpg [--n 12]
"""
import argparse
import json
import os
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import distance_transform_edt

from lcc import LccScan
from render import Renderer, Intrinsics

SC = 0.5


def edge_match(real, rend, w, h):
    """Return (center%, periphery%): fraction of real edges within 3px of a render edge."""
    re = cv2.Canny(cv2.cvtColor(real, cv2.COLOR_RGB2GRAY), 80, 160)
    ce = cv2.Canny(cv2.cvtColor(rend, cv2.COLOR_RGB2GRAY), 80, 160)
    dt = distance_transform_edt(~(ce > 0))
    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    center = (re > 0) & (rad < 0.5)
    periph = (re > 0) & (rad > 0.8)
    c = 100 * np.mean(dt[center] < 3) if center.any() else 0.0
    p = 100 * np.mean(dt[periph] < 3) if periph.any() else 0.0
    return c, p


def overlay(real, rend):
    """Real image with render edges drawn in green."""
    ce = cv2.Canny(cv2.cvtColor(rend, cv2.COLOR_RGB2GRAY), 80, 160)
    out = real.copy()
    out[ce > 0] = (0, 255, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--refined", required=True, help="realgt_poses_*_refined.json")
    ap.add_argument("--frames", required=True, help="dir of real frame jpgs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12, help="frames in the montage")
    args = ap.parse_args()

    P = json.load(open(args.refined))
    intr = Intrinsics.from_dict(P["intrinsics"]).scaled(SC)
    frames = P["frames"]

    scan = LccScan.open(args.scan)
    splats = scan.decode(lod=0)
    splats.quat[:] = 0.0
    splats.quat[:, 0] = 1.0
    renderer = Renderer(splats)

    stats = []
    for fr in frames:
        path = os.path.join(args.frames, fr["name"])
        if not os.path.exists(path):
            continue
        real = np.array(Image.open(path).convert("RGB").resize((intr.w, intr.h)))
        rend, _ = renderer.render(np.array(fr["w2c"]), intr, depth=False)
        c, p = edge_match(real, rend, intr.w, intr.h)
        stats.append(dict(name=fr["name"], center=c, periph=p, inl=fr.get("inl", 0),
                          refined=fr.get("refined", False), real=real, rend=rend))

    if not stats:
        print("[qc] no frames matched — check --frames path"); return

    c_all = np.array([s["center"] for s in stats])
    p_all = np.array([s["periph"] for s in stats])
    nref = sum(s["refined"] for s in stats)
    print(f"[qc] {len(stats)} frames | refined {nref} | "
          f"center edge-match {c_all.mean():.0f}% (min {c_all.min():.0f}) | "
          f"periphery {p_all.mean():.0f}% (min {p_all.min():.0f})")

    order = np.argsort(c_all)
    pick = sorted(set(np.linspace(0, len(order) - 1, args.n).round().astype(int)))
    tiles = []
    for i in pick:
        s = stats[order[i]]
        tile = overlay(s["real"], s["rend"])
        label = f"{s['name']} c{s['center']:.0f} p{s['periph']:.0f} i{s['inl']}"
        cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 0) if s["refined"] else (0, 0, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    h, w = intr.h, intr.w
    grid = np.zeros((rows * h, cols * w, 3), np.uint8)
    for idx, t in enumerate(tiles):
        r, c = divmod(idx, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    Image.fromarray(grid).save(args.out)
    print(f"[qc] wrote {args.out}  (green=refined, red label=not refined; edges should sit on real geometry)")


if __name__ == "__main__":
    main()

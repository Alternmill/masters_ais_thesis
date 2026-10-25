#!/usr/bin/env python
"""Render a smooth fly-through video along a scan's capture trajectory.

Interpolates and stabilizes the capture path to a high-framerate tour.

  python render_path.py --scan scans/bathroom_2/lcc-result --out out/bath2.mp4 \
      --fps 60 --duration 60 --wh 1280 960
"""
import argparse, os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcc import LccScan, forward_of_pose, gravity_upright_c2w
from render import Renderer, Intrinsics


def smooth(a, win):
    if win <= 1: return a
    k = np.ones(win) / win
    pad = win // 2
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(ap[:, c], k, "valid")[:a.shape[0]] for c in range(a.shape[1])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="scans/bathroom_2/lcc-result")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--wh", type=int, nargs=2, default=[1280, 960])
    ap.add_argument("--fov-scale", type=float, default=0.6, help="fx = fov-scale * width (lower = wider)")
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--smooth-win", type=int, default=15)
    ap.add_argument("--lod", type=int, default=0)
    a = ap.parse_args()

    scan = LccScan.open(a.scan)
    splats = scan.decode(lod=a.lod)
    splats.quat[:] = 0.0; splats.quat[:, 0] = 1.0
    poses = scan.load_poses()
    centers = scan.pose_centers().astype(np.float32)

    lo, hi = centers.min(0) - a.margin, centers.max(0) + a.margin
    m = splats.means
    keep = np.all((m >= lo) & (m <= hi), axis=1)
    splats = splats.crop_box((lo + hi) / 2, (hi - lo) / 2)
    print(f"[path] {len(splats)}/{keep.size} splats kept; {len(poses)} capture poses @10fps")

    fwds = np.array([forward_of_pose(p) for p in poses], np.float32)
    fwds /= np.linalg.norm(fwds, axis=1, keepdims=True)
    pos_s = smooth(centers, a.smooth_win)
    fwd_s = smooth(fwds, a.smooth_win); fwd_s /= np.linalg.norm(fwd_s, axis=1, keepdims=True)

    N = int(a.fps * a.duration)
    src = np.linspace(0, len(poses) - 1, N)
    i0 = np.floor(src).astype(int); i1 = np.minimum(i0 + 1, len(poses) - 1); fr = (src - i0)[:, None]
    pos_i = pos_s[i0] * (1 - fr) + pos_s[i1] * fr
    fwd_i = fwd_s[i0] * (1 - fr) + fwd_s[i1] * fr
    fwd_i /= np.linalg.norm(fwd_i, axis=1, keepdims=True)

    W, H = a.wh
    f = a.fov_scale * W
    intr = Intrinsics(W, H, f, f, W / 2, H / 2, (), "pinhole")
    renderer = Renderer(splats)

    tmp = a.out + "_frames"
    os.makedirs(tmp, exist_ok=True)
    for j in range(N):
        c2w = gravity_upright_c2w(fwd_i[j], pos_i[j])
        rgb, _ = renderer.render(np.linalg.inv(c2w), intr, depth=False)
        from PIL import Image
        Image.fromarray(rgb).save(f"{tmp}/{j:05d}.jpg", quality=92)
        if j % 100 == 0: print(f"[path] {j}/{N}", flush=True)

    print("[path] encoding...")
    subprocess.run(["ffmpeg", "-y", "-framerate", str(a.fps), "-i", f"{tmp}/%05d.jpg",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", a.out], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["rm", "-rf", tmp])
    print(f"[path] DONE -> {a.out}  ({N} frames @ {a.fps}fps = {a.duration:.0f}s)")


if __name__ == "__main__":
    main()

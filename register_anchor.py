"""
Fix the global orientation of the COLMAP-to-splat registration using one anchor frame.

Trajectory-centre ICP (export_traj) recovers scale + position but leaves a global
rotation ambiguity. One texture-rich anchor frame's orientation is found by a
gravity-de-rolled yaw/pitch search (SIFT-scored against the real photo), giving the
global rotation that aligns every frame.

Usage:
    python register_anchor.py --scan <lcc-result> --traj traj_data.json \
        --frames <img_dir> --out realgt_poses.json
"""
import argparse
import json
import numpy as np
import cv2
from PIL import Image

from lcc import LccScan, c2w_from_yaw_pitch
from render import Renderer, Intrinsics

SC = 0.5


def sift_inliers(sift, bf, render_gray, kp_r, des_r):
    k1, d1 = sift.detectAndCompute(render_gray, None)
    if d1 is None or len(k1) < 10:
        return 0
    m = bf.knnMatch(d1, des_r, k=2)
    good = [x for x, y in m if x.distance < 0.75 * y.distance]
    if len(good) < 8:
        return len(good)
    src = np.float32([k1[x.queryIdx].pt for x in good])
    dst = np.float32([kp_r[x.trainIdx].pt for x in good])
    Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--traj", required=True, help="traj_data.json from export_traj")
    ap.add_argument("--frames", required=True, help="dir of real frame jpgs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors", type=int, default=12, help="candidate anchor frames to try")
    args = ap.parse_args()

    td = json.load(open(args.traj))
    s = td["scale"]
    frames = td["frames"]
    intr = Intrinsics.from_dict(td["intrinsics"]).scaled(SC)

    scan = LccScan.open(args.scan)
    splats = scan.decode(lod=0)
    splats.quat[:] = 0.0
    splats.quat[:, 0] = 1.0
    renderer = Renderer(splats)

    def render_gray(c2w):
        rgb, _ = renderer.render(np.linalg.inv(c2w), intr, depth=False)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create(nfeatures=1500)
    bf = cv2.BFMatcher()

    def real_gray(name):
        im = Image.open(f"{args.frames}/{name}").convert("RGB").resize((intr.w, intr.h))
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)

    cand_idx = np.linspace(0, len(frames) - 1, args.anchors).round().astype(int)

    def search_orientation(fr):
        real = real_gray(fr["name"])
        kp_r, des_r = sift.detectAndCompute(real, None)
        if des_r is None:
            return (-1, 0.0, 0.0)
        pos = np.array(fr["c_spl"])
        bst = (-1, 0.0, 0.0)
        for yaw in np.radians(np.arange(0, 360, 15)):
            for pitch in np.radians(np.arange(-20, 21, 10)):
                n = sift_inliers(sift, bf, render_gray(c2w_from_yaw_pitch(pos, yaw, pitch)), kp_r, des_r)
                if n > bst[0]:
                    bst = (n, yaw, pitch)
        for yaw in bst[1] + np.radians(np.arange(-12, 13, 3)):
            for pitch in bst[2] + np.radians(np.arange(-8, 9, 2)):
                n = sift_inliers(sift, bf, render_gray(c2w_from_yaw_pitch(pos, yaw, pitch)), kp_r, des_r)
                if n > bst[0]:
                    bst = (n, yaw, pitch)
        return bst

    best = None
    for i in cand_idx:
        fr = frames[int(i)]
        n, yaw, pitch = search_orientation(fr)
        print(f"[anchor] cand {fr['name']}: orientation inliers {n}", flush=True)
        if best is None or n > best[0]:
            best = (n, yaw, pitch, fr)
    inl_a, yaw_a, pitch_a, anchor = best
    print(f"[anchor] chose {anchor['name']} inliers {inl_a} yaw {np.degrees(yaw_a):.0f} pitch {np.degrees(pitch_a):.0f}")
    if inl_a < 20:
        print("[anchor] WARNING: weak best anchor (<20 inliers) — global rotation may be off. "
              "Increase --anchors or check the scan has textured views.")

    pos = np.array(anchor["c_spl"])
    R_true = np.linalg.inv(c2w_from_yaw_pitch(pos, yaw_a, pitch_a))[:3, :3]
    R_col = np.array(anchor["R_col"])
    c_col = np.array(anchor["c_col"])
    Rg = R_true.T @ R_col
    t = pos - s * (Rg @ c_col)
    bst = (inl_a,)

    out = {"scale": s, "intrinsics": td["intrinsics"], "anchor": anchor["name"],
           "anchor_inliers": int(bst[0]), "frames": []}
    for fr in frames:
        cc = np.array(fr["c_col"])
        Rc = np.array(fr["R_col"])
        c_spl = s * (Rg @ cc) + t
        R_ws = Rc @ Rg.T
        w2c = np.eye(4)
        w2c[:3, :3] = R_ws
        w2c[:3, 3] = -R_ws @ c_spl
        out["frames"].append(dict(name=fr["name"], w2c=w2c.tolist()))
    json.dump(out, open(args.out, "w"))
    print(f"[anchor] wrote {len(out['frames'])} poses to {args.out}")


if __name__ == "__main__":
    main()

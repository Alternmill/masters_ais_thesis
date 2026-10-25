"""Changed-region masks between scans of one room in different states (stall doors A/B, person P).

For test state Y and other state X: render splat X, aligned into Y's frame, at every Y test
pose with the same camera model as the Y real-GT, and mark changed = |dY - dX| > max(0.10, 0.05 dY)
where both are valid, dilated by 3 px. Also reports the static-structure residual as an alignment check.
The scan-to-scan transforms align/T_b2_to_b1.json and align/T_b3_to_b1.json come from align_scans.py.

Usage:
    python stall_masks.py [--sample N]   -> align/masks_<X>to<Y>.npz, align/residuals.json
"""
import argparse, glob, json, os
import numpy as np, cv2
from lcc import LccScan
from render import Renderer, Intrinsics
B = "scans"; H = os.path.dirname(os.path.abspath(__file__))
ST = {"A": ("bathroom_1", "q2_bath1/poses_refined.json", "aug19_bath1"), "B": ("bathroom_2", "q2_bath2/poses_refined_d.json", "aug19_bath2"),
      "P": ("bathroom_3", "q2_bath3/poses_refined.json", "aug19_bath3")}
T21 = np.array(json.load(open(f"{H}/align/T_b2_to_b1.json"))["T_src_to_ref"]); T31 = np.array(json.load(open(f"{H}/align/T_b3_to_b1.json"))["T_src_to_ref"])
TO_A = {"A": np.eye(4), "B": T21, "P": T31}
def T_XtoY(X, Y): return np.linalg.inv(TO_A[Y]) @ TO_A[X]
ap = argparse.ArgumentParser(); ap.add_argument("--sample", type=int, default=0); ap.add_argument("--pairs", default="")
a = ap.parse_args()
rend = {}
def renderer(X):
    if X not in rend:
        sp = LccScan.open(f"{B}/{ST[X][0]}/lcc-result").decode(lod=0); sp.quat[:] = 0.0; sp.quat[:, 0] = 1.0
        rend.clear(); rend[X] = Renderer(sp)
    return rend[X]
res = json.load(open(f"{H}/align/residuals.json")) if os.path.exists(f"{H}/align/residuals.json") else {}
pairs = [p.split(">") for p in a.pairs.split(",")] if a.pairs else [(X, Y) for X in "ABP" for Y in "ABP" if X != Y]
if a.sample: pairs = [(Y, Y) for Y in "ABP"] + pairs
pairs = sorted(pairs, key=lambda p: p[0])
for X, Y in pairs:
    Pz = json.load(open(f"{B}/{ST[Y][1]}"))
    intr = Intrinsics.from_dict(Pz["intrinsics"]).scaled(640 / Pz["intrinsics"]["w"]); intr.model = "fisheye"
    w2c = {f["name"][:-4]: np.array(f["w2c"]) for f in Pz["frames"]}
    paths = sorted(glob.glob(f"{B}/evalsets_2026-09-27/{ST[Y][2]}/*_rgb.jpg"))
    if a.sample: paths = paths[:: max(1, len(paths) // a.sample)]
    names = [os.path.basename(p)[:-8] for p in paths]
    R = renderer(X); T = T_XtoY(X, Y); M, stat = [], []
    for n, p in zip(names, paths):
        gY = np.load(p.replace("_rgb.jpg", "_depth.npy"))
        _, dX = R.render(w2c[n] @ T, intr, depth=True)
        v = (gY > 0.1) & (dX > 0.1) & (gY < 15) & (dX < 15)
        diff = dX - gY
        ch = v & (np.abs(diff) > np.maximum(0.10, 0.05 * gY))
        ch = cv2.dilate(ch.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        st = v & ~ch
        stat.append((float(np.median(diff[st])) if st.any() else np.nan, float(np.median(np.abs(diff[st]))) if st.any() else np.nan,
                     float((np.abs(diff[st]) < 0.02).mean()) if st.any() else np.nan, float(ch.mean()), float(v.mean())))
        M.append(ch)
    s = np.array(stat)
    res[f"{X}>{Y}"] = dict(n=len(names), static_median_signed_cm=float(np.nanmedian(s[:, 0]) * 100),
                           static_median_abs_cm=float(np.nanmedian(s[:, 1]) * 100), static_within_2cm=float(np.nanmean(s[:, 2])),
                           changed_frac=float(np.mean(s[:, 3])), valid_frac=float(np.mean(s[:, 4])))
    print(X, ">", Y, res[f"{X}>{Y}"], flush=True)
    if not a.sample:
        np.savez_compressed(f"{H}/align/masks_{X}to{Y}.npz", names=np.array(names), m=np.stack(M))
    json.dump(res, open(f"{H}/align/residuals{'_sample' if a.sample else ''}.json", "w"), indent=1)

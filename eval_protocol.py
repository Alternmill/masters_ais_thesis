"""Evaluation protocol: one checkpoint (or zero-shot) on all test sets.

Per set: frame-mean delta1 / AbsRel / MAE for each prediction scale, 95 % frame-bootstrap
CI of delta1, per-band metrics, and metrics on masked dynamic-object pixels.

Usage:
    python eval_protocol.py --ckpt best.pt --out eval.json [--scales 1.0]   (no --ckpt = zero-shot)
    python eval_protocol.py --sstar DIR [DIR...] --out sstar.json           (site-calibration scalar s*)
"""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from bootstrap import frame_stats, stat
B = "scans"
SETS = {"office_aug18": f"{B}/q2_office_185719/real_gt",
        "office_aug19": f"{B}/evalsets_2026-09-27/office_morning_all",
        "bath_aug27": f"{B}/q2_restroom2/real_gt_d1200ts",
        "bath_aug19_A": f"{B}/evalsets_2026-09-27/aug19_bath1",
        "bath_aug19_B": f"{B}/evalsets_2026-09-27/aug19_bath2",
        "bath_aug19_person": f"{B}/evalsets_2026-09-27/aug19_bath3",
        "lviv": f"{B}/q2_lviv_open/real_gt_d1500ts",
        "second_floor": f"{B}/q2_second_floor/real_gt_d1200ts"}
BANDS = [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 15)]
HERE = os.path.dirname(os.path.abspath(__file__))


def model(ckpt):
    from depth_anything_3.api import DepthAnything3
    m = DepthAnything3.from_pretrained("depth-anything/da3metric-large").to("cuda")
    if ckpt:
        sd = torch.load(ckpt, map_location="cuda"); miss = m.model.load_state_dict(sd, strict=False)
        print("loaded", ckpt, "unexpected keys:", len(miss.unexpected_keys))
    return m.model.eval(), m


def predict(net, api, paths):
    out = []
    for p in paths:
        im, _, _ = api._preprocess_inputs([np.array(Image.open(p).convert("RGB").resize((640, 480)))])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = net(im.unsqueeze(0).cuda(), None, None, [], False, False, "saddle_balanced")["depth"][0, 0].float()
        out.append(F.interpolate(o[None, None], size=(480, 640), mode="bilinear", align_corners=False)[0, 0].cpu().numpy())
    return out


def frame_metrics(p, g, m):
    a, b = p[m], g[m]
    return (float(np.mean(np.maximum(a / b, b / a) < 1.25)), float(np.mean(np.abs(a - b) / b)), float(np.mean(np.abs(a - b)) * 100))


def summarize(P, G, s, lo=0.0, hi=15.0, masks=None, minpx=50):
    rows = []
    for i, (p, g) in enumerate(zip(P, G)):
        m = (g >= max(lo, 1e-3)) & (g < hi) & (g > 0)
        if masks is not None: m &= masks[i]
        if m.sum() < minpx: continue
        rows.append(frame_metrics(p * s, g, m))
    if not rows: return None
    r = np.array(rows)
    return dict(d1=float(r[:, 0].mean()), absrel=float(r[:, 1].mean()), mae_cm=float(r[:, 2].mean()), n=len(rows)), r


def cart_masks(d):
    cache = f"{HERE}/_cart_masks.npz"
    if os.path.exists(cache): return dict(np.load(cache))
    import cv2
    fr = json.load(open(f"{B}/q2_second_floor/cart_frames.json")); M = {}
    for name, *_ in fr:
        im = cv2.imread(f"{d}/{name}_rgb.jpg"); im = cv2.resize(im, (640, 480))
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        m = (((hsv[..., 0] <= 15) | (hsv[..., 0] >= 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 70)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n > 1: m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        M[name] = cv2.erode(m, np.ones((11, 11), np.uint8)).astype(bool)
    np.savez_compressed(cache, **M); return M


def evaluate(net, api, scales, nboot=1000):
    res = {}; rng = np.random.default_rng(0)
    for key, d in SETS.items():
        paths = sorted(glob.glob(f"{d}/*_rgb.jpg"))
        G = [np.load(p.replace("_rgb.jpg", "_depth.npy")).astype(np.float32) for p in paths]
        P = predict(net, api, paths)
        fr = frame_stats(P, G, 0, 15); boots = [rng.integers(0, len(P), len(P)) for _ in range(nboot)]
        r = {"n_frames": len(P)}
        for s in [1.0] + scales:
            tag = "raw" if s == 1.0 else f"x{s:.4f}"
            full, _ = summarize(P, G, s)
            v = "raw" if s == 1.0 else f"fixed={s}"
            bs = np.array([stat(fr, b, v)[0] for b in boots])
            full["d1_ci"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
            full["bands"] = {f"{lo}-{hi}": (summarize(P, G, s, lo, hi) or [None])[0] for lo, hi in BANDS}
            full["near_0_1"] = (summarize(P, G, s, 0, 1) or [None])[0]
            r[tag] = full
        if key == "bath_aug27" and os.path.exists(f"{d}/_person_masks.npz"):
            Mp = np.load(f"{d}/_person_masks.npz")["m"]
            use = [i for i in range(len(P)) if (Mp[i] & (G[i] > 0.2)).mean() >= 0.005]
            r["person"] = {("raw" if s == 1.0 else f"x{s:.4f}"): summarize([P[i] for i in use], [G[i] for i in use], s,
                           0.2, 15, masks=[Mp[i] for i in use])[0] for s in [1.0] + scales}
        if key == "second_floor":
            CM = cart_masks(d); names = [os.path.basename(p)[:-8] for p in paths]; use = [i for i, n in enumerate(names) if n in CM]
            r["cart"] = {("raw" if s == 1.0 else f"x{s:.4f}"): summarize([P[i] for i in use], [G[i] for i in use], s,
                         0.05, 15, masks=[CM[names[i]] for i in use], minpx=300)[0] for s in [1.0] + scales}
        res[key] = r
        print(key, "raw d1 %.3f" % r["raw"]["d1"], flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--scales", default=""); ap.add_argument("--sstar", nargs="*")
    a = ap.parse_args()
    net, api = model(a.ckpt)
    if a.sstar:
        out = {}
        for d in a.sstar:
            paths = sorted(glob.glob(f"{d}/rgb_*.jpg")) or sorted(glob.glob(f"{d}/*_rgb.jpg"))
            G = [np.load(p.replace("rgb_", "depth_").replace(".jpg", ".npy") if "rgb_" in os.path.basename(p)
                         else p.replace("_rgb.jpg", "_depth.npy")).astype(np.float32) for p in paths]
            fr = frame_stats(predict(net, api, paths), G, 0, 15)
            out[d] = dict(s_star=float(stat(fr, range(len(fr)), "sitecal")[1]), n=len(paths)); print(d, out[d], flush=True)
        json.dump(out, open(a.out, "w"), indent=1); sys.exit()
    scales = [float(x) for x in a.scales.split(",") if x]
    json.dump(evaluate(net, api, scales), open(a.out, "w"), indent=1)
    print("saved", a.out)

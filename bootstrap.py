"""Frame-bootstrap 95% confidence intervals for bandeval.py results, from its prediction cache.

Reuses the exact per-frame metric of bandeval.py (per-frame delta1 / MAE over a depth band,
then mean over frames). Resamples FRAMES with replacement; for the site-cal variant the global
scalar s* is re-estimated inside every resample (so its uncertainty is included).

    python bootstrap.py --realgt <heldout dir> --labels "DA3 zero-shot,DA3 real+aug" \
        [--lo 0 --hi 15] [--variants raw,sitecal] [--n 2000] [--out ci.json]
"""
import argparse
import glob
import json
import os

import numpy as np


def load(realgt, label):
    hits = sorted(glob.glob(f"{realgt}/_predcache/{glob.escape(label)}__*.npy"), key=os.path.getmtime)
    if not hits:
        raise FileNotFoundError(f"no cache for '{label}' in {realgt}/_predcache")
    return np.load(hits[-1]).astype(np.float32)


EDGES = np.linspace(np.log(0.2), np.log(5.0), 4001)
CENT = 0.5 * (EDGES[1:] + EDGES[:-1])


def frame_stats(P, gts, lo, hi):
    """Per frame: histogram of log(g/p) inside the band (+ raw delta1/MAE). None if < 50 px."""
    out = []
    for p, g in zip(P, gts):
        m = (g >= lo) & (g < hi)
        if m.sum() < 50:
            out.append(None); continue
        a, b = p[m], g[m]
        lr = np.clip(np.log(np.clip(b, 1e-6, None) / np.clip(a, 1e-6, None)), EDGES[0], EDGES[-1] - 1e-9)
        h = np.histogram(lr, EDGES)[0].astype(np.float64)
        cal = (b > 0.2) & (b < 15)
        hc = np.histogram(lr[cal], EDGES)[0].astype(np.float64)
        out.append(dict(h=h / h.sum(), hc=hc, d1=float(np.mean(np.maximum(a / b, b / a) < 1.25)),
                        mae=float(np.mean(np.abs(a - b))) * 100))
    return out


def d1_at(h, s):
    """delta1 of a frame after scaling predictions by s: |log(g/(p s))| < log 1.25."""
    return float(h[np.abs(CENT - np.log(s)) < np.log(1.25)].sum())


def stat(frames, idx, variant):
    sel = [frames[i] for i in idx if frames[i] is not None]
    if not sel:
        return np.nan, np.nan
    if variant == "raw":
        return np.mean([f["d1"] for f in sel]), np.mean([f["mae"] for f in sel])
    if variant.startswith("fixed="):
        s = float(variant.split("=")[1])
        return np.mean([d1_at(f["h"], s) for f in sel]), s
    hc = np.sum([f["hc"] for f in sel], axis=0); c = np.cumsum(hc)
    s = float(np.exp(CENT[np.searchsorted(c, c[-1] / 2)]))
    return np.mean([d1_at(f["h"], s) for f in sel]), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realgt", required=True)
    ap.add_argument("--labels", required=True, help="comma-separated cache labels")
    ap.add_argument("--lo", type=float, default=0.0); ap.add_argument("--hi", type=float, default=15.0)
    ap.add_argument("--variants", default="raw,sitecal")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pairs = sorted(glob.glob(f"{a.realgt}/*_rgb.jpg"))
    gts = [np.load(p.replace("_rgb.jpg", "_depth.npy")).astype(np.float32) for p in pairs]
    rng = np.random.default_rng(a.seed)
    boots = [rng.integers(0, len(pairs), len(pairs)) for _ in range(a.n)]
    res = {}
    for label in [l.strip() for l in a.labels.split(",") if l.strip()]:
        P = load(a.realgt, label)
        assert len(P) == len(gts), f"{label}: cache has {len(P)} frames, eval set {len(gts)}"
        fr = frame_stats(P, gts, a.lo, a.hi)
        res[label] = {}
        for v in a.variants.split(","):
            pt = stat(fr, range(len(fr)), v)
            bs = np.array([stat(fr, b, v) for b in boots])
            lo_d1, hi_d1 = np.nanpercentile(bs[:, 0], [2.5, 97.5])
            lo_m, hi_m = np.nanpercentile(bs[:, 1], [2.5, 97.5])
            second = "mae_cm" if v == "raw" else ("s_fixed" if v.startswith("fixed=") else "s_star")
            res[label][v] = {"d1": pt[0], "d1_ci": [lo_d1, hi_d1], second: pt[1], second + "_ci": [lo_m, hi_m],
                             "n_frames": sum(f is not None for f in fr)}
            print(f"{label:<28} {v:<8} delta1 {pt[0]:.3f} [{lo_d1:.3f}, {hi_d1:.3f}]   "
                  f"{second} {pt[1]:6.3f} [{lo_m:.3f}, {hi_m:.3f}]   ({res[label][v]['n_frames']} frames)")
    out = a.out or f"{a.realgt}/bootstrap_ci.json"
    json.dump(dict(realgt=a.realgt, band=[a.lo, a.hi], n_boot=a.n, results=res), open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()

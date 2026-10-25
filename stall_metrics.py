"""Room-state matrix metrics from saved predictions and change masks.

For each (method, train state X, test state Y): whole-frame, changed, static and person pixels;
delta1 (higher is better), AbsRel and MAE in cm (lower is better), per depth band, with a
frame-bootstrap 95 % CI of delta1 (1000 resamples). Diagonal X == Y uses the same-session
held-out set (every 5th frame). Person pixels come from align/person_P.npz (stall_person_masks.py).
"""
import glob, json, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); B = "scans"
SET = {"A": "aug19_bath1", "B": "aug19_bath2", "P": "aug19_bath3"}; HO = {"A": "q2_bath1", "B": "q2_bath2", "P": "q2_bath3"}
BANDS = [(0, .5), (.5, 1), (1, 2), (2, 4), (4, 15)]
C = {"da3": 0.9579, "moge": 0.9868}
rng = np.random.default_rng(0); _gt = {}
def gt(Y, names, heldout=False):
    d = f"{B}/{HO[Y]}/v2/front_gt_heldout" if heldout else f"{B}/evalsets_2026-09-27/{SET[Y]}"
    key = (d, tuple(names))
    if key not in _gt: _gt[key] = np.stack([np.load(f"{d}/{n}_depth.npy") for n in names]).astype(np.float32)
    return _gt[key]
def load(tag, k):
    f = f"{H}/preds/{tag}__{k}.npz"
    if not os.path.exists(f): return None
    z = np.load(f); return list(z["names"]), z["p"].astype(np.float32)
def frame_rows(P, G, M=None, lo=0, hi=15, minpx=50):
    R = []
    for i in range(len(P)):
        m = (G[i] > max(lo, 1e-3)) & (G[i] < hi)
        if M is not None: m &= M[i]
        if m.sum() < minpx: R.append(None); continue
        a, b = P[i][m], G[i][m]
        R.append((np.mean(np.maximum(a / b, b / a) < 1.25), np.mean(np.abs(a - b) / b), np.mean(np.abs(a - b)) * 100))
    return R
def summ(P, G, M=None):
    R = [r for r in frame_rows(P, G, M) if r is not None]
    if not R: return None
    R = np.array(R); bs = [R[rng.integers(0, len(R), len(R)), 0].mean() for _ in range(1000)]
    out = dict(d1=float(R[:, 0].mean()), absrel=float(R[:, 1].mean()), mae_cm=float(R[:, 2].mean()), n=len(R),
               d1_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    out["bands"] = {}
    for lo, hi in BANDS:
        Rb = [r for r in frame_rows(P, G, M, lo, hi) if r is not None]
        out["bands"][f"{lo}-{hi}"] = None if not Rb else dict(d1=float(np.mean([r[0] for r in Rb])), mae_cm=float(np.mean([r[2] for r in Rb])), n=len(Rb))
    return out
METHODS = [("DA3 zero-shot", "M0", 1.0, "da3"), ("DA3 × C", "M0", C["da3"], "da3"), ("DA3 LoRA", "M4", 1.0, "da3"),
           ("MoGe-2 zero-shot", "M6", 1.0, "moge"), ("MoGe-2 × C", "M6", C["moge"], "moge"), ("MoGe-2 LoRA", "M8", 1.0, "moge")]
pm = np.load(f"{H}/align/person_P.npz"); PERS = dict(zip(pm["names"], pm["m"]))
res = {}
for X in "ABP":
    for Y in "ABP":
        for name, tag, s, _ in METHODS:
            ft = tag in ("M4", "M8")
            if X == Y:
                if ft:
                    L = load(f"{tag}_{X}", "heldout")
                    if L is None: continue
                    names, P = L; G = gt(X, names, heldout=True)
                else:
                    L = load(tag, "heldout_" + X) or None
                    ho = sorted(os.path.basename(p)[:-10] for p in glob.glob(f"{B}/{HO[X]}/v2/front_gt_heldout/*_depth.npy"))
                    full = load(tag, Y)
                    if full is None: continue
                    idx = [full[0].index(n) for n in ho if n in full[0]]
                    if not idx: continue
                    names = [full[0][i] for i in idx]; P = full[1][idx]; G = gt(Y, names)
                res[f"{name}|{X}>{Y}"] = dict(whole=summ(P * s, G)); continue
            L = load(f"{tag}_{X}" if ft else tag, Y)
            if L is None: continue
            names, P = L; P = P * s; G = gt(Y, names)
            mz = np.load(f"{H}/align/masks_{X}to{Y}.npz"); MM = dict(zip(mz["names"], mz["m"]))
            CH = np.stack([MM[n] for n in names]); r = dict(whole=summ(P, G), changed=summ(P, G, CH), static=summ(P, G, ~CH))
            if Y == "P":
                PM = np.stack([PERS[n] for n in names]); use = [i for i in range(len(names)) if PM[i].mean() > 0.002]
                r["person"] = summ(P[use], G[use], PM[use]) if use else None
            res[f"{name}|{X}>{Y}"] = r
        print(X, "done", flush=True)
json.dump(res, open(f"{H}/results.json", "w"), indent=1); print("saved results.json", len(res))

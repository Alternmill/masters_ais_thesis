"""
Band-breakdown evaluator for depth models on a real-GT set.

Two phases: PREDICT caches each model's depth maps once (GPU); TABULATE computes metric
delta1 + MAE per band from the cache (CPU), so re-binning is instant.

Variants: raw (deployable output), sitecal (one dataset-wide scalar), aligned (per-frame
median scale — needs GT, diagnostic only).

Usage:
    Model roster from a finetune runs dir (DA2/DA3 x zero-shot/real/real+aug):
    python bandeval.py --realgt <dir> --ft-dir <ft_dir> \
        --bands "0-0.5,0.5-1,1-2,2-15" --variant raw
    Custom models:
    python bandeval.py --realgt <dir> --models "DA3 zs=da3:,DA3 ft=da3:/path/best.pt" --bands "..."
"""
import argparse
import glob
import hashlib
import json
import os
import numpy as np

RW, RH = 640, 480
DA2_HF = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DA3_HF = "depth-anything/da3metric-large"


def parse_bands(s):
    out = []
    for tok in s.split(","):
        lo, hi = tok.split("-")
        out.append((float(lo), float(hi)))
    return out


def default_roster(ft):
    return [("DA2 zero-shot", "da2", None), ("DA2 real", "da2", f"{ft}/da2_real_c/best.pt"),
            ("DA2 real+aug", "da2", f"{ft}/da2_realaug_c/best.pt"),
            ("DA3 zero-shot", "da3", None), ("DA3 real", "da3", f"{ft}/da3_real_c/best.pt"),
            ("DA3 real+aug", "da3", f"{ft}/da3_realaug_c/best.pt")]


def parse_models(s):
    r = []
    for spec in s.split(","):
        label, rest = spec.split("=")
        kind, ck = rest.split(":", 1)
        r.append((label.strip(), kind.strip(), ck.strip() or None))
    return r


DA3_FOCAL = {"on": False, "fx": None}


def _sig(label, ck, pairs):
    """Cache signature: eval frames + checkpoint identity (mtime+size), so retraining a
    checkpoint auto-invalidates its cache — no stale predictions."""
    h = hashlib.md5(("|".join(pairs)).encode())
    if ck and os.path.exists(ck):
        st = os.stat(ck)
        h.update(f"{ck}:{st.st_mtime_ns}:{st.st_size}".encode())
    else:
        h.update(b"zeroshot")
    if DA3_FOCAL["on"]:
        h.update(f"focal{DA3_FOCAL['fx']}".encode())
    return f"{label}__{h.hexdigest()[:12]}.npy"


def predict_all(models, pairs, cache, use_cache=True):
    """Ensure every model's predictions on `pairs` are cached (GPU). Returns {label: preds}."""
    os.makedirs(cache, exist_ok=True)
    cfile = {m[0]: f"{cache}/{_sig(m[0], m[2], pairs)}" for m in models}
    preds = {}
    if use_cache:
        for m in models:
            if os.path.exists(cfile[m[0]]):
                preds[m[0]] = np.load(cfile[m[0]]).astype(np.float32)
    need = [m for m in models if m[0] not in preds]
    if not need:
        print(f"[predict] all {len(models)} models cached (checkpoint-keyed)")
        return preds

    import torch
    import torch.nn.functional as F
    from PIL import Image
    dev = "cuda"
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
    rgbs = [np.array(Image.open(p).convert("RGB").resize((RW, RH))) for p in pairs]

    def da2_preds(ck):
        from transformers import AutoModelForDepthEstimation
        m = AutoModelForDepthEstimation.from_pretrained(DA2_HF).to(dev)
        if ck:
            m.load_state_dict(torch.load(ck))
        m.eval()
        out = []
        for r in rgbs:
            t = torch.from_numpy(r.astype(np.float32) / 255).permute(2, 0, 1)[None]
            t = F.interpolate(t, (518, 518), mode="bilinear", align_corners=False).to(dev)
            t = (t - MEAN) / STD
            with torch.no_grad():
                d = m(pixel_values=t).predicted_depth
            out.append(F.interpolate(d[:, None], size=(RH, RW), mode="bilinear", align_corners=False)[0, 0].cpu().numpy())
        del m
        torch.cuda.empty_cache()
        return np.stack(out)

    da3 = [None]

    def da3_preds(ck):
        from depth_anything_3.api import DepthAnything3
        if da3[0] is None:
            da3[0] = DepthAnything3.from_pretrained(DA3_HF).to(dev)
        da3[0].model.load_state_dict(DepthAnything3.from_pretrained(DA3_HF).model.state_dict())
        if ck:
            da3[0].model.load_state_dict(torch.load(ck), strict=False)
        da3[0].model.eval()
        out = []
        for r in rgbs:
            im, _, _ = da3[0]._preprocess_inputs([r])
            im = im.unsqueeze(0).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                d = da3[0].model(im, None, None, [], False, False, "saddle_balanced")["depth"][0, 0].float()
            if DA3_FOCAL["on"]:
                d = d * (DA3_FOCAL["fx"] * im.shape[-1] / r.shape[1]) / 300.0
            out.append(F.interpolate(d[None, None], size=(RH, RW), mode="bilinear", align_corners=False)[0, 0].cpu().numpy())
        torch.cuda.empty_cache()
        return np.stack(out)

    for label, kind, ck in need:
        print(f"[predict] {label} ...", flush=True)
        p = da2_preds(ck) if kind == "da2" else da3_preds(ck)
        if use_cache:
            np.save(cfile[label], p.astype(np.float16))
        preds[label] = p
    return preds


def scale_for(preds, gts, variant):
    """Return a per-frame scale array for the chosen variant (1.0 raw / global s* const)."""
    if variant in ("sitecal", "const"):
        r = np.concatenate([g[(g > 0.2) & (g < 15)] / np.clip(p[(g > 0.2) & (g < 15)], 1e-6, None)
                            for p, g in zip(preds, gts)])
        return float(np.median(r))
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realgt", required=True)
    ap.add_argument("--ft-dir", default=None, help="dir with da{2,3}_{real,realaug}_c/best.pt (default roster)")
    ap.add_argument("--models", default=None, help="override: 'Label=kind:ckpt,...' (ckpt empty = zero-shot)")
    ap.add_argument("--bands", default="0-0.5,0.5-1,1-2,2-15")
    ap.add_argument("--variant", choices=["raw", "sitecal", "const", "aligned"], default="raw")
    ap.add_argument("--metric", choices=["d1", "mae", "both"], default="both")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--min-px", type=int, default=50)
    ap.add_argument("--no-cache", action="store_true", help="ignore + skip the prediction cache (force recompute)")
    ap.add_argument("--da3-focal", type=int, default=0,
                    help="1 = DA3 metric depth = output x fx/300 (model-card rule; use for checkpoints trained with "
                         "finetune.py --da3-focal 1). 0 = raw canonical output")
    ap.add_argument("--fx", type=float, default=0.0, help="fx at 640x480 (0 = manifest.json of --realgt or PortalCam default)")
    args = ap.parse_args()
    if args.da3_focal:
        from finetune import fx_for_dir
        DA3_FOCAL.update(on=True, fx=args.fx or fx_for_dir(args.realgt))

    pairs = sorted(glob.glob(f"{args.realgt}/*_rgb.jpg"))
    gts = [np.load(p.replace("_rgb.jpg", "_depth.npy")).astype(np.float32) for p in pairs]
    bands = parse_bands(args.bands)
    models = parse_models(args.models) if args.models else default_roster(args.ft_dir)
    cache = args.cache or f"{args.realgt}/_predcache"
    preds = predict_all(models, pairs, cache, use_cache=not args.no_cache)

    def cell(P, gts, lo, hi, s_global):
        d1s, maes = [], []
        for p, g in zip(P, gts):
            m = (g >= lo) & (g < hi)
            if m.sum() < args.min_px:
                continue
            a, b = p[m], g[m]
            if args.variant == "aligned":
                a = a * (np.median(b) / max(np.median(a), 1e-6))
            else:
                a = a * s_global
            d1s.append(np.mean(np.maximum(a / b, b / a) < 1.25))
            maes.append(np.mean(np.abs(a - b)))
        n = len(d1s)
        return (float(np.mean(d1s)) if n else float("nan"),
                float(np.mean(maes)) * 100 if n else float("nan"), n)

    span = (min(b[0] for b in bands), max(b[1] for b in bands))
    cols = bands + [span]
    names = [f"{lo:g}-{hi:g}" for lo, hi in bands] + ["all"]
    print(f"\nband-eval | {args.realgt.split('/')[-1]} | {len(pairs)} frames | variant={args.variant}")
    table = {}
    for label, kind, ck in models:
        P = preds[label]
        s = scale_for(P, gts, args.variant)
        table[label] = {"s": s, "bands": {}}
        for (lo, hi), nm in zip(cols, names):
            d1, mae, n = cell(P, gts, lo, hi, s)
            table[label]["bands"][nm] = dict(d1=d1, mae=mae, n=n)

    vname = {"raw": "raw output", "sitecal": "site-cal (1 global scalar)",
             "const": "site-cal (1 global scalar)", "aligned": "per-frame aligned (ORACLE)"}[args.variant]
    band_hdr = "  ".join(f"{n} m ({table[models[0][0]]['bands'][n]['n']}f)" for n in names)
    show_s = args.variant in ("sitecal", "const")

    def block(title, key, fmtcell, wcol=13):
        print(f"\n{title}")
        print(f"  bands (m) [# frames with pixels in band]:")
        print(f"  {'model':<14}{'  s*  ' if show_s else '  '}" + "  ".join(f"{n+'m':>{wcol}}" for n in names))
        for label, _, _ in models:
            row = f"  {label:<14}" + (f" {table[label]['s']:.2f} " if show_s else "  ")
            row += "  ".join(f"{fmtcell(table[label]['bands'][n][key]):>{wcol}}" for n in names)
            print(row)

    print(f"\n=== band-eval: {args.realgt.split('/')[-1]} · {len(pairs)} frames · variant = {vname} ===")
    if args.metric in ("d1", "both"):
        block("METRIC: delta1 = fraction of pixels within 25% of GT depth  (↑ better, max 1.0)",
              "d1", lambda v: f"{v:.3f}")
    if args.metric in ("mae", "both"):
        block("METRIC: MAE = mean absolute depth error in CENTIMETRES  (↓ better)",
              "mae", lambda v: f"{v:.0f}")
    outp = f"{args.realgt}/bandeval_{args.variant}.json"
    json.dump(table, open(outp, "w"), indent=2)
    print(f"\nsaved {outp}")


if __name__ == "__main__":
    main()

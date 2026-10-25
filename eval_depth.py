"""Evaluate depth models on a real-GT set with a near / mid / far band breakdown.

Reports metric and median-aligned AbsRel + delta1, overall and per band.

Usage:
    python eval_depth.py --realgt <real_gt_dir> --ckpt-dir <finetune_runs_dir> [--out results.json]
"""
import argparse
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

RW, RH = 640, 480
DEV = "cuda"
DA2_HF = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DA3_HF = "depth-anything/da3metric-large"
BANDS = [("near", 0.2, 1.5), ("mid", 1.5, 3.0), ("far", 3.0, 15.0)]

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)


def band_metrics(pred, gt):
    """Return {band: [absrel, d1, align_absrel, align_d1]} for one frame."""
    if gt.shape != (RH, RW):
        gt = F.interpolate(torch.tensor(gt)[None, None], size=(RH, RW), mode="nearest")[0, 0].numpy()
    out = {}
    for name, lo, hi in [("all", 0.2, 15.0)] + BANDS:
        m = (gt > lo) & (gt < hi)
        if m.sum() < 100:
            out[name] = None
            continue
        p, g = pred[m], gt[m]
        ps = p * (np.median(g) / max(np.median(p), 1e-6))
        out[name] = [float(np.mean(np.abs(p - g) / g)),
                     float(np.mean(np.maximum(p / g, g / p) < 1.25)),
                     float(np.mean(np.abs(ps - g) / g)),
                     float(np.mean(np.maximum(ps / g, g / ps) < 1.25))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realgt", required=True)
    ap.add_argument("--ckpt-dir", required=True,
                    help="dir holding run_realft_da2/best.pt and run_realft_da3/best_head.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForDepthEstimation
    from depth_anything_3.api import DepthAnything3

    pairs = sorted(glob.glob(f"{args.realgt}/*_rgb.jpg"))
    print(f"[eval] {len(pairs)} real-GT pairs")
    gt_of = lambda p: np.load(p.replace("_rgb.jpg", "_depth.npy")).astype(np.float32)
    rgb_of = lambda p: np.array(Image.open(p).convert("RGB").resize((RW, RH)))

    def da2_pred(model, rgb):
        t = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None]
        t = F.interpolate(t, (518, 518), mode="bilinear", align_corners=False).to(DEV)
        t = (t - MEAN) / STD
        with torch.no_grad():
            d = model(pixel_values=t).predicted_depth
        return F.interpolate(d[:, None], size=(RH, RW), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

    ACD = torch.bfloat16
    da3 = DepthAnything3.from_pretrained(DA3_HF).to(DEV)

    def da3_pred(path):
        imgs, _, _ = da3._preprocess_inputs([path])
        imgs = imgs.unsqueeze(0).to(DEV)
        da3.model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=ACD):
            d = da3.model(imgs, None, None, [], False, False, "saddle_balanced")["depth"][0, 0].float()
        return F.interpolate(d[None, None], size=(RH, RW), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

    models = [
        ("DA2_zeroshot", "da2", None),
        ("DA2_realft",   "da2", f"{args.ckpt_dir}/run_realft_da2/best.pt"),
        ("DA3_zeroshot", "da3", None),
        ("DA3_realft",   "da3", f"{args.ckpt_dir}/run_realft_da3/best_head.pt"),
    ]

    results = {}
    for label, kind, ckpt in models:
        print(f"[eval] {label} ...", flush=True)
        if kind == "da2":
            model = AutoModelForDepthEstimation.from_pretrained(DA2_HF).to(DEV)
            if ckpt:
                model.load_state_dict(torch.load(ckpt))
            model.eval()
            preds = [da2_pred(model, rgb_of(p)) for p in pairs]
            del model
        else:
            da3.model.load_state_dict(DepthAnything3.from_pretrained(DA3_HF).model.state_dict())
            if ckpt:
                da3.model.load_state_dict(torch.load(ckpt), strict=False)
            preds = [da3_pred(p) for p in pairs]
        torch.cuda.empty_cache()

        per = [band_metrics(pr, gt_of(p)) for pr, p in zip(preds, pairs)]
        agg = {}
        for band in ["all"] + [b[0] for b in BANDS]:
            rows = np.array([f[band] for f in per if f[band] is not None])
            agg[band] = dict(absrel=float(rows[:, 0].mean()), d1=float(rows[:, 1].mean()),
                             align_absrel=float(rows[:, 2].mean()), align_d1=float(rows[:, 3].mean()),
                             n=len(rows))
        results[label] = agg

    print("\n" + "=" * 78)
    hdr = f"{'model':<15}" + "".join(f"{b:>15}" for b in ["all", "near", "mid", "far"])
    for metric, title in [("d1", "metric delta1"), ("align_d1", "aligned delta1"),
                          ("absrel", "metric AbsRel"), ("align_absrel", "aligned AbsRel")]:
        print(f"\n[{title}]  (near<1.5m  mid1.5-3  far>3)")
        print(hdr)
        for label in results:
            row = f"{label:<15}"
            for b in ["all", "near", "mid", "far"]:
                row += f"{results[label][b][metric]:>15.3f}"
            print(row)

    out = args.out or f"{args.realgt}/eval_results.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n[eval] wrote {out}")


if __name__ == "__main__":
    main()

"""Final evaluation of DA3 and MoGe-2 (zero-shot and fine-tuned) on all test sets.

Uses the eval_protocol sets and metrics. DA3 predictions are converted to metric depth
as canonical * fx / 300, with fx the focal length of the image as fed.

Usage:
    python eval_v2.py --kind da3|moge [--ckpt best.pt] --out eval.json [--scales 1.0]
    python eval_v2.py --kind da3|moge --sstar --out sstar.json
"""
import argparse, glob, json, os, sys
import numpy as np, torch
from PIL import Image
import eval_protocol as FE
import finetune as FT
from bootstrap import frame_stats, stat

SSTAR_SETS = "scans/final_eval_2026-09-29/evals/sstar_zeroshot.json"
_FX = {}


def fx_of_dir(d):
    if d not in _FX:
        _FX[d] = FT.fx_for_dir(d)
    return _FX[d]


def build(kind, ckpt):
    if kind == "da3":
        W = FT.DA3Wrapper(); W.focal = True
    else:
        W = FT.MoGeWrapper()
    if ckpt:
        sd = torch.load(ckpt, map_location="cuda")
        miss = W.model.load_state_dict(sd, strict=False)
        print("loaded", ckpt, "tensors", len(sd), "unexpected", len(miss.unexpected_keys), flush=True)
    W.model.eval()
    return W


def make_predict(W):
    @torch.no_grad()
    def predict(_net, _api, paths):
        fx = fx_of_dir(os.path.dirname(paths[0]))
        return [W.predict(np.array(Image.open(p).convert("RGB").resize((640, 480))), (480, 640), fx=fx).cpu().numpy()
                for p in paths]
    return predict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["da3", "moge"]); ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--scales", default=""); ap.add_argument("--sstar", action="store_true")
    a = ap.parse_args()
    W = build(a.kind, a.ckpt)
    FE.predict = make_predict(W)
    if a.sstar:
        out = {}
        for d in json.load(open(SSTAR_SETS)):
            paths = sorted(glob.glob(f"{d}/rgb_*.jpg"))
            G = [np.load(p.replace("rgb_", "depth_").replace(".jpg", ".npy")).astype(np.float32) for p in paths]
            fr = frame_stats(FE.predict(None, None, paths), G, 0, 15)
            out[d] = dict(s_star=float(stat(fr, range(len(fr)), "sitecal")[1]), n=len(paths), fx=fx_of_dir(d))
            print(d, out[d], flush=True)
        json.dump(out, open(a.out, "w"), indent=1); sys.exit()
    scales = [float(x) for x in a.scales.split(",") if x]
    json.dump(FE.evaluate(None, None, scales), open(a.out, "w"), indent=1)
    print("saved", a.out)

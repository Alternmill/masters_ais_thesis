"""Location finetune for DA2 (transformers), DA3 (depth-anything-3) and MoGe-2.

Recipes: head-only (backbone frozen) or LoRA on the backbone attention (--lora R).
Loss L1 + SiLog; the checkpoint is selected on the eval set's
median-aligned delta1 (--select).
DA3 outputs canonical depth; metric depth = output * fx / 300, with fx the focal length
of the image as fed to the network (--da3-focal 1). Optional per-pixel Laplace
uncertainty head (--uncert laplace).

Usage:
    python finetune.py --model da2|da3|moge --data <train_val_dir> --eval <realgt_dir> --out <run_dir> [--steps 600] [--lora 8]
"""
import argparse
import glob
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eval_depth import band_metrics, BANDS

RW, RH = 640, 480
DEV = "cuda"
DA2_HF = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DA3_HF = "depth-anything/da3metric-large"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)


DA3_CANON_F = 300.0
PORTALCAM_FX_1280 = 614.0


def fx_for_dir(d, width=RW):
    """fx (px) of the images in dir `d` once resized to `width`: manifest.json 'intrinsics' of d or its parents, else PortalCam."""
    d = os.path.abspath(d)
    for _ in range(4):
        for c in (f"{d}/manifest.json",):
            if os.path.exists(c):
                try:
                    j = json.load(open(c))
                    if isinstance(j, dict) and "intrinsics" in j:
                        k = j["intrinsics"]
                        return float(k["fx"]) * width / float(k["w"])
                except Exception:
                    pass
        d = os.path.dirname(d)
    return PORTALCAM_FX_1280 * width / 1280.0


def silog(p, g, m, lam=0.5):
    dd = torch.log(p[m].clamp_min(1e-3)) - torch.log(g[m].clamp_min(1e-3))
    return torch.sqrt((dd ** 2).mean() - lam * (dd.mean() ** 2) + 1e-6)


class DA2Wrapper:
    tag = "da2"

    def __init__(self):
        from transformers import AutoModelForDepthEstimation
        self.model = AutoModelForDepthEstimation.from_pretrained(DA2_HF).to(DEV)

    def freeze_backbone(self):
        nb = nt = 0
        for n, p in self.model.named_parameters():
            train = not n.startswith("backbone")
            p.requires_grad_(train)
            nt += p.numel() * train
            nb += p.numel() * (not train)
        return nb, nt

    def predict(self, rgb_np, out_hw):
        t = torch.from_numpy(rgb_np.astype(np.float32) / 255).permute(2, 0, 1)[None]
        t = F.interpolate(t, (518, 518), mode="bilinear", align_corners=False).to(DEV)
        t = (t - MEAN) / STD
        d = self.model(pixel_values=t).predicted_depth
        return F.interpolate(d[:, None], size=out_hw, mode="bilinear", align_corners=False)[0, 0]

    def save(self, path):
        torch.save(self.model.state_dict(), path)


class LoRALinear(torch.nn.Module):
    """y = W0 x + (alpha/r) B A x ; W0 frozen, A,B trainable (B init 0 -> starts at the pretrained model)."""

    def __init__(self, base, r=16, alpha=16):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) * (1.0 / r))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t().to(x.dtype) @ self.B.t().to(x.dtype)) * self.scale

    def merged_weight(self):
        return self.base.weight.data + self.scale * (self.B.data @ self.A.data).to(self.base.weight.dtype)


class UncertHead(torch.nn.Module):
    """Per-pixel Laplace scale b on log-depth, from the DPT neck feature (input of output_conv2).
    b = softplus(x) + 1e-3; initialised so that b ~ 0.1 everywhere."""

    def __init__(self, cin=128):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Conv2d(cin, 32, 3, padding=1), torch.nn.ReLU(inplace=True),
                                       torch.nn.Conv2d(32, 1, 1))
        torch.nn.init.normal_(self.net[2].weight, std=1e-3)
        torch.nn.init.constant_(self.net[2].bias, float(np.log(np.expm1(0.1))))

    def forward(self, feat):
        return F.softplus(self.net(feat)) + 1e-3


class DA3Wrapper:
    tag = "da3"

    def __init__(self):
        from depth_anything_3.api import DepthAnything3
        self.da3 = DepthAnything3.from_pretrained(DA3_HF).to(DEV)
        self.model = self.da3.model
        self.ACD = torch.bfloat16
        self.focal = False

    def _conv(self, d, imgs, rgb_np, fx):
        if not self.focal:
            return d
        assert fx, "DA3 focal conversion needs fx of the input image"
        return d * (fx * imgs.shape[-1] / rgb_np.shape[1]) / DA3_CANON_F

    def freeze_backbone(self):
        nb = nt = 0
        for n, p in self.model.named_parameters():
            train = not n.startswith("backbone.")
            p.requires_grad_(train)
            nt += p.numel() * train
            nb += p.numel() * (not train)
        return nb, nt

    def predict(self, rgb_np, out_hw, fx=None):
        imgs, _, _ = self.da3._preprocess_inputs([rgb_np])
        imgs = imgs.unsqueeze(0).to(DEV)
        with torch.autocast("cuda", dtype=self.ACD):
            d = self.model(imgs, None, None, [], False, False, "saddle_balanced")["depth"][0, 0].float()
        d = self._conv(d, imgs, rgb_np, fx)
        return F.interpolate(d[None, None], size=out_hw, mode="bilinear", align_corners=False)[0, 0]

    def add_uncert(self, detach_feat=True):
        """Attach an uncertainty head fed by the DPT neck feature (captured with a forward hook)."""
        self.uncert_detach = detach_feat
        cin = self.model.head.scratch.output_conv2[0].in_channels
        self.model.uncert_head = UncertHead(cin).to(DEV)
        self._feat = None

        def hook(mod, inp):
            self._feat = inp[0]
        self.model.head.scratch.output_conv2.register_forward_pre_hook(hook)
        return sum(p.numel() for p in self.model.uncert_head.parameters())

    def predict_with_uncert(self, rgb_np, out_hw, fx=None):
        """-> (depth [H,W] metres, b [H,W] Laplace scale of log-depth)."""
        imgs, _, _ = self.da3._preprocess_inputs([rgb_np])
        imgs = imgs.unsqueeze(0).to(DEV)
        with torch.autocast("cuda", dtype=self.ACD):
            d = self.model(imgs, None, None, [], False, False, "saddle_balanced")["depth"][0, 0].float()
            feat = self._feat.detach() if self.uncert_detach else self._feat
            b = self.model.uncert_head(feat)[0, 0].float()
        d = self._conv(d, imgs, rgb_np, fx)
        d = F.interpolate(d[None, None], size=out_hw, mode="bilinear", align_corners=False)[0, 0]
        b = F.interpolate(b[None, None], size=out_hw, mode="bilinear", align_corners=False)[0, 0]
        return d, b

    def add_lora(self, r):
        """LoRA on every backbone attention qkv (ViT-L: 24 layers); call after freeze_backbone()."""
        self.lora = {}
        for name, mod in list(self.model.named_modules()):
            if name.startswith("backbone.") and name.endswith("attn.qkv") and isinstance(mod, torch.nn.Linear):
                parent = self.model.get_submodule(name.rsplit(".", 1)[0])
                wrapped = LoRALinear(mod, r=r, alpha=r)
                setattr(parent, "qkv", wrapped)
                self.lora[name] = wrapped
        return sum(p.numel() for w in self.lora.values() for p in (w.A, w.B))

    def save(self, path):
        sd = {k: v for k, v in self.model.state_dict().items() if not k.startswith("backbone.")}
        for name, w in getattr(self, "lora", {}).items():
            sd[f"{name}.weight"] = w.merged_weight().cpu()
            sd[f"{name}.bias"] = w.base.bias.data.cpu() if w.base.bias is not None else None
        torch.save({k: v for k, v in sd.items() if v is not None}, path)


class MoGeWrapper:
    """MoGe-2-L (metric) fine-tuning with the DA3 recipe. Depth = (points_z + shift) x metric_scale, exactly as
    MoGeModel.infer with a known FoV (from fx); the shift is solved without gradient (recover_focal_shift), gradients
    flow through points_z and metric_scale. Trainable: neck + points_head + scale_head (+ LoRA on encoder qkv)."""
    tag = "moge"
    HF = "Ruicheng/moge-2-vitl-normal"

    def __init__(self, num_tokens=3600):
        from moge.model.v2 import MoGeModel
        self.model = MoGeModel.from_pretrained(self.HF).to(DEV)
        self.num_tokens = num_tokens
        self.focal = True

    TRAIN_PREFIX = ("neck.", "points_head.", "scale_head.")

    def freeze_backbone(self):
        nb = nt = 0
        for n, p in self.model.named_parameters():
            train = n.startswith(self.TRAIN_PREFIX)
            p.requires_grad_(train)
            nt += p.numel() * train
            nb += p.numel() * (not train)
        return nb, nt

    def add_lora(self, r):
        self.lora = {}
        for name, mod in list(self.model.named_modules()):
            if name.startswith("encoder.") and name.endswith("attn.qkv") and isinstance(mod, torch.nn.Linear):
                parent = self.model.get_submodule(name.rsplit(".", 1)[0])
                wrapped = LoRALinear(mod, r=r, alpha=r)
                setattr(parent, "qkv", wrapped)
                self.lora[name] = wrapped
        return sum(p.numel() for w in self.lora.values() for p in (w.A, w.B))

    def predict(self, rgb_np, out_hw, fx=None):
        from moge.utils.geometry_torch import recover_focal_shift
        assert fx, "MoGe needs fx"
        x = torch.from_numpy(np.ascontiguousarray(rgb_np)).float().div(255).permute(2, 0, 1)[None].to(DEV)
        h, w = x.shape[-2:]; ar = w / h
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = self.model(x, num_tokens=self.num_tokens)
        pts, ms = o["points"].float(), o["metric_scale"].float()
        fov_x = 2 * np.arctan(w / 2 / fx)
        focal = torch.tensor([ar / (1 + ar ** 2) ** 0.5 / np.tan(fov_x / 2)], device=DEV, dtype=torch.float32)
        with torch.no_grad():
            mask = o["mask"] > 0.5 if o.get("mask") is not None else None
            _, shift = recover_focal_shift(pts.detach(), mask, focal=focal)
        d = (pts[..., 2] + shift[..., None, None]) * ms[:, None, None]
        return F.interpolate(d[:, None], size=out_hw, mode="bilinear", align_corners=False)[0, 0]

    def save(self, path):
        sd = {k: v for k, v in self.model.state_dict().items() if k.startswith(self.TRAIN_PREFIX)}
        for name, w in getattr(self, "lora", {}).items():
            sd[f"{name}.weight"] = w.merged_weight().cpu()
            if w.base.bias is not None:
                sd[f"{name}.bias"] = w.base.bias.data.cpu()
        torch.save(sd, path)


def photo_aug(rgb, rng):
    """Cross-session lighting/exposure jitter: brightness, contrast, gamma, per-channel white balance, blur, noise."""
    x = rgb.astype(np.float32) / 255.0
    x = x * rng.uniform(0.7, 1.3)
    x = (x - x.mean()) * rng.uniform(0.8, 1.2) + x.mean()
    x = np.clip(x, 0, 1) ** rng.uniform(0.8, 1.25)
    x = x * rng.uniform(0.9, 1.1, size=3)[None, None]
    if rng.random() < 0.3:
        import cv2
        k = int(rng.choice([3, 5])); x = cv2.GaussianBlur(x, (k, k), 0)
    if rng.random() < 0.3:
        x = x + rng.normal(0, rng.uniform(0.005, 0.02), x.shape)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def edge_band(d, px):
    """GT depth-discontinuity band (log-depth gradient > 0.4, dilated) — mixed/uncertain GT pixels."""
    import cv2
    lg = np.log(np.clip(d, 0.1, None)); gx = cv2.Sobel(lg, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(lg, cv2.CV_32F, 0, 1, ksize=3)
    e = (np.hypot(gx, gy) > 0.4).astype(np.uint8)
    return cv2.dilate(e, np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool)


def load_pair(rgb_path):
    rgb = np.array(Image.open(rgb_path).convert("RGB").resize((RW, RH)))
    d = np.load(rgb_path.replace("rgb_", "depth_").replace(".jpg", ".npy")).astype(np.float32)
    return rgb, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["da2", "da3", "moge"])
    ap.add_argument("--data", required=True, help="dir with train/ and val/ rgb_*.jpg + depth_*.npy")
    ap.add_argument("--eval", required=True, help="dir with <stem>_rgb.jpg/_depth.npy (the eval target)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--far", type=float, default=15.0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lora", type=int, default=0, help="DA3/MoGe-2: LoRA rank on backbone qkv (0 = head-only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-data", default="", help="comma-separated extra dirs with train/rgb_*.jpg (multi-session)")
    ap.add_argument("--photo-aug", type=float, default=0.0, help="probability of photometric jitter per sample")
    ap.add_argument("--loss", default="l1silog", choices=["l1silog", "logl1silog", "logl1", "silog"])
    ap.add_argument("--batch", type=int, default=1, help="images per optimizer step (gradient accumulation)")
    ap.add_argument("--edge-mask", type=int, default=0, help="drop a GT depth-edge band of this radius (px) from the loss")
    ap.add_argument("--select", default="align_d1", choices=["align_d1", "d1"], help="checkpoint criterion on --eval")
    ap.add_argument("--uncert", default="none", choices=["none", "laplace"], help="DA3: add a per-pixel uncertainty head")
    ap.add_argument("--uncert-lambda", type=float, default=1.0, help="weight of the Laplace NLL (depth detached inside it)")
    ap.add_argument("--uncert-shared", type=int, default=0,
                    help="1 = let the NLL gradient flow into the shared DPT neck/LoRA (0 = head sees a detached feature)")
    ap.add_argument("--da3-focal", type=int, default=1,
                    help="DA3: train/eval on metric depth = canonical output x f_in/300 (model-card rule). "
                         "0 = raw canonical output")
    ap.add_argument("--fx", type=float, default=0.0,
                    help="fx (px) of the images at 640x480; 0 = manifest.json of the data/eval dir or PortalCam default")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = open(f"{args.out}/train.log", "a")

    def L(*m):
        s = " ".join(str(x) for x in m)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    W = DA2Wrapper() if args.model == "da2" else (MoGeWrapper() if args.model == "moge" else DA3Wrapper())
    if args.model == "da3":
        W.focal = bool(args.da3_focal)
    nb, nt = W.freeze_backbone()
    L(f"[{W.tag}] frozen backbone {nb/1e6:.1f}M | trainable {nt/1e6:.1f}M")
    if args.lora:
        assert args.model in ("da3", "moge"), "--lora implemented for da3/moge"
        nl = W.add_lora(args.lora)
        L(f"[{W.tag}] + LoRA r={args.lora} on {len(W.lora)} qkv layers: {nl/1e6:.2f}M adapter params")

    if args.uncert != "none":
        assert args.model == "da3", "--uncert implemented for da3"
        nu = W.add_uncert(detach_feat=not args.uncert_shared)
        L(f"[{W.tag}] + uncertainty head (Laplace, lambda {args.uncert_lambda}, shared={args.uncert_shared}): {nu} params")

    train = sorted(glob.glob(f"{args.data}/train/rgb_*.jpg"))
    for ex in [e for e in args.extra_data.split(",") if e]:
        train += sorted(glob.glob(f"{ex}/train/rgb_*.jpg"))
    eval_pairs = sorted(glob.glob(f"{args.eval}/*_rgb.jpg"))
    fx_cache = {}

    def fx_of(path):
        if args.fx:
            return args.fx
        dd = os.path.dirname(path)
        if dd not in fx_cache:
            fx_cache[dd] = fx_for_dir(dd)
        return fx_cache[dd]
    L(f"[{W.tag}] focal conversion {'ON' if getattr(W, 'focal', False) else 'off'}; "
      f"fx train {fx_of(train[0]):.1f} eval {fx_of(eval_pairs[0]):.1f} (px at {RW}x{RH})")
    L(f"[{W.tag}] train {len(train)} | eval {len(eval_pairs)}")

    @torch.no_grad()
    def evaluate():
        W.model.eval()
        per, ratios = [], []
        for p in eval_pairs:
            rgb = np.array(Image.open(p).convert("RGB").resize((RW, RH)))
            gt = np.load(p.replace("_rgb.jpg", "_depth.npy")).astype(np.float32)
            pred = (W.predict(rgb, (RH, RW), fx=fx_of(p)) if W.tag in ("da3", "moge") else W.predict(rgb, (RH, RW))).cpu().numpy()
            per.append(band_metrics(pred, gt))
            mm = (gt > 0.2) & (gt < 15)
            ratios.append((gt[mm] / np.clip(pred[mm], 1e-6, None))[::7])
        agg = {}
        for band in ["all"] + [b[0] for b in BANDS]:
            rows = np.array([f[band] for f in per if f[band] is not None])
            agg[band] = dict(absrel=float(rows[:, 0].mean()), d1=float(rows[:, 1].mean()),
                             align_absrel=float(rows[:, 2].mean()), align_d1=float(rows[:, 3].mean()))
        agg["s_star"] = float(np.median(np.concatenate(ratios)))
        return agg

    base = evaluate()
    L(f"[{W.tag}] BASELINE eval: all d1 {base['all']['d1']:.3f} align_d1 {base['all']['align_d1']:.3f} "
      f"| near align_d1 {base['near']['align_d1']:.3f} near absrel {base['near']['absrel']:.3f}")

    lr = args.lr or (1e-4 if args.model == "da2" else 2e-5)
    params = [p for p in W.model.parameters() if p.requires_grad]
    uhead = list(W.model.uncert_head.parameters()) if args.uncert != "none" else []
    groups = [{"params": [p for p in params if not any(p is q for w in getattr(W, "lora", {}).values() for q in (w.A, w.B))
                          and not any(p is q for q in uhead)], "lr": lr}]
    if uhead:
        groups.append({"params": uhead, "lr": 1e-3})
    if args.lora:
        groups.append({"params": [q for w in W.lora.values() for q in (w.A, w.B)], "lr": lr * 5})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    best, hist, t0 = -1.0, [], time.time()
    for step in range(1, args.steps + 1):
        W.model.train()
        opt.zero_grad()
        nacc = 0
        for _ in range(args.batch):
            tp = train[rng.integers(len(train))]
            rgb, d = load_pair(tp)
            if rng.random() < 0.5:
                rgb = rgb[:, ::-1].copy()
                d = d[:, ::-1].copy()
            if args.photo_aug and rng.random() < args.photo_aug:
                rgb = photo_aug(rgb, rng)
            g = torch.tensor(d, device=DEV)
            kw = dict(fx=fx_of(tp)) if W.tag in ("da3", "moge") else {}
            if args.uncert != "none":
                pred, ub = W.predict_with_uncert(rgb, g.shape, **kw)
            else:
                pred = W.predict(rgb, g.shape, **kw)
            mnp = (d > 0.1) & (d < args.far)
            if args.edge_mask:
                mnp &= ~edge_band(d, args.edge_mask)
            m = torch.tensor(mnp, device=DEV)
            if m.sum() < 100:
                continue
            if args.loss == "l1silog":
                loss = F.l1_loss(pred[m], g[m]) + 0.5 * silog(pred, g, m)
            elif args.loss == "logl1silog":
                loss = (torch.log(pred[m].clamp_min(1e-3)) - torch.log(g[m])).abs().mean() + 0.5 * silog(pred, g, m)
            elif args.loss == "logl1":
                loss = (torch.log(pred[m].clamp_min(1e-3)) - torch.log(g[m])).abs().mean()
            else:
                loss = silog(pred, g, m)
            if args.uncert != "none":
                r = (torch.log(pred[m].detach().clamp_min(1e-3)) - torch.log(g[m])).abs()
                nll = (r / ub[m] + torch.log(ub[m])).mean()
                loss = loss + args.uncert_lambda * nll
            (loss / args.batch).backward()
            nacc += 1
        if nacc == 0:
            continue
        uparams = list(W.model.uncert_head.parameters()) if args.uncert != "none" else []
        torch.nn.utils.clip_grad_norm_([p for p in W.model.parameters() if p.requires_grad and not any(p is q for q in uparams)], 1.0)
        if uparams:
            torch.nn.utils.clip_grad_norm_(uparams, 1.0)
        opt.step()
        if step % 100 == 0 or step == args.steps:
            ev = evaluate()
            ev["step"] = step
            hist.append(ev)
            L(f"[{W.tag}] step {step}/{args.steps} loss {loss.item():.3f} | "
              f"all d1 {ev['all']['d1']:.3f} align_d1 {ev['all']['align_d1']:.3f} | "
              f"NEAR d1 {ev['near']['d1']:.3f} align_d1 {ev['near']['align_d1']:.3f} absrel {ev['near']['absrel']:.3f}")
            json.dump(dict(baseline=base, history=hist), open(f"{args.out}/history.json", "w"), indent=2)
            crit = ev["all"][args.select]
            if crit > best:
                best = crit
                W.save(f"{args.out}/best.pt")
                json.dump(ev, open(f"{args.out}/best.json", "w"), indent=2)
                L(f"  saved best.pt (all {args.select} {best:.3f}, eval s* {ev['s_star']:.3f})")
    L(f"[{W.tag}] DONE best all-align_d1 {best:.3f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

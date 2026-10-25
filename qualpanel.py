"""
Qualitative depth panel: for chosen frames, tile [real | GT | model1 | ...] with each
depth in turbo under the GT's own colour scale, so a correct model looks like the GT.

Usage:
    python qualpanel.py --realgt <dir> --models "GT=,DA3 zs=da3:,DA3 ft=da3:/ck" \
        --frames f0123,f0456 --out panel.jpg [--align]
"""
import argparse, glob, numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg"); import matplotlib.cm as cm
RW, RH = 640, 480; DEV = "cuda"
DA2_HF = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"; DA3_HF = "depth-anything/da3metric-large"
MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); STD = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realgt", required=True); ap.add_argument("--models", required=True)
    ap.add_argument("--frames", required=True, help="comma list of frame stems (e.g. f0123) or 'auto:N'")
    ap.add_argument("--out", required=True); ap.add_argument("--align", action="store_true",
                    help="median-align each prediction to GT before colouring (show structure, hide scale)")
    a = ap.parse_args()
    specs = []
    for s in a.models.split(","):
        lbl, rest = s.split("="); kind, ck = (rest.split(":",1)+[""])[:2] if ":" in rest else (rest, "")
        specs.append((lbl.strip(), kind.strip(), (ck or "").strip() or None))

    pairs = sorted(glob.glob(f"{a.realgt}/*_rgb.jpg"))
    stems = [p.split("/")[-1].replace("_rgb.jpg","") for p in pairs]
    if a.frames.startswith("auto:"):
        n = int(a.frames.split(":")[1]); sel = [stems[i] for i in np.linspace(0,len(stems)-1,n).round().astype(int)]
    else:
        sel = a.frames.split(",")

    glob_mean = MEAN.to(DEV); glob_std = STD.to(DEV)
    da3 = [None]
    def predict(kind, ck, rgb):
        if kind == "da2":
            from transformers import AutoModelForDepthEstimation
            m = AutoModelForDepthEstimation.from_pretrained(DA2_HF).to(DEV)
            if ck: m.load_state_dict(torch.load(ck))
            m.eval()
            t = torch.from_numpy(rgb.astype(np.float32)/255).permute(2,0,1)[None]
            t = F.interpolate(t,(518,518),mode="bilinear",align_corners=False).to(DEV); t=(t-glob_mean)/glob_std
            with torch.no_grad(): d = m(pixel_values=t).predicted_depth
            out = F.interpolate(d[:,None],size=(RH,RW),mode="bilinear",align_corners=False)[0,0].cpu().numpy(); del m
            return out
        from depth_anything_3.api import DepthAnything3
        if da3[0] is None: da3[0] = DepthAnything3.from_pretrained(DA3_HF).to(DEV)
        da3[0].model.load_state_dict(DepthAnything3.from_pretrained(DA3_HF).model.state_dict())
        if ck: da3[0].model.load_state_dict(torch.load(ck), strict=False)
        da3[0].model.eval()
        im,_,_ = da3[0]._preprocess_inputs([rgb]); im=im.unsqueeze(0).to(DEV)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            d = da3[0].model(im,None,None,[],False,False,"saddle_balanced")["depth"][0,0].float()
        return F.interpolate(d[None,None],size=(RH,RW),mode="bilinear",align_corners=False)[0,0].cpu().numpy()

    def turbo(d, lo, hi):
        dn = np.clip((d-lo)/max(hi-lo,1e-6),0,1); return (cm.turbo(dn)[...,:3]*255).astype(np.uint8)

    rows = []
    for st in sel:
        rgb = np.array(Image.open(f"{a.realgt}/{st}_rgb.jpg").convert("RGB").resize((RW,RH)))
        gt = np.load(f"{a.realgt}/{st}_depth.npy").astype(np.float32)
        m = (gt>0.2)&(gt<15); lo,hi = np.percentile(gt[m],2), np.percentile(gt[m],98)
        tiles = [rgb, turbo(gt,lo,hi)]
        titles = ["real", "GT depth"]
        for lbl, kind, ck in specs:
            if kind == "": continue
            p = predict(kind, ck, rgb)
            if a.align: p = p*(np.median(gt[m])/max(np.median(p[m]),1e-6))
            tiles.append(turbo(p,lo,hi)); titles.append(lbl)
        row = np.concatenate(tiles,1)
        rows.append((row, st, titles))
    W1 = RW*len(rows[0][2]); H1 = RH
    grid = Image.new("RGB",(W1, H1*len(rows)+24),(18,18,18)); d=ImageDraw.Draw(grid)
    for c,t in enumerate(rows[0][2]): d.text((c*RW+6,4),t,fill=(255,255,0))
    for r,(row,st,_) in enumerate(rows):
        grid.paste(Image.fromarray(row),(0,24+r*H1)); d.text((6,24+r*H1+4),st,fill=(0,255,255))
    grid.save(a.out,quality=88); print("wrote",a.out,grid.size)


if __name__ == "__main__":
    main()

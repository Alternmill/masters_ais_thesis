"""
From the clean front-camera real-GT, carve a held-out eval set (every 5th frame) and
build train sets with a disjoint 80/20 split:
    front_gt_heldout/   <stem>_rgb.jpg / _depth.npy   (eval target)
    ds_real_c/train/    rgb_####.jpg / depth_####.npy  (real train)
    ds_realaug_c/train/ (real train + N near-pose renders each)

Usage:
    python build_clean_datasets.py --scan <lcc> --refined poses_front_refined.json \
        --realgt front_gt --outroot <Q> --min-inl 40 --naug 4
"""
import argparse, json, os, math, numpy as np
from PIL import Image
from lcc import LccScan
from render import Renderer, Intrinsics

RW, RH = 640, 480


def rand_rot(rng, d):
    ax = rng.standard_normal(3); ax /= np.linalg.norm(ax)
    th = math.radians(rng.uniform(-d, d))
    K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+math.sin(th)*K+(1-math.cos(th))*K@K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True); ap.add_argument("--refined", required=True)
    ap.add_argument("--realgt", required=True); ap.add_argument("--outroot", required=True)
    ap.add_argument("--min-inl", type=int, default=40); ap.add_argument("--naug", type=int, default=4)
    ap.add_argument("--radius", type=float, default=0.10); ap.add_argument("--rot", type=float, default=8.0)
    ap.add_argument("--heldout-every", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    P = json.load(open(a.refined))
    intr = Intrinsics.from_dict(P["intrinsics"]).scaled(RW / P["intrinsics"]["w"]); intr.model = "fisheye"
    ref = sorted([f for f in P["frames"] if f.get("refined") and f.get("inl",0) >= a.min_inl
                  and os.path.exists(f"{a.realgt}/{f['name'].replace('.jpg','')}_rgb.jpg")],
                 key=lambda f: f["name"])
    held = set(range(0, len(ref), a.heldout_every))
    train = [f for i,f in enumerate(ref) if i not in held]
    heldout = [f for i,f in enumerate(ref) if i in held]
    print(f"[clean] {len(ref)} clean frames -> {len(train)} train / {len(heldout)} held-out")

    Qh = f"{a.outroot}/front_gt_heldout"; os.makedirs(Qh, exist_ok=True)
    for f in heldout:
        stem = f["name"].replace(".jpg","")
        for ext in ("_rgb.jpg","_depth.npy"):
            src = f"{a.realgt}/{stem}{ext}"
            if os.path.exists(src):
                open(f"{Qh}/{stem}{ext}","wb").write(open(src,"rb").read())

    scan = LccScan.open(a.scan); sp = scan.decode(0); sp.quat[:] = 0; sp.quat[:,0] = 1
    renderer = Renderer(sp); rng = np.random.default_rng(a.seed)

    def build(outdir, naug):
        os.makedirs(f"{outdir}/train", exist_ok=True); os.makedirs(f"{outdir}/val", exist_ok=True)
        i = 0
        for f in train:
            stem = f["name"].replace(".jpg","")
            rgb = np.array(Image.open(f"{a.realgt}/{stem}_rgb.jpg").convert("RGB").resize((RW,RH)))
            d = np.load(f"{a.realgt}/{stem}_depth.npy")
            Image.fromarray(rgb).save(f"{outdir}/train/rgb_{i:04d}.jpg", quality=92); np.save(f"{outdir}/train/depth_{i:04d}.npy", d); i += 1
            if naug:
                c2w = np.linalg.inv(np.array(f["w2c"]))
                for _ in range(naug):
                    c = c2w.copy(); c[:3,3] += np.clip(rng.normal(0,a.radius*0.6,3),-a.radius,a.radius); c[:3,:3] = c2w[:3,:3]@rand_rot(rng,a.rot)
                    r_rgb, r_d = renderer.render(np.linalg.inv(c), intr, depth=True)
                    Image.fromarray(r_rgb).save(f"{outdir}/train/rgb_{i:04d}.jpg", quality=92); np.save(f"{outdir}/train/depth_{i:04d}.npy", r_d); i += 1
        Image.fromarray(np.zeros((RH,RW,3),np.uint8)).save(f"{outdir}/val/rgb_0000.jpg"); np.save(f"{outdir}/val/depth_0000.npy", np.zeros((RH,RW),np.float32))
        print(f"[clean] {outdir}: {i} train images")
    build(f"{a.outroot}/ds_real_c", 0)
    build(f"{a.outroot}/ds_realaug_c", a.naug)


if __name__ == "__main__":
    main()

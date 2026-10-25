"""
Split PortalCam frames by camera. The .xbin stream interleaves a rectilinear FRONT
camera (operator-free) and a circular ~180deg FISHEYE camera (operator in frame); the
fisheye frames have a black vignette, so mean corner brightness separates them cleanly.
Only the front camera is a valid depth-GT target. Writes front.txt / fisheye.txt and a
contact sheet.

Usage:
    python classify_frames.py --img <img_dir> --out <dir> [--thresh 35]
"""
import argparse
import glob
import os
import numpy as np
from PIL import Image


def corner_brightness(path, frac=0.13):
    im = np.asarray(Image.open(path).convert("L").resize((400, 300)), float)
    H, W = im.shape
    k = int(min(H, W) * frac)
    corners = [im[:k, :k], im[:k, -k:], im[-k:, :k], im[-k:, -k:]]
    return float(np.mean([c.mean() for c in corners]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=35.0,
                    help="corner brightness below this = circular-fisheye/operator frame")
    ap.add_argument("--sheet", type=int, default=40, help="frames in the QC contact sheet")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frames = sorted(glob.glob(f"{args.img}/*.jpg"))
    b = np.array([corner_brightness(f) for f in frames])
    front = [os.path.basename(f) for f, v in zip(frames, b) if v >= args.thresh]
    fish = [os.path.basename(f) for f, v in zip(frames, b) if v < args.thresh]
    open(f"{args.out}/front.txt", "w").write("\n".join(front))
    open(f"{args.out}/fisheye.txt", "w").write("\n".join(fish))
    print(f"[classify] {len(frames)} frames -> front {len(front)} ({100*len(front)/len(frames):.0f}%) "
          f"| fisheye/operator {len(fish)} ({100*len(fish)/len(frames):.0f}%)")

    def sheet(names, path, cols=8, rows=3):
        tw, th = 200, 150
        img = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
        pick = names[:: max(1, len(names) // (cols * rows))][:cols * rows]
        for i, n in enumerate(pick):
            im = Image.open(f"{args.img}/{n}").convert("RGB").resize((tw, th))
            img.paste(im, ((i % cols) * tw, (i // cols) * th))
        img.save(path, quality=84)
    sheet(front, f"{args.out}/sheet_front.jpg")
    sheet(fish, f"{args.out}/sheet_fisheye.jpg")
    print(f"[classify] wrote front.txt, fisheye.txt, sheet_front.jpg, sheet_fisheye.jpg to {args.out}")


if __name__ == "__main__":
    main()

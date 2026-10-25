"""
Carve the embedded H.264 stream out of an XGRIDS `.xbin` and decode it to JPEG frames.

The .xbin interleaves an H.264 elementary stream; access units start at the SPS NAL
start-code (00 00 00 01 67). We concatenate them into a raw .h264 and let ffmpeg decode.

Usage:
    python extract_frames.py --xbin <scan>.xbin --out <dir> [--fps 4]
"""
import argparse
import os
import subprocess

SPS_START = b"\x00\x00\x00\x01\x67"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xbin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=0,
                    help="decode at this many frames/sec (0 = every decoded frame)")
    ap.add_argument("--scale", default="", help="decode straight to WxH (e.g. 1280x960); order-safe (fps_mode passthrough)")
    ap.add_argument("--drop-stream", action="store_true", help="delete the carved stream.h264 after decoding (disk)")
    args = ap.parse_args()
    os.makedirs(f"{args.out}/img", exist_ok=True)

    data = open(args.xbin, "rb").read()
    offs = []
    i = data.find(SPS_START)
    while i != -1:
        offs.append(i)
        i = data.find(SPS_START, i + 5)
    offs.append(len(data))
    print(f"[carve] {len(offs) - 1} SPS-led access units in {len(data) / 1e9:.2f} GB")

    stream = f"{args.out}/stream.h264"
    with open(stream, "wb") as f:
        for k in range(len(offs) - 1):
            f.write(data[offs[k]:offs[k + 1]])
    print(f"[carve] wrote {stream} ({os.path.getsize(stream) / 1e6:.0f} MB)")

    filters = ([f"fps={args.fps}"] if args.fps > 0 else []) + ([f"scale={args.scale.replace('x', ':')}"] if args.scale else [])
    vf = ["-vf", ",".join(filters)] if filters else []
    passthrough = ["-fps_mode", "passthrough"] if args.scale and args.fps <= 0 else []
    cmd = ["ffmpeg", "-y", "-fflags", "+genpts", "-i", stream, *passthrough, *vf,
           "-q:v", "2", "-start_number", "0", f"{args.out}/img/f%04d.jpg"]
    log = open(f"{args.out}/ffmpeg.log", "w")
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    n = len(os.listdir(f"{args.out}/img"))
    open(f"{args.out}/done.txt", "w").write(f"FRAMES={n}\n")
    if args.drop_stream:
        os.remove(stream)
    print(f"[carve] decoded {n} frames to {args.out}/img")


if __name__ == "__main__":
    main()

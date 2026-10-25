"""Save per-pixel predictions (float16, 480x640) of one model on the three room-state sets and a held-out set.

Usage:
    python stall_predict.py --kind da3|moge [--ckpt best.pt] --tag NAME [--heldout DIR]
    -> preds/<tag>__<set>.npz (keys: names, p)
"""
import argparse, glob, os
import numpy as np
import eval_v2 as V2
B = "scans"
SETS = {"A": f"{B}/evalsets_2026-09-27/aug19_bath1", "B": f"{B}/evalsets_2026-09-27/aug19_bath2",
        "P": f"{B}/evalsets_2026-09-27/aug19_bath3"}
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True); ap.add_argument("--ckpt", default="")
ap.add_argument("--tag", required=True); ap.add_argument("--heldout", default="")
a = ap.parse_args()
W = V2.build(a.kind, a.ckpt); pred = V2.make_predict(W)
sets = dict(SETS)
if a.heldout: sets["heldout"] = a.heldout
for k, d in sets.items():
    paths = sorted(glob.glob(f"{d}/*_rgb.jpg"))
    P = pred(None, None, paths)
    os.makedirs(f"{HERE}/preds", exist_ok=True)
    np.savez_compressed(f"{HERE}/preds/{a.tag}__{k}.npz", names=np.array([os.path.basename(p)[:-8] for p in paths]),
                        p=np.stack(P).astype(np.float16))
    print(a.tag, k, len(paths), flush=True)

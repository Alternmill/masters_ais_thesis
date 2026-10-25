#!/usr/bin/env bash
set -euo pipefail
XB="$1"; LCC="$2"; Q="$3"
PY="${PYTHON:-python}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
cd "$(dirname "$0")"
FILT='ARCH_LIST|warnings|INFO|Processed|Loading|xFormers|FFN|ffmpeg|^\[realgt\] [0-9]+/'

echo "### $Q : extract + classify + prep ###"
$PY extract_frames.py --xbin "$XB" --out "$Q" ${EXTRACT_ARGS:-} 2>&1 | grep -viE "$FILT"
$PY classify_frames.py --img "$Q/img" --out "$Q/classify" 2>&1 | grep -viE "$FILT"
$PY - "$Q" <<'PYEOF'
import sys,os,numpy as np
from PIL import Image
Q=sys.argv[1]; front=open(f"{Q}/classify/front.txt").read().split()
idx=sorted(set(int(i) for i in np.linspace(0,len(front)-1,min(450,len(front))).round().astype(int)))
os.makedirs(f"{Q}/colmap_front/images",exist_ok=True)
for i in idx: Image.open(f"{Q}/img/{front[i]}").convert("RGB").resize((1280,960),Image.LANCZOS).save(f"{Q}/colmap_front/images/{front[i]}",quality=95)
print("front COLMAP set:",len(idx),"of",len(front))
PYEOF

echo "### $Q : COLMAP ###"
bash run_colmap.sh "$Q/colmap_front" 2>&1 | grep -viE "$FILT" | tail -3
SPARSE=$($PY - "$Q/colmap_front/sparse" <<'PYEOF'
import sys,glob,pycolmap
best=None
for m in sorted(glob.glob(f"{sys.argv[1]}/*")):
    n=pycolmap.Reconstruction(m).num_reg_images()
    if best is None or n>best[1]: best=(m,n)
print(best[0])
PYEOF
)
echo "using $SPARSE"

echo "### $Q : register + sequential refine + GT ###"
$PY export_traj.py --sparse "$SPARSE" --scan "$LCC" --out "$Q/traj.json" 2>&1 | grep -viE "$FILT"
$PY register_anchor.py --scan "$LCC" --traj "$Q/traj.json" --frames "$Q/img" --out "$Q/poses.json" 2>&1 | grep -viE "$FILT"
$PY refine_poses.py --scan "$LCC" --poses "$Q/poses.json" --frames "$Q/img" --out "$Q/poses_refined.json" --sequential 2>&1 | grep -viE "$FILT"
$PY gen_realgt.py --scan "$LCC" --refined "$Q/poses_refined.json" --frames "$Q/img" --out "$Q/real_gt" --min-inl 40 --min-edge 85 2>&1 | grep -viE "$FILT"
echo "### $Q DONE: $(ls $Q/real_gt/*_rgb.jpg 2>/dev/null | wc -l) real-GT pairs ###"

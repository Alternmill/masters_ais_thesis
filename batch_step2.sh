#!/usr/bin/env bash
set -uo pipefail
PY="${PYTHON:-python}"
Q="${DATA_ROOT:?set DATA_ROOT to the morning-office dataset dir}"
EVAL="${EVAL:?set EVAL to the office real-GT dir}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
cd "$(dirname "$0")"

declare -A DATA=( [real]="$Q/ds_real" [realaug]="$Q/ds_realaug" [render]="${RENDER_DATA:-$Q/ds_render}" )

for MODEL in da2 da3; do
  STEPS=$([ "$MODEL" = da2 ] && echo 600 || echo 400)
  for DS in real realaug render; do
    OUT="$Q/ft_runs/${MODEL}_${DS}"
    echo "########## $MODEL on $DS (steps $STEPS) ##########"
    $PY finetune.py --model "$MODEL" --data "${DATA[$DS]}" --eval "$EVAL" \
        --out "$OUT" --steps "$STEPS" 2>&1 \
      | grep -viE 'INFO|Warning|Loading weights|cache|xFormers|FFN|Processed Images'
  done
done
echo "ALL_STEP2_DONE"

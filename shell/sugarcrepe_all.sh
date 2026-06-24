#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for cfg in \
  configs/sugarcrepe_cuhk_pedes.yaml \
  configs/sugarcrepe_icfg_pedes.yaml \
  configs/sugarcrepe_rstpreid.yaml
do
  echo "==> ${cfg}"
  dataset="$(basename "${cfg}" | sed -E 's/^sugarcrepe_(.*)\.yaml$/\1/')"
  export SUGARCREPE_CONFIG="${cfg}"
  export RETRIEVAL_CONFIG="configs/text_to_image_retrieval_${dataset}.yaml"
  uv run python text-to-image-retrieval.py
  uv run python sugarcrepe-pedes.py
done

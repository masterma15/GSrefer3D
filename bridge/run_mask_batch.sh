#!/usr/bin/env bash
# Batch mask (Grounding DINO + SAM2) for all data2 objects. Run in WSL roborefer env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# GroundingDINO: pip install addict timm yapf supervision pycocotools
# (pip install -e GroundingDINO may fail in roborefer; PYTHONPATH is enough)
export PYTHONPATH="${ROOT}/GroundingDINO:${PYTHONPATH:-}"

# --- adjust paths if needed ---
SAM2_CKPT="${SAM2_CKPT:-$ROOT/weights/sam2.1_hiera_large.pt}"
SAM2_CFG="${SAM2_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
BOX_THR="${BOX_THR:-0.10}"
TEXT_THR="${TEXT_THR:-0.12}"
MAX_AREA="${MAX_AREA:-0.35}"
ANCHOR_R="${ANCHOR_R:-64}"
GDINO_CFG="${GDINO_CFG:-$ROOT/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GDINO_CKPT="${GDINO_CKPT:-$ROOT/weights/groundingdino_swint_ogc.pth}"

for f in "$SAM2_CKPT" "$GDINO_CKPT"; do
  if [[ ! -f "$f" ]]; then
    echo "[error] missing weight: $f"
    exit 1
  fi
done
if [[ ! -f "$GDINO_CFG" ]]; then
  echo "[error] missing GroundingDINO config: $GDINO_CFG"
  echo "  clone: git clone https://github.com/IDEA-Research/GroundingDINO $ROOT/GroundingDINO"
  exit 1
fi

# Fix Windows rgb_path in projections_kept.json
python3 - <<'PY'
import json
from pathlib import Path

for p in sorted(Path("training_data").glob("data2_*/projections_kept.json")):
    recs = json.loads(p.read_text(encoding="utf-8"))
    changed = False
    for r in recs:
        rp = r.get("rgb_path", "")
        if len(rp) >= 3 and rp[:2].upper() == "E:" and rp[2] in ("\\", "/"):
            r["rgb_path"] = "/mnt/e/" + rp[3:].replace("\\", "/")
            changed = True
    if changed:
        p.write_text(json.dumps(recs, indent=2), encoding="utf-8")
        print("patched", p)
PY

run_one() {
  local out="$1" prompt="$2" obj="$3"
  echo "========== $out =========="
  python bridge/gen_training_data.py --stage mask \
    --mask-mode grounding \
    --out "$ROOT/training_data/$out" \
    --prompt "$prompt" \
    --object "$obj" \
    --sam2-checkpoint "$SAM2_CKPT" \
    --sam2-config "$SAM2_CFG" \
    --grounding-config "$GDINO_CFG" \
    --grounding-checkpoint "$GDINO_CKPT" \
    --grounding-box-threshold "$BOX_THR" \
    --grounding-text-threshold "$TEXT_THR" \
    --anchor-box-radius "$ANCHOR_R" \
    --max-box-area-ratio "$MAX_AREA"
}

run_one data2_bowl "Please point to the gold-colored bowl on the desk." "bowl"
run_one data2_bracelet "Please point to the white beaded bracelet on the desk." "bracelet"
run_one data2_cookie "Please point to the green square cookie package on the desk." "cookie package"
run_one data2_golden_retriever "Please point to the light yellow plush golden retriever toy." "plush golden retriever"
run_one data2_hair_clip "Please point to the purple hair clip on the desk." "hair clip"
run_one data2_medicine_bottle "Please point to the medicine bottle on the desk." "medicine bottle"
run_one data2_rabbit "Please point to the brown plush rabbit." "plush rabbit"
run_one data2_shaver "Please point to the electric shaver on the desk." "electric shaver"
run_one data2_toy_cake "Please point to the toy cake held by the brown plush rabbit." "small decorative toy cake"
run_one data2_umbrella "Please point to the black and red umbrella on the desk." "umbrella"

echo "[done] all mask stages finished"

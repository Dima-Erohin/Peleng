#!/usr/bin/env bash
# Обучение детектора позы грузовика на GPU (Linux, NVIDIA, CUDA) одной командой.
# Пример (из корня репозитория):
#   bash offline/detectors/run_truck_train_gpu.sh --data /data/project2 --out /work/truck_train
# Параметры (все необязательные, кроме --data и --out):
#   --model yolo11m-pose.pt  --imgsz 1280  --epochs 100  --batch 16  --stride 1  --device 0  --name All_m1280
# Повторный запуск той же командой продолжает с последней эпохи и не пересобирает датасет.
set -euo pipefail

DATA=""; OUT=""; MODEL="yolo11m-pose.pt"; IMGSZ=1280; EPOCHS=100; BATCH=16; STRIDE=1; DEVICE=0; NAME=""; WORKERS=8
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA="$2"; shift 2;;   --out) OUT="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;; --imgsz) IMGSZ="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;; --batch) BATCH="$2"; shift 2;;
    --stride) STRIDE="$2"; shift 2;; --device) DEVICE="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;   --workers) WORKERS="$2"; shift 2;;
    *) echo "неизвестный параметр: $1"; exit 1;;
  esac
done
[[ -n "$DATA" && -n "$OUT" ]] || { echo "нужны --data <папка с mp4 и csv> и --out <рабочая папка>"; exit 1; }
[[ -n "$NAME" ]] || NAME="All_$(basename "$MODEL" .pt)_$IMGSZ"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
PYBIN="${PYTHON:-python3}"
VENV="$OUT/venv"; PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "== создаю venv и ставлю torch с CUDA =="
  "$PYBIN" -m venv "$VENV"
  "$PY" -m pip install --upgrade pip
  # колёса torch для CUDA 12.4 (подходят к драйверам NVIDIA >= 550; на 4090 ок)
  "$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  "$PY" -m pip install -r "$HERE/requirements_truck.txt"
fi
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA не видна: проверьте nvidia-smi и драйвер'; print('GPU:', torch.cuda.get_device_name(0))"

DS="$OUT/truck_ds_s$STRIDE"
if [[ ! -f "$DS/data_All.yaml" ]]; then
  echo "== генерирую датасет (шаг по кадрам $STRIDE) =="
  "$PY" -I "$HERE/truck_dataset.py" --data "$DATA" --out "$DS" --stride "$STRIDE"
fi

echo "== обучение $MODEL, imgsz $IMGSZ, $EPOCHS эпох, batch $BATCH, device $DEVICE =="
"$PY" -I "$HERE/train_truck_pose.py" --data "$DS/data_All.yaml" --name "$NAME" --weights "$MODEL" \
  --device "$DEVICE" --workers "$WORKERS" --batch "$BATCH" --imgsz "$IMGSZ" --epochs "$EPOCHS" --out "$OUT/runs"

BEST="$OUT/runs/$NAME/weights/best.pt"
if [[ -f "$BEST" ]]; then
  mkdir -p "$OUT/RESULT"; cp "$BEST" "$OUT/RESULT/truck_pose_$NAME.pt"
  echo "ГОТОВО: веса -> $OUT/RESULT/truck_pose_$NAME.pt (пришлите этот файл)"
fi

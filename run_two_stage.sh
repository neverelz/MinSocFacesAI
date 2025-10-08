#!/usr/bin/env bash
set -euo pipefail

# ========= ПОД НАСТРОЙКУ =========
PYTHON_BIN="python3"
PROJECT_ROOT="/home/user/Documents/PycharmProjects/MinSocFacesAI"
VENV_DIR="/home/user/Documents/PycharmProjects/MinSocFacesAI/detectron310"
USE_EXISTING_VENV="yes"   # yes = активировать существующее venv

# Датасет (пилотный сейчас; подменишь путями на полный потом)
TRAIN_JSON="/home/user/mrgv2/pilot_per_class30_os/subset_train.json"
TRAIN_IMG="/home/user/mrgv2/pilot_per_class30_os"
VAL_JSON="/home/user/mrgv2/pilot_per_class30_os/subset_val.json"
VAL_IMG="/home/user/mrgv2/pilot_per_class30_os"
TEST_JSON="/home/user/mrgv2/pilot_per_class30_os/subset_val.json"
TEST_IMG="/home/user/mrgv2/pilot_per_class30_os"

NUM_CLASSES=47
DEVICE="cpu"
WORKERS=0
REPEAT_THRESHOLD=0.01
MIN_SIZE_TEST=640
BASE_LR=5e-4

# Stage-1: RetinaNet-R18 (быстрая прогревка)
OUT1="/home/user/mrgv2/pilot_results_retinanet_r18"
IMS_PER_BATCH1=2
MAX_ITER1=3000
EVAL_PERIOD1=500
RESNET_DEPTH1=18
FREEZE_BACKBONE_ITERS1=1500

# Stage-2: FCOS-R34 (дообучение; при наличии — с весов Stage-1)
OUT2="/home/user/mrgv2/pilot_results_fcos_r34"
IMS_PER_BATCH2=2
MAX_ITER2=4000
EVAL_PERIOD2=500
RESNET_DEPTH2=34
FREEZE_BACKBONE_ITERS2=800

# ========= УТИЛИТЫ =========
msg(){ echo -e "\033[1;36m[$(date +%H:%M:%S)]\033[0m $*"; }
ensure_dir(){ mkdir -p "$1"; }

activate_venv() {
  if [[ "$USE_EXISTING_VENV" == "yes" ]]; then
    [[ -f "$VENV_DIR/bin/activate" ]] || { echo "VENV_DIR не найден: $VENV_DIR"; exit 2; }
  else
    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
      msg "Создаю venv: $VENV_DIR"
      "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  python -V; pip -V
}

export_runtime_env() {
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OMP_WAIT_POLICY=PASSIVE
  export KMP_AFFINITY=disabled
  export PYTORCH_SHOW_CPP_STACKTRACES=1
  export OPENCV_LOG_LEVEL=SILENT
}

pick_stage1_weights() {
  # Возвращает путь к лучшему доступному чекпоинту Stage-1 (stdout)
  if [[ -f "$OUT1/model_final_safe.pth" ]]; then
    echo "$OUT1/model_final_safe.pth"
  elif [[ -f "$OUT1/model_final.pth" ]]; then
    echo "$OUT1/model_final.pth"
  else
    ls -1t "$OUT1"/model_*.pth 2>/dev/null | head -n1 || true
  fi
}

# ========= MAIN =========
activate_venv
export_runtime_env

# ---------- Stage-1: RetinaNet-R18 ----------
ensure_dir "$OUT1"
msg "Stage-1: RetinaNet-R${RESNET_DEPTH1}-FPN → $OUT1"
set -x
"$PYTHON_BIN" "$PROJECT_ROOT/R-CNN/FCOS/train_fcos.py" \
  --train-json "$TRAIN_JSON" \
  --train-img  "$TRAIN_IMG" \
  --val-json   "$VAL_JSON" \
  --val-img    "$VAL_IMG" \
  --test-json  "$TEST_JSON" \
  --test-img   "$TEST_IMG" \
  --outdir     "$OUT1" \
  --num-classes "$NUM_CLASSES" \
  --device "$DEVICE" \
  --ims-per-batch "$IMS_PER_BATCH1" \
  --workers "$WORKERS" \
  --base-lr "$BASE_LR" \
  --max-iter "$MAX_ITER1" \
  --eval-period "$EVAL_PERIOD1" \
  --repeat-threshold "$REPEAT_THRESHOLD" \
  --min-size-test "$MIN_SIZE_TEST" \
  --resnet-depth "$RESNET_DEPTH1" \
  --freeze-backbone-iters "$FREEZE_BACKBONE_ITERS1" \
  2>&1 | tee "$OUT1/run.log"
set +x

# Выберем чекпоинт
WEIGHTS_STAGE1="$(pick_stage1_weights || true)"
if [[ -n "${WEIGHTS_STAGE1:-}" ]]; then
  msg "Найден чекпоинт Stage-1: $WEIGHTS_STAGE1"
else
  msg "Внимание: чекпоинт Stage-1 не найден. Stage-2 стартует без --weights (инициализация из torchvision)."
fi

# ---------- Stage-2: FCOS-R34 ----------
ensure_dir "$OUT2"
msg "Stage-2: FCOS-R${RESNET_DEPTH2}-FPN → $OUT2"
set -x
"$PYTHON_BIN" "$PROJECT_ROOT/R-CNN/FCOS/train_fcos.py" \
  --train-json "$TRAIN_JSON" \
  --train-img  "$TRAIN_IMG" \
  --val-json   "$VAL_JSON" \
  --val-img    "$VAL_IMG" \
  --test-json  "$TEST_JSON" \
  --test-img   "$TEST_IMG" \
  --outdir     "$OUT2" \
  --num-classes "$NUM_CLASSES" \
  --device "$DEVICE" \
  --ims-per-batch "$IMS_PER_BATCH2" \
  --workers "$WORKERS" \
  --base-lr "$BASE_LR" \
  --max-iter "$MAX_ITER2" \
  --eval-period "$EVAL_PERIOD2" \
  --repeat-threshold "$REPEAT_THRESHOLD" \
  --min-size-test "$MIN_SIZE_TEST" \
  --prefer-fcos \
  --resnet-depth "$RESNET_DEPTH2" \
  --freeze-backbone-iters "$FREEZE_BACKBONE_ITERS2" \
  ${WEIGHTS_STAGE1:+--weights "$WEIGHTS_STAGE1"} \
  2>&1 | tee "$OUT2/run.log"
set +x

msg "Готово. Логи: $OUT1/run.log и $OUT2/run.log"

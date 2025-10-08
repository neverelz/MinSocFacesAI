#!/usr/bin/env bash
set -euo pipefail

# ========= ПАРАМЕТРЫ (ПОПРАВЬ ПОД СЕБЯ) =========
PYTHON_BIN="python3"                                    # Можно оставить python3
PROJECT_ROOT="/home/user/Documents/PycharmProjects/MinSocFacesAI"   # Корень проекта
VENV_DIR="/home/user/Documents/PycharmProjects/MinSocFacesAI/detectron310"  # Путь к venv
USE_EXISTING_VENV="yes"                                 # "yes" — активировать существующее venv

# Датасет (мини/пилотный)
TRAIN_JSON="/home/user/mrgv2/pilot_per_class30_os/subset_train.json"
TRAIN_IMG="/home/user/mrgv2/pilot_per_class30_os"
VAL_JSON="/home/user/mrgv2/pilot_per_class30_os/subset_val.json"
VAL_IMG="/home/user/mrgv2/pilot_per_class30_os"
TEST_JSON="/home/user/mrgv2/pilot_per_class30_os/subset_val.json"
TEST_IMG="/home/user/mrgv2/pilot_per_class30_os"

# Вывод
OUTDIR="/home/user/mrgv2/pilot_results"

# Гиперпараметры (минимально безопасные для CPU)
NUM_CLASSES=47
DEVICE="cpu"
IMS_PER_BATCH=2
WORKERS=0
BASE_LR=5e-4
MAX_ITER=6000
EVAL_PERIOD=500
REPEAT_THRESHOLD=0.01
MIN_SIZE_TEST=640
PREFER_FCOS="--prefer-fcos"   # Для FCOS. Для чистой RetinaNet — убери этот флаг.
LOGFILE="$OUTDIR/run.log"

# ========= ФУНКЦИИ =========
function msg() { echo -e "\033[1;36m[$(date +%H:%M:%S)]\033[0m $*"; }
function ensure_dir() { mkdir -p "$1"; }

function activate_venv() {
  if [[ "$USE_EXISTING_VENV" == "yes" ]]; then
    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
      echo "VENV_DIR не найден: $VENV_DIR"; exit 2
    fi
  else
    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
      msg "Создаю виртуальное окружение: $VENV_DIR"
      "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  python -V
  pip -V
}

function ensure_pkgs() {
  msg "Устанавливаю/проверяю базовые зависимости…"
  pip install --upgrade pip wheel setuptools
  pip install "yacs>=0.1.8" "pyyaml" "tqdm" "opencv-python-headless" "numpy" "pillow>=9.1"

  python - <<'PY'
import importlib
try:
    import torch, torchvision
    print("[check] torch:", torch.__version__, "torchvision:", torchvision.__version__)
except Exception as e:
    print("[install] torch cpu", e)
    import os
    os.system("pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision")
PY

  python - <<'PY'
import importlib
def ok(m):
    try: importlib.import_module(m); return True
    except: return False
print("[check] detectron2:", ok("detectron2"))
print("[check] AdelaiDet:", ok("adet"))
PY

  msg "Если detectron2/AdelaiDet не найдены — проверь их установку в текущем окружении."
}

function export_runtime_env() {
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OMP_WAIT_POLICY=PASSIVE
  export KMP_AFFINITY=disabled
  export PYTORCH_SHOW_CPP_STACKTRACES=1
  export OPENCV_LOG_LEVEL=SILENT
  msg "Экспортировал переменные окружения (устойчивость и стабильные логи)."
}

# ========= MAIN =========
ensure_dir "$OUTDIR"
activate_venv
ensure_pkgs
export_runtime_env

msg "Запускаю пайплайн (текущая конфигурация train_fcos.py)…"
set -x
python "$PROJECT_ROOT/R-CNN/FCOS/train_fcos.py" \
  --train-json "$TRAIN_JSON" \
  --train-img  "$TRAIN_IMG" \
  --val-json   "$VAL_JSON" \
  --val-img    "$VAL_IMG" \
  --test-json  "$TEST_JSON" \
  --test-img   "$TEST_IMG" \
  --outdir     "$OUTDIR" \
  --num-classes $NUM_CLASSES \
  --device $DEVICE \
  --ims-per-batch $IMS_PER_BATCH \
  --workers $WORKERS \
  --base-lr $BASE_LR \
  --max-iter $MAX_ITER \
  --eval-period $EVAL_PERIOD \
  --repeat-threshold $REPEAT_THRESHOLD \
  --min-size-test $MIN_SIZE_TEST \
  $PREFER_FCOS 2>&1 | tee "$LOGFILE"
set +x

msg "Готово. Логи: $LOGFILE"

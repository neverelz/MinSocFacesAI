Установка торча и детектрона(рабочая последовательность действий)

1) python3.10 -m venv torch112

2) source torch112/bin/activate

3) pip install --upgrade pip

4) python -V

5) sudo dnf install -y gcc gcc-c++ make ninja-build python3.10-devel \
  libjpeg-turbo-devel libpng-devel mesa-libGL || true

6) pip install -r requirements.txt

            ИЛИ

6.1) pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  "torch==1.12.0+cpu" "torchvision==0.13.0+cpu" "torchaudio==0.12.0+cpu" --no-cache-dir

6.2) pip install \
  numpy==1.23.5 \
  opencv-python==4.9.0.80 \
  pycocotools==2.0.6 \
  pillow==10.4.0 \
  tqdm==4.66.4 \
  matplotlib==3.8.4 \
  pyyaml==6.0.1 \
  tabulate==0.9.0 \
  psutil==5.9.8

# pip list 

7) pip uninstall -y pybind11 || true

8) pip install "pybind11==2.10.4" --no-cache-dir  

9) pip install "fvcore<0.1.6" "iopath<0.1.10" "omegaconf<2.3.1" "yacs>=0.1.8" \
  "tabulate>=0.8" "termcolor>=1.1" "cloudpickle>=1.2" "tensorboard>=2.10" \
  "pydot" --no-cache-dir

10) export CC=$(command -v gcc)
export CXX=$(command -v g++)
export FORCE_CUDA=0
export MAX_JOBS=$(nproc)

11) PYT_PYBIND="$(python - <<'PY'
import site, sys, pathlib
for p in site.getsitepackages() + [site.getusersitepackages()]:
    p = pathlib.Path(p) / "torch/include/pybind11"
    if p.exists():
        print(p)
        sys.exit(0)
print("")
PY
)"

12) echo "PYTORCH pybind11 dir: $PYT_PYBIND"

13) cp "$PYT_PYBIND/attr.h" "$PYT_PYBIND/attr.h.bak"
grep -q "<cstdint>" "$PYT_PYBIND/attr.h" || \
  sed -i '1i #include <cstdint>' "$PYT_PYBIND/attr.h"

14) pip uninstall -y detectron2 || true
rm -rf /tmp/pip-req-build-* ~/.cache/pip

15)  export MAX_JOBS=$(nproc)                                                                                                                                                                  
pip install --no-build-isolation --no-cache-dir \
  "git+https://github.com/facebookresearch/detectron2.git@v0.6"

16) python - << 'PY'
import torch, detectron2
print("torch:", torch.__version__)
print("detectron2:", detectron2.__version__)
PY

(Компилятор берёт pybind11 из PyTorch
.../torch/include/pybind11/attr.h, а наш флаг -include cstdint так и не попал в команду сборки (его нет в длинной строке g++). Поэтому и сыпется std::uint16_t.
Самый быстрый и надёжный фикс здесь — вколоть <cstdint> прямо в копию pybind11 внутри PyTorch. Это локальная правка в вашем venv и ничего системного не ломает.)

(Почему код работает

У PyTorch 1.12 своя копия pybind11. В GCC 13 из-за порядка инклюдов бывает, что attr.h используется без явного <cstdint>, и std::uint16_t «исчезает». Добавление одной строки в attr.h устраняет проблему на месте, и остальные ошибки про nargs_* тоже уходят (это каскад от первой).)
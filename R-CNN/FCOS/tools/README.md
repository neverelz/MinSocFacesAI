Запуск train_fcos.py
mkdir -p /home/user/mrgv2/exp_fcos_tiny50

PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
KMP_AFFINITY=disabled PYTORCH_SHOW_CPP_STACKTRACES=1 \
python -u R-CNN/FCOS/train_fcos.py \
  --train-json /home/user/mrgv2/tiny50/tiny_train.json \
  --train-img  /home/user/mrgv2/tiny50 \
  --val-json   /home/user/mrgv2/tiny50/tiny_val.json \
  --val-img    /home/user/mrgv2/tiny50 \
  --test-json  /home/user/mrgv2/tiny50/tiny_val.json \
  --test-img   /home/user/mrgv2/tiny50 \
  --outdir     /home/user/mrgv2/exp_fcos_tiny50 \
  --num-classes 47 \
  --device cpu \
  --ims-per-batch 2 \
  --workers 0 \
  --base-lr 5e-4 \
  --max-iter 50 \
  --eval-period 100 \
  --repeat-threshold 0.01 \
  --min-size-test 640 \
  --prefer-fcos \
  2>&1 | tee /home/user/mrgv2/exp_fcos_tiny50/run.log
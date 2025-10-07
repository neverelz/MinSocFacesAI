#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_fcos.py
Detectron2: FCOS R50-FPN, online augmentations, RepeatFactorSampler, multi-scale,
robust training with progress/ETA, periodic evaluation, checkpoints,
and runtime P95/FPS measurement.

Usage (пример):
  python R-CNN/FCOS/train_fcos.py\
    --train-json /home/user/mrgv2/train/annotations.coco.json --train-img /home/user/mrgv2/train/images \
    --val-json   /home/user/mrgv2/val/annotations.coco.json   --val-img   /home/user/mrgv2/val/images \
    --test-json  /home/user/mrgv2/test/annotations.coco.json  --test-img  /home/user/mrgv2/test/images \
    --outdir     /home/user/mrgv2/exp_fcos_mvp \
    --num-classes 47 \
    --device cpu \
    --ims-per-batch 8 --max-iter 2000 --eval-period 500

Если CUDA доступна, можно опустить --device.
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Any

# --- Pillow shim: фикс для Pillow>=10 (нет Image.LINEAR) ---------------------
try:
    from PIL import Image as _PIL_Image
    if not hasattr(_PIL_Image, "LINEAR"):
        # В новых Pillow используют enum Resampling.*; берём эквивалент
        if hasattr(_PIL_Image, "BILINEAR"):
            _PIL_Image.LINEAR = _PIL_Image.BILINEAR
        else:
            from PIL.Image import Resampling
            _PIL_Image.LINEAR = Resampling.BILINEAR
except Exception:
    # Не мешаем запуску, если Pillow вдруг не установлен (его подтянет detectron2)
    pass
# ---------------------------------------------------------------------------

import numpy as np
import torch
import cv2

# Detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, HookBase, default_setup
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from detectron2.solver.build import get_default_optimizer_params

# ---------- utils ----------

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_categories_from_coco(json_path: str) -> List[str]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    cats = data.get("categories", []) or []
    # детерминированно по id
    cats_sorted = sorted(cats, key=lambda c: int(c.get("id", 0)))
    return [c.get("name", str(c.get("id"))) for c in cats_sorted]

def build_augs():
    # Лёгкие и быстрые онлайн-аугментации; не убиваем мелочь
    return [
        T.RandomFlip(horizontal=True, vertical=False),
        T.RandomRotation(angle=[-10, 10], sample_style="range", expand=False),
        T.RandomBrightness(0.9, 1.1),
        T.RandomContrast(0.9, 1.1),
        T.RandomSaturation(0.9, 1.1),
        T.RandomExtent(scale_range=(0.9, 1.1), shift_range=(0.05, 0.05)),
    ]

def mapper_with_augs(dataset_dict):
    # Custom mapper с аугментациями
    dataset_dict = dataset_dict.copy()
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    augs = build_augs()
    aug_input = T.AugInput(image)
    tfms = T.AugmentationList(augs)(aug_input)
    image = aug_input.image
    annos = [
        utils.transform_instance_annotations(obj, tfms, image.shape[:2])
        for obj in dataset_dict.get("annotations", [])
    ]
    return {
        "image": torch.as_tensor(image.transpose(2, 0, 1).copy()).float(),
        "instances": utils.annotations_to_instances(annos, image.shape[:2])
    }

# ---------- hooks ----------

class ETAHook(HookBase):
    """Печатает прогресс, ETA и расход времени; пишет в файл и консоль."""
    def __init__(self, total_iter: int, log_path: Path, every: int = 50):
        self.total_iter = total_iter
        self.every = every
        self.log_path = log_path
        safe_mkdir(log_path.parent)

    def after_step(self):
        iter_num = self.trainer.iter + 1
        if iter_num % self.every != 0:
            return
        elapsed = time.time() - self.trainer.start_time
        frac = min(1.0, iter_num / max(1, self.total_iter))
        eta_sec = elapsed / max(1e-9, frac) - elapsed
        # total_loss может отсутствовать в самом начале — подстрахуемся
        try:
            loss_val = float(self.trainer.storage.history('total_loss').latest())
        except Exception:
            loss_val = float('nan')
        msg = (f"[Iter {iter_num}/{self.total_iter}] "
               f"elapsed={timedelta(seconds=int(elapsed))} "
               f"eta={timedelta(seconds=int(max(0, eta_sec)))} "
               f"lr={self.trainer.optimizer.param_groups[0]['lr']:.6f} "
               f"loss={loss_val:.4f}")
        print(msg)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

class SafeCheckpointHook(HookBase):
    """Если что-то падает в обучении — аккуратно сохранит чекпоинт."""
    def __init__(self, outdir: Path):
        self.outdir = outdir

    def after_step(self):
        pass

    def after_train(self):
        try:
            self.trainer.checkpointer.save("model_final_safe")
        except Exception:
            print("[WARN] Failed to save final safe checkpoint:", traceback.format_exc())

# ---------- trainer ----------

class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # RepeatFactorTrainingSampler активируется через cfg.DATALOADER.SAMPLER_TRAIN
        return build_detection_train_loader(cfg, mapper=mapper_with_augs)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        safe_mkdir(Path(output_folder))
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

# ---------- latency / FPS test ----------

@torch.no_grad()
def measure_latency_fps(predictor, image_paths: List[str], warmup: int = 20, runs: int = 200) -> Tuple[float, float]:
    """Возвращает (P95_ms, FPS_mean)."""
    if len(image_paths) == 0:
        return (0.0, 0.0)
    # подрежем список для прогонов
    paths = (image_paths * ((warmup + runs) // max(1, len(image_paths)) + 1))[: warmup + runs]
    times = []
    for i, p in enumerate(paths):
        im = cv2.imread(p)
        if im is None:
            continue
        t0 = time.perf_counter()
        _ = predictor(im)  # DefaultPredictor(cfg) API совместим
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    if not times:
        return (0.0, 0.0)
    lat_ms = np.array(times) * 1000.0
    p95 = float(np.percentile(lat_ms, 95))
    fps = float(1.0 / np.mean(times))
    return (p95, fps)

# ---------- registration ----------

def register_datasets(train_json, train_img, val_json, val_img, test_json, test_img):
    register_coco_instances("psy_train", {}, train_json, train_img)
    register_coco_instances("psy_val",   {}, val_json,   val_img)
    register_coco_instances("psy_test",  {}, test_json,  test_img)

    # Синхронизация списков классов: берём из train
    train_classes = read_categories_from_coco(train_json)
    MetadataCatalog.get("psy_val").thing_classes  = train_classes
    MetadataCatalog.get("psy_test").thing_classes = train_classes
    return train_classes

# ---------- config builder ----------

def build_cfg(args, num_classes: int) -> Any:
    cfg = get_cfg()

    # --- Пытаемся взять FCOS YAML из Model Zoo (если он есть в этой версии) ---
    try:
        fcos_yaml = "COCO-Detection/fcos_R_50_FPN_1x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(fcos_yaml))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(fcos_yaml)
    except Exception:
        # --- Fallback: собираем FCOS вручную на базе FPN ---
        # Базовый FPN (тот точно есть) + включаем meta-архитектуру FCOS.
        cfg.merge_from_file(model_zoo.get_config_file("Base-RCNN-FPN.yaml"))

        # Ключевая строка: переключаем meta-архитектуру на FCOS
        cfg.MODEL.META_ARCHITECTURE = "FCOS"

        # Спинка: ResNet-50 + FPN
        cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
        cfg.MODEL.RESNETS.DEPTH = 50
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]

        # Инициализация: можно оставить пусто (обучение с нуля),
        # или задать ImageNet-инициализацию для ResNet-50 если она у тебя есть.
        cfg.MODEL.WEIGHTS = ""  # train from scratch

        # Базовые гиперпараметры FCOS (есть дефолты в detectron2, но зададим явные)
        cfg.MODEL.FCOS.NUM_CLASSES = num_classes
        cfg.MODEL.FCOS.INFERENCE_TH = args.infer_th
        cfg.MODEL.FCOS.NMS_TH = 0.6
        cfg.MODEL.FCOS.PRIOR_PROB = 0.01
        cfg.MODEL.FCOS.SIZES_OF_INTEREST = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
        cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
        cfg.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
        cfg.MODEL.FCOS.CENTER_SAMPLE = True
        cfg.MODEL.FCOS.POS_RADIUS = 1.5
        cfg.MODEL.FCOS.YIELD_PROPOSAL = False
        cfg.MODEL.FCOS.NORM = "GN"  # GroupNorm устойчивее на CPU/малых батчах

    # Датасеты
    cfg.DATASETS.TRAIN = ("psy_train",)
    cfg.DATASETS.TEST  = ("psy_val",)
    cfg.DATALOADER.NUM_WORKERS = max(2, args.workers)

    # RepeatFactorSampler для дисбаланса
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = args.repeat_threshold

    # Инпуты/масштабы
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 720, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST  = args.min_size_test
    cfg.INPUT.MAX_SIZE_TEST  = max(args.min_size_test, 1280)

    # Пороги
    cfg.MODEL.FCOS.INFERENCE_TH = args.infer_th
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # для совместимости
    cfg.MODEL.FCOS.NUM_CLASSES = num_classes

    # Солвер/тренинг
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.WARMUP_ITERS = min(1000, max(200, args.max_iter // 20))
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    # CPU/GPU
    if args.device:
        cfg.MODEL.DEVICE = args.device

    cfg.OUTPUT_DIR = args.outdir
    safe_mkdir(Path(cfg.OUTPUT_DIR))
    return cfg

# ---------- tiny predictor for FPS ----------

class DefaultPredictorLite:
    """Лёгкая обёртка вокруг модели Detectron2 (без сторонних зависимостей)."""
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = Trainer.build_model(self.cfg)
        self.model.eval()
        # загрузим лучшие веса, если есть, иначе — из cfg.MODEL.WEIGHTS
        from detectron2.checkpoint import DetectionCheckpointer
        DetectionCheckpointer(self.model).load(self.cfg.MODEL.WEIGHTS)

        # преобразование картинки
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        self.device = torch.device(self.cfg.MODEL.DEVICE)
        self.model.to(self.device)

    @torch.no_grad()
    def __call__(self, original_image: np.ndarray):
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        outputs = self.model([inputs])[0]
        return outputs

# ---------- main ----------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--train-img", required=True)
    ap.add_argument("--val-json",   required=True)
    ap.add_argument("--val-img",    required=True)
    ap.add_argument("--test-json",  required=True)
    ap.add_argument("--test-img",   required=True)
    ap.add_argument("--outdir",     required=True)
    ap.add_argument("--num-classes", type=int, default=None, help="Если задано — переопределит число классов (напр., 47)")
    ap.add_argument("--device", choices=["cpu","cuda"], default=None)
    ap.add_argument("--ims-per-batch", type=int, default=8)
    ap.add_argument("--base-lr", type=float, default=5e-4)
    ap.add_argument("--max-iter", type=int, default=12000)
    ap.add_argument("--eval-period", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repeat-threshold", type=float, default=0.002)
    ap.add_argument("--min-size-test", type=int, default=720)
    ap.add_argument("--infer-th", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def sanity_paths(args):
    for p in [args.train_json, args.val_json, args.test_json]:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Missing file: {p}")
    for p in [args.train_img, args.val_img, args.test_img]:
        if not Path(p).is_dir():
            raise FileNotFoundError(f"Missing dir: {p}")

def list_some_images(img_dir: str, k: int = 20) -> List[str]:
    paths = []
    for root, _, files in os.walk(img_dir):
        for nm in files:
            ext = os.path.splitext(nm)[1].lower()
            if ext in [".jpg",".jpeg",".png",".bmp",".webp"]:
                paths.append(os.path.join(root, nm))
        if len(paths) >= k:
            break
    return paths[:k]

def main():
    args = parse_args()
    safe_mkdir(Path(args.outdir))
    log_file = Path(args.outdir) / "train_log.txt"
    sanity_paths(args)

    # Регистрация датасетов
    classes = register_datasets(args.train_json, args.train_img, args.val_json, args.val_img, args.test_json, args.test_img)
    num_classes = args.num_classes if args.num_classes is not None else len(classes)
    if args.num_classes is not None and args.num_classes != len(classes):
        print(f"[INFO] Overriding num_classes from {len(classes)} -> {args.num_classes}")
    print(f"[INFO] NUM_CLASSES = {num_classes}")

    # Конфиг
    cfg = build_cfg(args, num_classes)

    # Логи Detectron2
    setup_logger(output=str(Path(args.outdir)))
    default_setup(cfg, {})

    # Тренер
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)

    # Хуки: ETA + safe checkpoint
    trainer.start_time = time.time()
    trainer.register_hooks([
        ETAHook(total_iter=cfg.SOLVER.MAX_ITER, log_path=log_file, every=50),
        SafeCheckpointHook(outdir=Path(cfg.OUTPUT_DIR)),
    ])

    # Обучение с перехватом ошибок
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving last checkpoint...")
        trainer.checkpointer.save("model_interrupted")
        sys.exit(130)
    except Exception as e:
        print("[FATAL] Training crashed. Saving last checkpoint…")
        with (Path(cfg.OUTPUT_DIR) / "crash_trace.txt").open("w", encoding="utf-8") as f:
            f.write("".join(traceback.format_exc()))
        try:
            trainer.checkpointer.save("model_crash")
        except Exception:
            pass
        raise

    # ---------- Валидация на VAL ----------
    evaluator = COCOEvaluator("psy_val", cfg, True, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference", "psy_val"))
    val_loader = build_detection_test_loader(cfg, "psy_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print("\n[VAL] COCO metrics:\n", results)

    # ---------- Тест на TEST ----------
    evaluator_te = COCOEvaluator("psy_test", cfg, True, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference", "psy_test"))
    test_loader = build_detection_test_loader(cfg, "psy_test")
    results_te = inference_on_dataset(trainer.model, test_loader, evaluator_te)
    print("\n[TEST] COCO metrics:\n", results_te)

    # ---------- Замер P95 / FPS ----------
    from detectron2.checkpoint import DetectionCheckpointer
    final_path = Path(cfg.OUTPUT_DIR) / "model_final.pth"
    if final_path.exists():
        cfg.MODEL.WEIGHTS = str(final_path)
    else:
        last = Path(cfg.OUTPUT_DIR) / "model_final_safe.pth"
        if last.exists():
            cfg.MODEL.WEIGHTS = str(last)

    predictor = DefaultPredictorLite(cfg)
    few_imgs = list_some_images(args.val_img, k=30)
    p95_ms, fps = measure_latency_fps(predictor, few_imgs, warmup=20, runs=200)
    print(f"\n[Runtime] P95 latency: {p95_ms:.1f} ms; FPS(mean): {fps:.1f}  @ {args.min_size_test}p on {cfg.MODEL.DEVICE}")

    with (Path(cfg.OUTPUT_DIR) / "runtime_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"P95_ms={p95_ms:.2f}\nFPS_mean={fps:.2f}\nMIN_SIZE_TEST={args.min_size_test}\nDEVICE={cfg.MODEL.DEVICE}\n")

    print("\nDone. Artifacts:", cfg.OUTPUT_DIR)

if __name__ == "__main__":
    main()

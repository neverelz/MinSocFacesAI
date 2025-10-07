#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_fcos.py  (коммерчески чистая версия)

Архитектура: RetinaNet R50-FPN (Detectron2 0.6) с инициализацией бэкбона
лицензионно-чистыми весами из torchvision (ResNet50 ImageNet1K_V2, BSD-3).
FCOS не используем (в Detectron2 0.6 нет стабильного FCOS), зато сохраняем
весь функционал: онлайн-аугментации, RepeatFactorSampler, мульти-скейл,
прогресс/ETA, периодическая валидация, сейф-чекпоинты, замер P95/FPS,
обработка ошибок, пауза/возобновление обучения.

Запуск (пример):
  python R-CNN/FCOS/train_fcos.py \
    --train-json /home/user/mrgv2/train/annotations.coco.json --train-img /home/user/mrgv2/train/images \
    --val-json   /home/user/mrgv2/val/annotations.coco.json   --val-img   /home/user/mrgv2/val/images \
    --test-json  /home/user/mrgv2/test/annotations.coco.json  --test-img  /home/user/mrgv2/test/images \
    --outdir     /home/user/mrgv2/exp_fcos_mvp \
    --num-classes 47 \
    --device cpu \
    --ims-per-batch 8 --max-iter 2000 --eval-period 500

Пауза:
  - Создайте пустой файл <outdir>/pause.flag (или отправьте процессу SIGUSR1).
  - Тренер сохранит чекпоинт и «уснёт», пока файл не удалят.
Возобновление:
  - Удалите <outdir>/pause.flag — тренинг продолжится с того же места.

Требования окружения:
  - detectron2==0.6, torchvision совместимой версии, Pillow>=9.1
"""

import argparse
import json
import os
import sys
import time
import signal
import traceback
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple, Any

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
from detectron2.utils.logger import setup_logger


# ---------- utils ----------

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_categories_from_coco(json_path: str) -> List[str]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    cats = data.get("categories", []) or []
    cats_sorted = sorted(cats, key=lambda c: int(c.get("id", 0)))
    return [c.get("name", str(c.get("id"))) for c in cats_sorted]

def build_augs():
    # Лёгкие и безопасные аугментации
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
        try:
            latest_loss = float(self.trainer.storage.history('total_loss').latest())
        except Exception:
            latest_loss = float('nan')
        msg = (f"[Iter {iter_num}/{self.total_iter}] "
               f"elapsed={timedelta(seconds=int(elapsed))} "
               f"eta={timedelta(seconds=int(max(0, eta_sec)))} "
               f"lr={self.trainer.optimizer.param_groups[0].get('lr', 0.0):.6f} "
               f"loss={latest_loss:.4f}")
        print(msg)
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

class SafeCheckpointHook(HookBase):
    """В конце обучения попробует сохранить чекпоинт даже при ошибках."""
    def __init__(self, outdir: Path):
        self.outdir = outdir

    def after_train(self):
        try:
            self.trainer.checkpointer.save("model_final_safe")
        except Exception:
            print("[WARN] Failed to save final safe checkpoint:\n", traceback.format_exc())

class PauseHook(HookBase):
    """
    Пауза по флагу <outdir>/pause.flag или сигналу SIGUSR1.
    - При обнаружении паузы сохраняет чекпоинт 'model_pause_iterXXXX'.
    - Ждёт, пока файл не удалят.
    """
    def __init__(self, outdir: Path, poll_every: int = 10):
        self.flag_path = outdir / "pause.flag"
        self.poll_every = poll_every
        safe_mkdir(outdir)

        def _sigusr1(_signo, _frame):
            try:
                self.flag_path.touch(exist_ok=True)
                print(f"[PAUSE] SIGUSR1 received. Created {self.flag_path}")
            except Exception:
                print("[PAUSE] Failed to create pause.flag:", traceback.format_exc())
        try:
            signal.signal(signal.SIGUSR1, _sigusr1)
        except Exception:
            pass  # Не везде доступно

    def after_step(self):
        it = self.trainer.iter + 1
        if it % self.poll_every != 0:
            return
        if self.flag_path.exists():
            name = f"model_pause_iter{it:06d}"
            try:
                self.trainer.checkpointer.save(name)
                print(f"[PAUSE] Saved checkpoint '{name}'. Waiting for {self.flag_path} removal…")
            except Exception:
                print("[PAUSE] Failed to save pause checkpoint:", traceback.format_exc())
            while self.flag_path.exists():
                time.sleep(2.0)
            print("[PAUSE] Flag removed. Resuming training.")


# ---------- trainer ----------

class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
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
    paths = (image_paths * ((warmup + runs) // max(1, len(image_paths)) + 1))[: warmup + runs]
    times = []
    for i, p in enumerate(paths):
        im = cv2.imread(p)
        if im is None:
            continue
        t0 = time.perf_counter()
        _ = predictor(im)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    if not times:
        return (0.0, 0.0)
    lat_ms = np.array(times) * 1000.0
    p95 = float(np.percentile(lat_ms, 95))
    fps = float(1.0 / np.mean(times))
    return (p95, fps)


# ---------- registration (с автозаполнением width/height) ----------

def _register_coco_with_sizes(name, json_file, image_root):
    """
    COCO-лоадер, который добавляет width/height, если их нет в json.
    """
    def _loader():
        records = load_coco_json(json_file, image_root, name)
        miss, total = 0, len(records)
        for d in records:
            if "width" not in d or "height" not in d or d["width"] is None or d["height"] is None:
                path = d.get("file_name", "")
                if not os.path.isabs(path):
                    path = os.path.join(image_root, path)
                im = cv2.imread(path)
                if im is None:
                    continue
                h, w = im.shape[:2]
                d["width"], d["height"] = int(w), int(h)
                miss += 1
        if miss:
            print(f"[COCO] Filled width/height for {miss}/{total} records in '{name}'.")
        return records
    DatasetCatalog.register(name, _loader)
    MetadataCatalog.get(name).set(json_file=json_file, image_root=image_root, evaluator_type="coco")

def register_datasets(train_json, train_img, val_json, val_img, test_json, test_img):
    _register_coco_with_sizes("psy_train", train_json, train_img)
    _register_coco_with_sizes("psy_val",   val_json,   val_img)
    _register_coco_with_sizes("psy_test",  test_json,  test_img)

    train_classes = read_categories_from_coco(train_json)
    MetadataCatalog.get("psy_val").thing_classes  = train_classes
    MetadataCatalog.get("psy_test").thing_classes = train_classes
    return train_classes


# ---------- torchvision ResNet50 -> Detectron2 backbone (BSD-3) ----------

def load_torchvision_resnet50_to_backbone(model):
    """
    Инициализирует бэкбон ResNet50 в Detectron2 совместимыми весами из torchvision.
    Лицензия на веса — BSD-3 (подходит для коммерции).
    """
    try:
        from torchvision.models import resnet50, ResNet50_Weights
    except Exception as e:
        print("[WEIGHTS] torchvision не установлен или несовместим:", e)
        return

    print("[WEIGHTS] Loading torchvision ResNet-50 (ImageNet1K_V2, BSD-3) …")
    tv_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    tv_state = tv_model.state_dict()

    d2_backbone = model.backbone.bottom_up
    d2_state = d2_backbone.state_dict()
    remap = {}

    # stem
    if "stem.conv1.weight" in d2_state and "conv1.weight" in tv_state:
        remap["stem.conv1.weight"] = tv_state["conv1.weight"]
    bn_pairs = [
        ("stem.norm.weight", "bn1.weight"),
        ("stem.norm.bias",   "bn1.bias"),
        ("stem.norm.running_mean", "bn1.running_mean"),
        ("stem.norm.running_var",  "bn1.running_var"),
    ]
    for k_d2, k_tv in bn_pairs:
        if k_d2 in d2_state and k_tv in tv_state:
            remap[k_d2] = tv_state[k_tv]

    # layer1..4 -> res2..5
    layer_map = [("layer1", "res2"), ("layer2", "res3"), ("layer3", "res4"), ("layer4", "res5")]

    def copy_block(tv_prefix: str, d2_prefix: str):
        for k_tv, v_tv in tv_state.items():
            if not k_tv.startswith(tv_prefix):
                continue
            k_tail = k_tv[len(tv_prefix):]  # '0.conv1.weight' или '0.bn1.weight'
            if ".bn" in k_tail:
                k_tail2 = (k_tail.replace(".bn", ".conv")
                                   .replace(".weight", ".norm.weight")
                                   .replace(".bias", ".norm.bias")
                                   .replace(".running_mean", ".norm.running_mean")
                                   .replace(".running_var",  ".norm.running_var"))
            else:
                k_tail2 = k_tail
            k_d2 = f"{d2_prefix}{k_tail2}"
            if k_d2 in d2_state and d2_state[k_d2].shape == v_tv.shape:
                remap[k_d2] = v_tv

    for tv_layer, d2_layer in layer_map:
        copy_block(f"{tv_layer}.", f"{d2_layer}.")

    try:
        d2_backbone.load_state_dict(remap, strict=False)
        print(f"[WEIGHTS] Backbone initialized from torchvision. "
              f"Filled tensors: {len(remap)}/{len(d2_state)}")
    except Exception as e:
        print("[WEIGHTS] Failed to load torchvision weights into backbone:", e)


# ---------- config builder ----------

def build_cfg(args, num_classes: int) -> Any:
    cfg = get_cfg()

    # База RetinaNet; НЕ берём детектроновские COCO-веса (оставим cfg.MODEL.WEIGHTS="")
    retinanet_yaml = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(retinanet_yaml))

    cfg.DATASETS.TRAIN = ("psy_train",)
    cfg.DATASETS.TEST  = ("psy_val",)
    cfg.DATALOADER.NUM_WORKERS = max(2, args.workers)
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = args.repeat_threshold
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False

    cfg.INPUT.MIN_SIZE_TRAIN = (640, 720, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST  = args.min_size_test
    cfg.INPUT.MAX_SIZE_TEST  = max(args.min_size_test, 1280)
    cfg.INPUT.RANDOM_FLIP = "horizontal"

    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.infer_th
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes

    if args.device:
        cfg.MODEL.DEVICE = args.device

    cfg.MODEL.WEIGHTS = ""  # важно: не грузим COCO-веса Detectron2

    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.WARMUP_ITERS = min(1000, max(200, args.max_iter // 20))
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

    cfg.TEST.EVAL_PERIOD = args.eval_period

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
        from detectron2.checkpoint import DetectionCheckpointer
        DetectionCheckpointer(self.model).load(self.cfg.MODEL.WEIGHTS)
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


# ---------- CLI / helpers ----------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--train-img", required=True)
    ap.add_argument("--val-json",   required=True)
    ap.add_argument("--val-img",    required=True)
    ap.add_argument("--test-json",  required=True)
    ap.add_argument("--test-img",   required=True)
    ap.add_argument("--outdir",     required=True)
    ap.add_argument("--num-classes", type=int, default=None)
    ap.add_argument("--device", choices=["cpu","cuda"], default=None)
    ap.add_argument("--ims-per-batch", type=int, default=8)
    ap.add_argument("--base-lr", type=float, default=5e-4)
    ap.add_argument("--max-iter", type=int, default=12000)
    ap.add_argument("--eval-period", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repeat-threshold", type=float, default=0.002)
    ap.add_argument("--min-size-test", type=int, default=720)
    ap.add_argument("--infer-th", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=-1)
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


# ---------- main ----------

def main():
    args = parse_args()
    safe_mkdir(Path(args.outdir))
    log_file = Path(args.outdir) / "train_log.txt"
    sanity_paths(args)

    classes = register_datasets(args.train_json, args.train_img, args.val_json, args.val_img, args.test_json, args.test_img)
    num_classes = args.num_classes if args.num_classes is not None else len(classes)
    if args.num_classes is not None and args.num_classes != len(classes):
        print(f"[INFO] Overriding num_classes from {len(classes)} -> {args.num_classes}")
    print(f"[INFO] NUM_CLASSES = {num_classes}")

    cfg = build_cfg(args, num_classes)

    setup_logger(output=str(Path(args.outdir)))
    default_setup(cfg, {})  # пишет полный конфиг на диск

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)

    # Инициализация бэкбона из torchvision (BSD-3), если внешних весов нет
    if not cfg.MODEL.WEIGHTS:
        load_torchvision_resnet50_to_backbone(trainer.model)

    # Хуки
    trainer.start_time = time.time()
    trainer.register_hooks([
        ETAHook(total_iter=cfg.SOLVER.MAX_ITER, log_path=log_file, every=50),
        PauseHook(outdir=Path(cfg.OUTPUT_DIR), poll_every=10),
        SafeCheckpointHook(outdir=Path(cfg.OUTPUT_DIR)),
    ])

    # Обучение с перехватом ошибок
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving last checkpoint...")
        try:
            trainer.checkpointer.save("model_interrupted")
        except Exception:
            pass
        sys.exit(130)
    except Exception:
        print("[FATAL] Training crashed. Saving last checkpoint…")
        try:
            with (Path(cfg.OUTPUT_DIR) / "crash_trace.txt").open("w", encoding="utf-8") as f:
                f.write("".join(traceback.format_exc()))
        except Exception:
            pass
        try:
            trainer.checkpointer.save("model_crash")
        except Exception:
            pass
        raise

    # ---------- Валидация ----------
    evaluator = COCOEvaluator("psy_val", cfg, True, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference", "psy_val"))
    val_loader = build_detection_test_loader(cfg, "psy_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print("\n[VAL] COCO metrics:\n", results)

    # ---------- Тест ----------
    evaluator_te = COCOEvaluator("psy_test", cfg, True, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference", "psy_test"))
    test_loader = build_detection_test_loader(cfg, "psy_test")
    results_te = inference_on_dataset(trainer.model, test_loader, evaluator_te)
    print("\n[TEST] COCO metrics:\n", results_te)

    # ---------- Замер P95 / FPS ----------
    final_path = Path(cfg.OUTPUT_DIR) / "model_final.pth"
    if final_path.exists():
        cfg.MODEL.WEIGHTS = str(final_path)
    else:
        last_safe = Path(cfg.OUTPUT_DIR) / "model_final_safe.pth"
        if last_safe.exists():
            cfg.MODEL.WEIGHTS = str(last_safe)

    predictor = DefaultPredictorLite(cfg)
    few_imgs = list_some_images(args.val_img, k=30)
    p95_ms, fps = measure_latency_fps(predictor, few_imgs, warmup=20, runs=200)
    print(f"\n[Runtime] P95 latency: {p95_ms:.1f} ms; FPS(mean): {fps:.1f}  @ {args.min_size_test}p on {cfg.MODEL.DEVICE}")

    try:
        with (Path(cfg.OUTPUT_DIR) / "runtime_report.txt").open("w", encoding="utf-8") as f:
            f.write(f"P95_ms={p95_ms:.2f}\nFPS_mean={fps:.2f}\nMIN_SIZE_TEST={args.min_size_test}\nDEVICE={cfg.MODEL.DEVICE}\n")
    except Exception:
        pass

    print("\nDone. Artifacts:", cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()

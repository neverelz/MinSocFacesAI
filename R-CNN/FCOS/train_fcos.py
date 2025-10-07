#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_fcos.py  (коммерчески чистая версия)

Цель: FCOS R50-FPN с инициализацией бэкбона лицензией BSD-3 (torchvision
ResNet50 ImageNet1K_V2). Если текущая версия detectron2 не поддерживает FCOS,
включается фолбэк на RetinaNet R50-FPN при сохранении функционала: онлайн-
аугментации, RepeatFactorSampler, мульти-скейл, прогресс/ETA, периодическая
валидация, сейф-чекпоинты, замер P95/FPS, обработка ошибок и пауза/возобнов-
ление обучения. Веса Detectron2 COCO не используются (коммерческая чистота).

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
from yacs.config import CfgNode as CN

# --- Pillow shim: фиксы для Pillow>=10 (совместимость с детектроновскими аугментациями)
try:
    from PIL import Image as _PIL_Image
    if not hasattr(_PIL_Image, "LINEAR"):
        if hasattr(_PIL_Image, "BILINEAR"):
            _PIL_Image.LINEAR = _PIL_Image.BILINEAR
        else:
            from PIL.Image import Resampling
            _PIL_Image.LINEAR = Resampling.BILINEAR
except Exception:
    pass

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
from detectron2.structures import ImageList


# Регистрация AdelaiDet (если установлен) — до сборки модели
try:
    import importlib
    import adet  # noqa: F401
    # Явная подгрузка FCOS, чтобы зарегистрировать META_ARCH
    importlib.import_module("adet.modeling.fcos.fcos")
except Exception:
    pass

# ---------- FCOS shim meta-arch (совместимость с Detectron2 0.6) ----------
try:
    from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
    import importlib
    _adet_fcos_mod = importlib.import_module("adet.modeling.fcos.fcos")

    if not hasattr(_adet_fcos_mod, "FCOS"):
        _adet_fcos_mod = None

    # Зарегистрируем совместимую оболочку, если нет корректного FCOSDetector
    need_register_shim = False
    try:
        META_ARCH_REGISTRY.get("FCOSDetector")
    except KeyError:
        need_register_shim = _adet_fcos_mod is not None

    if need_register_shim:
        import torch.nn as nn

        @META_ARCH_REGISTRY.register()
        class FCOSDetector(nn.Module):  # noqa: N801 - имя для реестра (старый, может быть переопределён)
            def __init__(self, cfg):
                super().__init__()
                # Построим бэкбон, получим его выходные формы
                self.backbone = build_backbone(cfg)
                input_shape = self.backbone.output_shape()

                # Синхронизируем IN_FEATURES/STRIDES/SIZES_OF_INTEREST с реально доступными уровнями
                desired = ["p3", "p4", "p5", "p6", "p7"]
                available = [k for k in input_shape.keys() if k.startswith("p")]
                # Сохраним порядок по номеру пирамиды
                def _p_level(s):
                    try:
                        return int(s[1:])
                    except Exception:
                        return 0
                available = sorted(available, key=_p_level)
                in_feats = [l for l in desired if l in available] or available

                cfg.MODEL.FCOS.IN_FEATURES = in_feats
                # Подберём соответствующие страйды
                stride_map = {"p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64, "p7": 128}
                cfg.MODEL.FCOS.FPN_STRIDES = [stride_map.get(l, 8) for l in in_feats]
                # Сократим интервалы интереса под количество уровней (3 или 5)
                full_soi = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
                cfg.MODEL.FCOS.SIZES_OF_INTEREST = full_soi[: len(in_feats)]

                # Инициализируем оригинальный FCOS с нужной сигнатурой
                self.model = _adet_fcos_mod.FCOS(cfg, input_shape)

                # Нормализация изображений
                import torch
                pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
                pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
                self.register_buffer("pixel_mean", pixel_mean, persistent=False)
                self.register_buffer("pixel_std", pixel_std, persistent=False)
                self.device = torch.device(cfg.MODEL.DEVICE)
                self.to(self.device)

            def forward(self, batched_inputs):
                # Извлекаем тензоры изображений, нормализуем и формируем ImageList
                images = [x["image"].to(self.device) for x in batched_inputs]
                images = [(img - self.pixel_mean) / self.pixel_std for img in images]
                images = ImageList.from_tensors(images, self.backbone.size_divisibility)
                # Получаем пирамиду признаков
                features = self.backbone(images.tensor)
                # Вызов FCOS: эта версия ожидает features и batched_inputs
                return self.model(features, batched_inputs)

        # Регистрируем шиму с уникальным именем и явным использованием нового forward
        @META_ARCH_REGISTRY.register()
        class FCOSDetectorShim(FCOSDetector):  # noqa: N801
            pass
except Exception:
    pass

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




# ---------- FCOS config defaults ----------

def _fill_missing_fcos_keys(cfg):
    f = cfg.MODEL.FCOS
    # Библиотека AdelaiDet ожидает ряд ключей в cfg.MODEL.FCOS — заполним дефолтами
    defaults = {
        "IN_FEATURES": ["p3", "p4", "p5", "p6", "p7"],
        "FPN_STRIDES": [8, 16, 32, 64, 128],
        "SIZES_OF_INTEREST": [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]],
        "INFERENCE_TH": 0.3,
        "INFERENCE_TH_TEST": 0.3,
        "INFERENCE_TH_TRAIN": 0.05,
        "NMS_TH": 0.6,
        "PRIOR_PROB": 0.01,
        "LOSS_ALPHA": 0.25,
        "LOSS_GAMMA": 2.0,
        "LOC_LOSS_TYPE": "giou",
        "PRE_NMS_TOPK_TRAIN": 2000,
        "PRE_NMS_TOPK_TEST": 1000,
        "POST_NMS_TOPK_TRAIN": 1000,
        "POST_NMS_TOPK_TEST": 1000,
        "THRESH_WITH_CTR": False,
        "CENTERNESS_ON_REG": False,
        "BOX_QUALITY": "centerness",
        "CENTER_SAMPLE": True,
        "POS_RADIUS": 1.5,
        "YIELD_PROPOSAL": False,
        "YIELD_BOX_FEATURES": False,
        "USE_DEFORMABLE": False,
        "DEFORMABLE_GROUPS": 1,
        "NUM_CLS_CONVS": 4,
        "NUM_BOX_CONVS": 4,
        "NUM_SHARE_CONVS": 0,
        "USE_SCALE": True,
        "LOSS_NORMALIZER_CLS": "fg",
        "LOSS_NORMALIZER_BOX": "fg",
        "LOSS_WEIGHT_CLS": 1.0,
        "LOSS_WEIGHT_BOX": 1.0,
        "LOSS_WEIGHT_CTR": 1.0,
        "CLS_LOSS_NORMALIZER": "fg",       # allowed: moving_fg | fg | all
        "BOX_LOSS_NORMALIZER": "fg",
        "CTR_LOSS_NORMALIZER": "fg",
        "NORM": "GN",
    }
    for k, v in defaults.items():
        if not hasattr(f, k):
            setattr(f, k, v)

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

    # Найдём реальный бэкбон внутри модели (учитывая FCOSDetector shim)
    target_model = model
    # unwrap FCOSDetector -> FCOS
    if hasattr(target_model, "model") and not hasattr(target_model, "backbone"):
        target_model = getattr(target_model, "model")

    # возьмём backbone / backbone.bottom_up
    if hasattr(target_model, "backbone"):
        bb = getattr(target_model, "backbone")
        d2_backbone = getattr(bb, "bottom_up", bb)
    else:
        print("[WEIGHTS] No backbone found on model; skip torchvision init")
        return
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

    # Определим доступность FCOS в текущей сборке detectron2
    def _has_fcos():
        # Вариант 1: FCOS из AdelaiDet (предпочтительно)
        try:
            import adet  # noqa: F401  # регистрация моделей
            from adet.config import add_fcos_config  # noqa: F401
            return "adet"
        except Exception:
            pass
        # Вариант 2: FCOS внутри detectron2 (если присутствует)
        try:
            from detectron2.modeling.meta_arch import fcos  # noqa: F401
            return "d2"
        except Exception:
            return ""

    prefer_fcos = getattr(args, "prefer_fcos", True)
    fcos_src = _has_fcos()

    if prefer_fcos and fcos_src:
        # Собираем FCOS (AdelaiDet или встроенный D2)
        cfg.merge_from_file(model_zoo.get_config_file("Base-RCNN-FPN.yaml"))
        if fcos_src == "adet":
            try:
                from adet.config import add_fcos_config
                add_fcos_config(cfg)  # добавляет узел MODEL.FCOS и др.
            except Exception:
                print("[WARN] AdelaiDet найден, но add_fcos_config не сработал. Создам MODEL.FCOS вручную.")
        # Если после добавления узла его всё ещё нет — создаём вручную
        if not hasattr(cfg.MODEL, "FCOS"):
            cfg.MODEL.FCOS = CN()
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
            cfg.MODEL.FCOS.YIELD_BOX_FEATURES = False
            cfg.MODEL.FCOS.USE_DEFORMABLE = False
            cfg.MODEL.FCOS.DEFORMABLE_GROUPS = 1
            cfg.MODEL.FCOS.NORM = "GN"
            cfg.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
            # Минимальные требования FCOSHead из AdelaiDet
            cfg.MODEL.FCOS.NUM_CLS_CONVS = 4
            cfg.MODEL.FCOS.NUM_BOX_CONVS = 4
        cfg.MODEL.META_ARCHITECTURE = "FCOS"
        cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
        cfg.MODEL.RESNETS.DEPTH = 50
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        # Гиперпараметры FCOS
        cfg.MODEL.FCOS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.FCOS.INFERENCE_TH = args.infer_th
        cfg.MODEL.FCOS.NMS_TH = 0.6
        cfg.MODEL.FCOS.PRIOR_PROB = 0.01
        cfg.MODEL.FCOS.SIZES_OF_INTEREST = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
        cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
        cfg.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
        cfg.MODEL.FCOS.CENTER_SAMPLE = True
        cfg.MODEL.FCOS.POS_RADIUS = 1.5
        cfg.MODEL.FCOS.YIELD_PROPOSAL = False
        cfg.MODEL.FCOS.YIELD_BOX_FEATURES = False
        if not hasattr(cfg.MODEL.FCOS, "USE_DEFORMABLE"):
            cfg.MODEL.FCOS.USE_DEFORMABLE = False
        if not hasattr(cfg.MODEL.FCOS, "DEFORMABLE_GROUPS"):
            cfg.MODEL.FCOS.DEFORMABLE_GROUPS = 1
        cfg.MODEL.FCOS.NORM = "GN"
        if not hasattr(cfg.MODEL.FCOS, "IN_FEATURES"):
            cfg.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
        if not hasattr(cfg.MODEL.FCOS, "NUM_CLS_CONVS"):
            cfg.MODEL.FCOS.NUM_CLS_CONVS = 4
        if not hasattr(cfg.MODEL.FCOS, "NUM_BOX_CONVS"):
            cfg.MODEL.FCOS.NUM_BOX_CONVS = 4
        # Универсально заполним все недостающие ключи FCOS
        try:
            _fill_missing_fcos_keys(cfg)
        except Exception:
            pass
        cfg._SELECTED_META_ARCH = "FCOS"
    else:
        if prefer_fcos and not fcos_src:
            raise RuntimeError(
                "FCOS недоступен. Установите AdelaiDet (рекомендуется):\n"
                "  pip install 'git+https://github.com/aim-uofa/AdelaiDet'\n"
                "или используйте Detectron2 сборку с FCOS, либо запустите без --prefer-fcos."
            )
        retinanet_yaml = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(retinanet_yaml))
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.infer_th
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg._SELECTED_META_ARCH = "RetinaNet"

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

    if args.device:
        cfg.MODEL.DEVICE = args.device

    cfg.MODEL.WEIGHTS = ""

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
    ap.add_argument("--prefer-fcos", action="store_true", help="Принудительно использовать FCOS (ошибка, если недоступен)")
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
    print(f"[ARCH] Selected meta-architecture: {getattr(cfg, '_SELECTED_META_ARCH', 'unknown')}")

    # Гарантируем регистрацию выбранной meta-архитектуры
    try:
        from detectron2.modeling import META_ARCH_REGISTRY
        selected = getattr(cfg, "_SELECTED_META_ARCH", cfg.MODEL.META_ARCHITECTURE)
        candidates = [selected]
        if selected == "FCOS":
            # Предпочтительно использовать FCOSDetector (имеет сигнатуру Detectron2)
            candidates = ["FCOSDetector", "FCOS"]
        resolved = None
        for name in candidates:
            try:
                META_ARCH_REGISTRY.get(name)
                resolved = name
                break
            except KeyError:
                continue
        if resolved is None and selected == "FCOS":
            # Попробуем принудительно зарегистрировать класс из AdelaiDet
            import importlib
            try:
                mod = importlib.import_module("adet.modeling.fcos.fcos")
                for cls_name in ["FCOS", "FCOSDetector"]:
                    if hasattr(mod, cls_name):
                        try:
                            META_ARCH_REGISTRY.register(getattr(mod, cls_name))
                        except Exception:
                            pass
                # Проверим снова
                for name in ["FCOSDetector", "FCOS"]:
                    try:
                        META_ARCH_REGISTRY.get(name)
                        resolved = name
                        break
                    except KeyError:
                        continue
            except Exception:
                pass
        if resolved is None:
            raise RuntimeError("FCOS не зарегистрирован в META_ARCH. Проверьте установку AdelaiDet в текущем окружении.")
        # Принудительно используем шиму, чтобы точно задействовать новый forward
        cfg.MODEL.META_ARCHITECTURE = "FCOSDetectorShim"
        print("[ARCH] Using registered meta-architecture: FCOSDetectorShim")
    except Exception:
        raise

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

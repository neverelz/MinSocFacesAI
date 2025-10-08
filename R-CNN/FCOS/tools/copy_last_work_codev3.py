#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click pipeline:
  Stage A: RetinaNet R18-FPN  (быстрый старт, заморозка бэкбона + RepeatFactorSampler)
  Stage B: FCOS R34-FPN       (качество повыше, тот же датасет/аугменты)

Коммерчески-чистая инициализация (без COCO-весов Detectron2).
Работает на CPU, устойчивые хуки/чекпоинты/heartbeat.

Пример запуска:
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE \
KMP_AFFINITY=disabled PYTORCH_SHOW_CPP_STACKTRACES=1 \
python R-CNN/FCOS/train_fcos.py \
  --train-json /home/user/mrgv2/tiny50/tiny_train.json \
  --train-img  /home/user/mrgv2/tiny50 \
  --val-json   /home/user/mrgv2/tiny50/tiny_val.json \
  --val-img    /home/user/mrgv2/tiny50 \
  --test-json  /home/user/mrgv2/tiny50/tiny_val.json \
  --test-img   /home/user/mrgv2/tiny50 \
  --outdir     /home/user/mrgv2/exp_mvp \
  --num-classes 47 \
  --device cpu \
  --ims-per-batch 2 \
  --workers 0 \
  --base-lr 5e-4 \
  --max-iter 6000 \
  --eval-period 500 \
  --repeat-threshold 0.01 \
  --min-size-test 640 \
  --prefer-fcos | tee /home/user/mrgv2/exp_mvp/run.log
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
import importlib

import faulthandler, os as _os
faulthandler.enable()
_os.environ.setdefault("PYTHONFAULTHANDLER", "1")
_os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn as nn
import cv2
from yacs.config import CfgNode as CN

# Runtime stability
import os as _os_runtime
_os_runtime.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
_os_runtime.environ.setdefault("OMP_NUM_THREADS", "1")
_os_runtime.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    cv2.setNumThreads(0)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# Pillow >=10 shims
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

# Detectron2 / AdelaiDet
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, HookBase, default_setup
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data.datasets import load_coco_json
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger
from detectron2.structures import ImageList, BoxMode, Instances, Boxes
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.modeling import ShapeSpec

# try register AdelaiDet FCOS
try:
    import adet  # noqa: F401
    importlib.import_module("adet.modeling.fcos.fcos")
except Exception:
    pass

# ---------- FCOS shim meta-arch ----------
try:
    _adet_fcos_mod = importlib.import_module("adet.modeling.fcos.fcos")
    if not hasattr(_adet_fcos_mod, "FCOS"):
        _adet_fcos_mod = None
except Exception:
    _adet_fcos_mod = None

try:
    _map = META_ARCH_REGISTRY._obj_map
    _map.pop("FCOSDetector", None)
    _map.pop("FCOSDetectorShim", None)
except Exception:
    pass

@META_ARCH_REGISTRY.register()
class FCOSDetector(nn.Module):  # noqa: N801
    def __init__(self, cfg):
        super().__init__()
        try:
            _adet_fcos_mod = importlib.import_module("adet.modeling.fcos.fcos")
            assert hasattr(_adet_fcos_mod, "FCOS")
        except Exception:
            raise RuntimeError(
                "AdelaiDet (FCOS) не найден. Установите:\n"
                "  pip install 'git+https://github.com/aim-uofa/AdelaiDet'\n"
                "или запускайте без --prefer-fcos (fallback на RetinaNet)."
            )

        self.backbone = build_resnet_fpn_backbone(cfg, ShapeSpec(channels=3))
        input_shape = self.backbone.output_shape()

        desired = ["p3", "p4", "p5", "p6", "p7"]
        available = [k for k in input_shape.keys() if k.startswith("p")]
        def _plevel(s):
            try:
                return int(s[1:])
            except Exception:
                return 0
        available = sorted(available, key=_plevel)
        in_feats = [l for l in desired if l in available] or available

        stride_map = {"p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64, "p7": 128}
        fpn_strides = [stride_map.get(l, 8) for l in in_feats]

        soi_pairs_full = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
        soi_pairs = soi_pairs_full[: len(in_feats)]

        cfg.MODEL.FCOS.IN_FEATURES = in_feats
        cfg.MODEL.FCOS.FPN_STRIDES = fpn_strides
        cfg.MODEL.FCOS.SIZES_OF_INTEREST = soi_pairs
        if not hasattr(cfg.MODEL.FCOS, "BOX_QUALITY") or cfg.MODEL.FCOS.BOX_QUALITY not in ("ctrness", "iou"):
            cfg.MODEL.FCOS.BOX_QUALITY = "ctrness"

        self.model = _adet_fcos_mod.FCOS(cfg, input_shape)
        self.model.backbone = self.backbone
        self.model.in_features = in_feats

        if hasattr(self.model, "fcos_outputs"):
            if getattr(self.model.fcos_outputs, "fpn_strides", None) in (None, []):
                self.model.fcos_outputs.fpn_strides = fpn_strides

            raw_soi = getattr(self.model.fcos_outputs, "sizes_of_interest", None)
            def _coerce_soi_to_pairs(x, fallback_pairs):
                if x is None:
                    return [[float(a), float(b)] for a, b in fallback_pairs]
                if isinstance(x, (list, tuple)) and len(x) > 0 and not isinstance(x[0], (list, tuple)):
                    return [[float(a), float(b)] for a, b in fallback_pairs]
                pairs = []
                for el in list(x)[: len(in_feats)]:
                    if isinstance(el, (list, tuple)) and len(el) == 1 and isinstance(el[0], (list, tuple)):
                        el = el[0]
                    if isinstance(el, (list, tuple)) and len(el) == 2 \
                            and not isinstance(el[0], (list, tuple)) and not isinstance(el[1], (list, tuple)):
                        lo, hi = float(el[0]), float(el[1])
                    else:
                        idx = len(pairs)
                        lo, hi = map(float, fallback_pairs[idx])
                    pairs.append([lo, hi])
                while len(pairs) < len(in_feats):
                    idx = len(pairs)
                    lo, hi = map(float, fallback_pairs[idx])
                    pairs.append([lo, hi])
                return pairs
            coerced_soi = _coerce_soi_to_pairs(raw_soi, soi_pairs)
            self.model.fcos_outputs.sizes_of_interest = coerced_soi

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean, persistent=False)
        self.register_buffer("pixel_std", pixel_std, persistent=False)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.to(self.device)

    def _sanitize_targets(self, batched_inputs, image_sizes):
        for i, d in enumerate(batched_inputs):
            h, w = image_sizes[i]
            inst = d.get("instances", None)
            if inst is None:
                inst = Instances((h, w))
                d["instances"] = inst
            if not hasattr(inst, "image_size") or inst.image_size is None:
                inst._image_size = (h, w)
            if not inst.has("gt_boxes"):
                inst.set("gt_boxes", Boxes(torch.zeros((0, 4), device=self.device)))
            else:
                inst.gt_boxes.clip((h, w))
                inst.gt_boxes.tensor = inst.gt_boxes.tensor.to(self.device).float()
            if not inst.has("gt_classes"):
                inst.set("gt_classes", torch.zeros((0,), dtype=torch.int64, device=self.device))
            else:
                inst.gt_classes = inst.gt_classes.to(self.device).long()

    def forward(self, batched_inputs, **kwargs):
        imgs = [x["image"].to(self.device) for x in batched_inputs]
        imgs = [(img - self.pixel_mean) / self.pixel_std for img in imgs]
        images = ImageList.from_tensors(imgs, self.backbone.size_divisibility)
        features_dict = self.backbone(images.tensor)
        assert isinstance(features_dict, dict), "FPN must return dict of feature maps"

        targets = None
        if self.training:
            self._sanitize_targets(batched_inputs, images.image_sizes)
            targets = [d["instances"].to(self.device) for d in batched_inputs]
            if all((t.gt_boxes.tensor.numel() == 0) for t in targets):
                dummy = None
                for p in self.model.parameters():
                    if p.requires_grad:
                        v = p.sum() * 0.0
                        dummy = v if dummy is None else (dummy + v)
                if dummy is None:
                    dummy = torch.zeros([], device=self.device, dtype=torch.float32, requires_grad=True)
                return {"loss_fcos_cls": dummy, "loss_fcos_loc": dummy, "loss_fcos_ctr": dummy}

        if not hasattr(self, "_once_dbg"):
            soi = getattr(self.model.fcos_outputs, "sizes_of_interest", None)
            strides = getattr(self.model.fcos_outputs, "fpn_strides", None)
            def _pairs_shape(x):
                try:
                    n = len(x)
                    m = len(x[0]) if n > 0 and hasattr(x[0], "__len__") else None
                    return (n, m)
                except Exception:
                    return None
            print("[DBG] in_features:", self.model.in_features)
            print("[DBG] strides:", strides)
            print("[DBG] sizes_of_interest shape:", _pairs_shape(soi))
            print("[DBG] BOX_QUALITY:", getattr(self.model.fcos_outputs, "box_quality", None))
            self._once_dbg = True

        out = self.model(images, features_dict, targets)
        if self.training:
            if isinstance(out, tuple):
                _, losses = out
                return losses
            elif isinstance(out, dict):
                return out
            else:
                raise TypeError(f"Unexpected FCOS train output type: {type(out)}")
        else:
            if isinstance(out, tuple):
                preds, _ = out
                return preds
            return out

try:
    _map = META_ARCH_REGISTRY._obj_map
    _map.pop("FCOSDetector", None)
    _map.pop("FCOSDetectorShim", None)
except Exception:
    pass

@META_ARCH_REGISTRY.register()
class FCOSDetectorShim(FCOSDetector):  # noqa: N801
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
    return [
        T.RandomFlip(horizontal=True, vertical=False),
        T.RandomBrightness(0.95, 1.05),
        T.RandomContrast(0.95, 1.05),
        T.RandomSaturation(0.95, 1.05),
    ]

def mapper_with_augs(dataset_dict):
    dataset_dict = dataset_dict.copy()
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    augs = build_augs()
    aug_input = T.AugInput(image)
    tfms = T.AugmentationList(augs)(aug_input)
    image = aug_input.image

    annos_raw = dataset_dict.get("annotations", [])
    annos = []
    H, W = image.shape[:2]
    for obj in annos_raw:
        a = utils.transform_instance_annotations(obj, tfms, (H, W))
        bbox = a.get("bbox", None)
        if bbox is None:
            continue
        bbox_np = np.asarray(bbox, dtype=np.float32)
        bm = a.get("bbox_mode", BoxMode.XYXY_ABS)
        if bm != BoxMode.XYXY_ABS:
            bbox_np = BoxMode.convert(bbox_np, bm, BoxMode.XYXY_ABS)
        x0, y0, x1, y1 = map(float, bbox_np[:4])
        if (x1 - x0) < 0.5 or (y1 - y0) < 0.5:
            continue
        a["bbox"] = [x0, y0, x1, y1]
        a["bbox_mode"] = BoxMode.XYXY_ABS
        annos.append(a)

    H2, W2 = image.shape[:2]
    inst = utils.annotations_to_instances(annos, (H2, W2))
    if inst.has("gt_boxes"):
        inst.gt_boxes.clip((H2, W2))
    if not inst.has("gt_classes"):
        inst.set("gt_classes", torch.zeros((0,), dtype=torch.int64))
    return {
        "image": torch.as_tensor(image.transpose(2, 0, 1).copy()).float(),
        "height": H2,
        "width": W2,
        "instances": inst,
    }

def mapper_skip_none(d):
    out = mapper_with_augs(d)
    if out is None:
        raise ValueError("empty-sample")
    return out

# ---------- FCOS cfg defaults ----------
def _fill_missing_fcos_keys(cfg):
    f = cfg.MODEL.FCOS
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
        "BOX_QUALITY": "ctrness",
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
        "CLS_LOSS_NORMALIZER": "fg",
        "BOX_LOSS_NORMALIZER": "fg",
        "CTR_LOSS_NORMALIZER": "fg",
        "NORM": "GN",
    }
    for k, v in defaults.items():
        if not hasattr(f, k):
            setattr(f, k, v)

# ---------- hooks ----------
class ETAHook(HookBase):
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
    def __init__(self, outdir: Path):
        self.outdir = outdir
    def after_train(self):
        try:
            self.trainer.checkpointer.save("model_final_safe")
        except Exception:
            print("[WARN] Failed to save final safe checkpoint:\n", traceback.format_exc())

class HeartbeatHook(HookBase):
    def __init__(self, every: int = 1):
        self.every = max(1, int(every))
    def before_train(self):
        print("[HB] Training starting…", flush=True)
    def after_step(self):
        it = self.trainer.iter + 1
        if it % self.every == 0:
            try:
                loss = float(self.trainer.storage.history('total_loss').latest())
            except Exception:
                loss = float('nan')
            print(f"[HB] iter={it} loss={loss}", flush=True)
    def after_train(self):
        print("[HB] Training finished.", flush=True)

class PauseHook(HookBase):
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
            pass
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

class FreezeBackboneUntil(HookBase):
    """
    Заморозить бэкбон до until_iter (ускоряет CPU-обучение и стабилизирует лоссы).
    unfreeze_fpn=False — если нужно заморозить и FPN.
    """
    def __init__(self, module, until_iter: int = 1500, unfreeze_fpn: bool = True):
        self.module = module
        self.until_iter = until_iter
        self.unfreeze_fpn = unfreeze_fpn
        self._done = False
    def before_train(self):
        b = getattr(self.module, "backbone", None)
        if b is None:
            return
        bb = getattr(b, "bottom_up", b)  # ResNet
        for p in bb.parameters():
            p.requires_grad = False
        if not self.unfreeze_fpn:
            for p in b.parameters():
                p.requires_grad = False
        print(f"[FREEZE] backbone frozen until iter {self.until_iter}")
    def after_step(self):
        if self._done:
            return
        it = self.trainer.iter + 1
        if it >= self.until_iter:
            for p in self.module.parameters():
                p.requires_grad = True
            self._done = True
            print(f"[FREEZE] backbone unfrozen at iter {it}")

class FreezeBackboneHook(HookBase):
    """
    1) Перед обучением замораживает backbone.bottom_up (ResNet) и переводит norm-слои в eval().
    2) На указанной итерации размораживает и ПЕРЕСОБИРАЕТ оптимизатор, чтобы новые параметры попали в training.
    Работает для RetinaNet и FCOS (включая shim).
    """
    def __init__(self, iters: int):
        self.until_iter = int(max(0, iters))
        self._frozen = False
        self._unfrozen = False
        self._frozen_param_names = []

    # ---- helpers ----
    def _get_model(self):
        return self.trainer.model

    def _get_backbone(self, m):
        # unwrap shim → FCOS если нужно
        if hasattr(m, "model") and not hasattr(m, "backbone"):
            m = m.model
        # достаём backbone / backbone.bottom_up
        if hasattr(m, "backbone"):
            bb = getattr(m, "backbone")
            return getattr(bb, "bottom_up", bb)
        return None

    def _set_requires_grad(self, module, requires_grad: bool):
        for n, p in module.named_parameters(recurse=True):
            p.requires_grad = requires_grad
            if requires_grad is False:
                self._frozen_param_names.append(n)

    def _set_norm_eval(self, module):
        # Норм-слои в eval() во время фриза (BN/IN/LN/GN — безопасно)
        for m in module.modules():
            if hasattr(m, "running_mean") or m.__class__.__name__.endswith(("Norm", "BatchNorm2d")):
                m.eval()

    def _rebuild_optimizer(self):
        # корректно пересобираем оптимизатор под новое множество trainable-параметров
        try:
            # у DefaultTrainer есть build_optimizer(cls, cfg, model)
            new_opt = type(self.trainer).build_optimizer(self.trainer.cfg, self.trainer.model)
        except Exception:
            from detectron2.solver import build_optimizer
            new_opt = build_optimizer(self.trainer.cfg, self.trainer.model)
        self.trainer.optimizer = new_opt

    # ---- lifecycle ----
    def before_train(self):
        if self.until_iter <= 0:
            print(f"[FREEZE] freeze-backbone-iters=0 → хук отключён", flush=True)
            return
        m = self._get_model()
        bb = self._get_backbone(m)
        if bb is None:
            print("[FREEZE] Backbone не найден → пропускаю заморозку", flush=True)
            return

        # замораживаем
        self._set_requires_grad(bb, False)
        self._set_norm_eval(bb)
        self._frozen = True
        self._frozen_param_names = sorted(set(self._frozen_param_names))
        print(f"[FREEZE] Заморозил backbone.bottom_up на первые {self.until_iter} итераций. "
              f"Параметров заморожено: {len(self._frozen_param_names)}", flush=True)

        # пересобираем оптимизатор, чтобы исключить замороженные параметры
        self._rebuild_optimizer()
        print("[FREEZE] Optimizer пересобран под 'замороженную' конфигурацию.", flush=True)

    def after_step(self):
        if not self._frozen or self._unfrozen:
            return
        it = self.trainer.iter + 1
        if it >= self.until_iter:
            m = self._get_model()
            bb = self._get_backbone(m)
            if bb is None:
                print("[FREEZE] Не нашёл backbone при разморозке → пропуск", flush=True)
                self._unfrozen = True
                return

            # разморозка
            self._set_requires_grad(bb, True)
            # можно вернуть norm-слои к train(), но detectron2 сам управляет .train()
            bb.train()

            # важное: пересобираем оптимизатор заново — теперь с параметрами бэкбона
            self._rebuild_optimizer()
            self._unfrozen = True
            print(f"[FREEZE] Разморозил backbone на итерации {it}. Optimizer пересобран для full-train.", flush=True)


# ---------- trainer ----------
class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper_skip_none)
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        safe_mkdir(Path(output_folder))
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

# ---------- latency / FPS test ----------
@torch.no_grad()
def measure_latency_fps(predictor, image_paths: List[str], warmup: int = 20, runs: int = 200) -> Tuple[float, float]:
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

# ---------- registration (width/height auto) ----------
def _register_coco_with_sizes(name, json_file, image_root):
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

# ---------- optional R50 weights loader (kept for compatibility) ----------
def load_torchvision_resnet50_to_backbone(model):
    try:
        from torchvision.models import resnet50, ResNet50_Weights
    except Exception as e:
        print("[WEIGHTS] torchvision не установлен или несовместим:", e)
        return
    print("[WEIGHTS] Loading torchvision ResNet-50 (ImageNet1K_V2, BSD-3) …")
    tv_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    tv_state = tv_model.state_dict()

    target_model = model
    if hasattr(target_model, "model") and not hasattr(target_model, "backbone"):
        target_model = getattr(target_model, "model")

    if hasattr(target_model, "backbone"):
        bb = getattr(target_model, "backbone")
        d2_backbone = getattr(bb, "bottom_up", bb)
    else:
        print("[WEIGHTS] No backbone found on model; skip torchvision init")
        return
    d2_state = d2_backbone.state_dict()
    remap = {}

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

    layer_map = [("layer1", "res2"), ("layer2", "res3"), ("layer3", "res4"), ("layer4", "res5")]
    def copy_block(tv_prefix: str, d2_prefix: str):
        for k_tv, v_tv in tv_state.items():
            if not k_tv.startswith(tv_prefix):
                continue
            k_tail = k_tv[len(tv_prefix):]
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
        print(f"[WEIGHTS] Backbone initialized from torchvision. Filled: {len(remap)}/{len(d2_state)}")
    except Exception as e:
        print("[WEIGHTS] Failed to load torchvision weights into backbone:", e)

# ---------- generic cfg builder (kept; used by FCOS stage) ----------
def build_cfg(args, num_classes: int) -> Any:
    cfg = get_cfg()

    def _has_fcos():
        try:
            import adet  # noqa: F401
            from adet.config import add_fcos_config  # noqa: F401
            return "adet"
        except Exception:
            pass
        try:
            from detectron2.modeling.meta_arch import fcos  # noqa: F401
            return "d2"
        except Exception:
            return ""

    prefer_fcos = getattr(args, "prefer_fcos", True)
    fcos_src = _has_fcos()

    if prefer_fcos and fcos_src:
        cfg.merge_from_file(model_zoo.get_config_file("Base-RCNN-FPN.yaml"))
        if fcos_src == "adet":
            try:
                from adet.config import add_fcos_config
                add_fcos_config(cfg)
            except Exception:
                print("[WARN] AdelaiDet найден, но add_fcos_config не сработал. Создам MODEL.FCOS вручную.")
        if not hasattr(cfg.MODEL, "FCOS"):
            cfg.MODEL.FCOS = CN()

        cfg.MODEL.FCOS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.FCOS.INFERENCE_TH = args.infer_th
        cfg.MODEL.FCOS.NMS_TH = 0.6
        cfg.MODEL.FCOS.PRIOR_PROB = 0.01
        cfg.MODEL.FCOS.SIZES_OF_INTEREST = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
        cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
        cfg.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
        cfg.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
        cfg.MODEL.FCOS.CENTER_SAMPLE = True
        cfg.MODEL.FCOS.POS_RADIUS = 1.5
        cfg.MODEL.FCOS.YIELD_PROPOSAL = False
        if not hasattr(cfg.MODEL.FCOS, "YIELD_BOX_FEATURES"):
            cfg.MODEL.FCOS.YIELD_BOX_FEATURES = False
        if not hasattr(cfg.MODEL.FCOS, "USE_DEFORMABLE"):
            cfg.MODEL.FCOS.USE_DEFORMABLE = False
        if not hasattr(cfg.MODEL.FCOS, "DEFORMABLE_GROUPS"):
            cfg.MODEL.FCOS.DEFORMABLE_GROUPS = 1
        cfg.MODEL.FCOS.NORM = "GN"
        if not hasattr(cfg.MODEL.FCOS, "NUM_CLS_CONVS"):
            cfg.MODEL.FCOS.NUM_CLS_CONVS = 4
        if not hasattr(cfg.MODEL.FCOS, "NUM_BOX_CONVS"):
            cfg.MODEL.FCOS.NUM_BOX_CONVS = 4

        cfg.MODEL.META_ARCHITECTURE = "FCOS"
        cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
        cfg.MODEL.RESNETS.DEPTH = args.resnet_depth #<-- было 50, теперь из аргумента
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]


        try:
            _fill_missing_fcos_keys(cfg)
        except Exception:
            pass
        cfg._SELECTED_META_ARCH = "FCOS"
    else:
        retinanet_yaml = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(retinanet_yaml))
        cfg.MODEL.RESNETS.DEPTH = args.resnet_depth  # принудительно R18/R34 и т.д.
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.infer_th
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg._SELECTED_META_ARCH = "RetinaNet"

    cfg.DATASETS.TRAIN = ("psy_train",)
    cfg.DATASETS.TEST = ("psy_val",)

    cfg.DATALOADER.NUM_WORKERS = args.workers
    if hasattr(cfg.DATALOADER, "PERSISTENT_WORKERS"):
        cfg.DATALOADER.PERSISTENT_WORKERS = False
    # >>> RepeatFactorSampler включен по умолчанию <<<
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = args.repeat_threshold
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False

    cfg.INPUT.MIN_SIZE_TRAIN = (640, 720, 800, 896)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
    cfg.INPUT.MAX_SIZE_TEST = max(args.min_size_test, 1280)
    cfg.INPUT.RANDOM_FLIP = "horizontal"

    if args.device:
        cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = ""

    cfg.SOLVER.CHECKPOINT_PERIOD = 200
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.WARMUP_ITERS = min(1000, max(200, args.max_iter // 20))
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.OUTPUT_DIR = args.outdir
    safe_mkdir(Path(cfg.OUTPUT_DIR))
    return cfg

# ---------- slim cfg factories for stages ----------
def make_cfg_retinanet_r18(args, num_classes: int, outdir: str, max_iter: int,
                           ims_per_batch: int, min_size_train=(512, 640),
                           min_size_test=640, base_lr=5e-4):
    cfg = get_cfg()
    retinanet_yaml = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(retinanet_yaml))
    cfg._SELECTED_META_ARCH = "RetinaNet"
    cfg.MODEL.META_ARCHITECTURE = "RetinaNet"
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.infer_th

    cfg.MODEL.RESNETS.DEPTH = 18
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2","res3","res4","res5"]
    cfg.MODEL.FPN.IN_FEATURES = ["res2","res3","res4","res5"]

    cfg.DATASETS.TRAIN = ("psy_train",)
    cfg.DATASETS.TEST  = ("psy_val",)

    cfg.DATALOADER.NUM_WORKERS = args.workers
    if hasattr(cfg.DATALOADER, "PERSISTENT_WORKERS"): cfg.DATALOADER.PERSISTENT_WORKERS = False
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = args.repeat_threshold
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False

    cfg.INPUT.MIN_SIZE_TRAIN = tuple(min_size_train)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = min_size_test
    cfg.INPUT.MAX_SIZE_TEST = max(min_size_test, 1280)
    cfg.INPUT.RANDOM_FLIP = "horizontal"

    if args.device: cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = ""
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.WARMUP_ITERS = min(1000, max(200, max_iter//20))
    cfg.SOLVER.WARMUP_FACTOR = 1.0/1000
    cfg.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CHECKPOINT_PERIOD = 200
    cfg.TEST.EVAL_PERIOD = args.eval_period

    cfg.OUTPUT_DIR = outdir
    safe_mkdir(Path(cfg.OUTPUT_DIR))
    return cfg

def make_cfg_fcos_r34(args, num_classes: int, outdir: str, max_iter: int,
                      ims_per_batch: int, min_size_train=(640,), min_size_test=640,
                      base_lr=3e-4):
    cfg = build_cfg(args, num_classes)  # already FCOS+shim if prefer_fcos
    cfg._SELECTED_META_ARCH = "FCOS"
    cfg.MODEL.META_ARCHITECTURE = "FCOSDetectorShim"

    cfg.MODEL.RESNETS.DEPTH = 34
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2","res3","res4","res5"]
    cfg.MODEL.FPN.IN_FEATURES = ["res2","res3","res4","res5"]

    cfg.INPUT.MIN_SIZE_TRAIN = tuple(min_size_train)
    cfg.INPUT.MIN_SIZE_TEST = min_size_test
    cfg.INPUT.MAX_SIZE_TEST = max(min_size_test, 1280)

    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = args.repeat_threshold

    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.WARMUP_ITERS = min(1500, max(300, max_iter//5))
    cfg.SOLVER.CHECKPOINT_PERIOD = 200

    cfg.OUTPUT_DIR = outdir
    safe_mkdir(Path(cfg.OUTPUT_DIR))
    return cfg

# ---------- tiny predictor ----------
class DefaultPredictorLite:
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

# ---------- CLI ----------
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
    ap.add_argument("--prefer-fcos", action="store_true",
                    help="Принудительно использовать FCOS (ошибка, если недоступен)")
    ap.add_argument("--weights", default="", help="Путь к .pth для загрузки перед обучением")
    ap.add_argument("--resnet-depth", type=int, default=50, choices=[18, 34, 50, 101, 152],
                    help="Глубина ResNet бэкбона для RetinaNet/FCOS")
    ap.add_argument("--freeze-backbone-iters", type=int, default=0, help="Сколько итераций держать замороженным бэкбон (ResNet bottom_up). 0 = отключено."
    )

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
    sanity_paths(args)

    classes = register_datasets(args.train_json, args.train_img, args.val_json, args.val_img, args.test_json, args.test_img)
    num_classes = args.num_classes if args.num_classes is not None else len(classes)
    if args.num_classes is not None and args.num_classes != len(classes):
        print(f"[INFO] Overriding num_classes from {len(classes)} -> {args.num_classes}")
    print(f"[INFO] NUM_CLASSES = {num_classes}")

    # ===== ONE-CLICK PIPELINE =====
    # Итерации по умолчанию (можно менять --max-iter, влияет на Stage A):
    stageA_iters = max(6000, args.max_iter)   # RetinaNet R18
    stageB_iters = 7000                       # FCOS R34

    outA = str(Path(args.outdir) / "stageA_retinanet_r18")
    outB = str(Path(args.outdir) / "stageB_fcos_r34")

    # ---------- ЭТАП A: RetinaNet R18-FPN ----------
    cfgA = make_cfg_retinanet_r18(
        args, num_classes=num_classes, outdir=outA,
        max_iter=stageA_iters, ims_per_batch=max(1, args.ims_per_batch),
        min_size_train=(512, 640), min_size_test=args.min_size_test, base_lr=args.base_lr
    )
    setup_logger(output=outA)
    default_setup(cfgA, {})
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # если передали --weights, используем их (вместо torchvision init)
    if getattr(args, "weights", ""):
        cfgA.MODEL.WEIGHTS = args.weights

    trainerA = Trainer(cfgA)
    trainerA.resume_or_load(resume=False)
    trainerA.start_time = time.time()
    trainerA.register_hooks([
        HeartbeatHook(every=1),
        ETAHook(total_iter=cfgA.SOLVER.MAX_ITER, log_path=Path(outA)/"train_log.txt", every=20),
        PauseHook(outdir=Path(outA), poll_every=10),
        SafeCheckpointHook(outdir=Path(outA)),
        FreezeBackboneUntil(trainerA.model, until_iter=min(1500, cfgA.SOLVER.MAX_ITER//4), unfreeze_fpn=True),
    ])
    print("[PIPE] Stage A: training RetinaNet R18-FPN …", flush=True)
    try:
        trainerA.train()
    except KeyboardInterrupt:
        print("\n[PIPE][A] INTERRUPTED -> saving checkpoint …")
        try: trainerA.checkpointer.save("model_interrupted")
        except: pass
        sys.exit(130)
    except Exception:
        print("[PIPE][A] crash -> saving last checkpoint…")
        try: trainerA.checkpointer.save("model_crash")
        except: pass
        raise

    # (опционально можно вставить быструю валидацию Stage A здесь)

    # ---------- ЭТАП B: FCOS R34-FPN ----------
    # Проверим регистрацию FCOS в реестре и выберем shim
    try:
        selected = "FCOS"
        candidates = ["FCOSDetector", "FCOS"]
        resolved = None
        for name in candidates:
            try:
                META_ARCH_REGISTRY.get(name)
                resolved = name
                break
            except KeyError:
                continue
        if resolved is None:
            mod = importlib.import_module("adet.modeling.fcos.fcos")
            for cls_name in ["FCOS", "FCOSDetector"]:
                if hasattr(mod, cls_name):
                    try:
                        META_ARCH_REGISTRY.register(getattr(mod, cls_name))
                    except Exception:
                        pass
            for name in ["FCOSDetector", "FCOS"]:
                try:
                    META_ARCH_REGISTRY.get(name)
                    resolved = name
                    break
                except KeyError:
                    continue
        if resolved is None:
            raise RuntimeError("FCOS не зарегистрирован. Проверьте установку AdelaiDet.")
    except Exception:
        raise

    cfgB = make_cfg_fcos_r34(
        args, num_classes=num_classes, outdir=outB,
        max_iter=stageB_iters, ims_per_batch=max(1, args.ims_per_batch),
        min_size_train=(640,), min_size_test=args.min_size_test, base_lr=max(3e-4, args.base_lr * 0.6),
    )
    setup_logger(output=outB)
    default_setup(cfgB, {})
    # если передали --weights, используем их (вместо torchvision init)
    if getattr(args, "weights", ""):
        cfgB.MODEL.WEIGHTS = args.weights

    trainerB = Trainer(cfgB)
    trainerB.resume_or_load(resume=False)
    trainerB.start_time = time.time()
    trainerB.register_hooks([
        # 0) Авто-заморозка/разморозка бэкбона + пересборка оптимизатора
        FreezeBackboneHook(iters=getattr(args, "freeze_backbone_iters", 0)),

        # 1) Пульс-лог (каждый шаг)
        HeartbeatHook(every=1),

        # 2) Прогноз ETA + лог в файл
        ETAHook(total_iter=cfgB.SOLVER.MAX_ITER,
                log_path=Path(outB) / "train_log.txt",
                every=20),

        # 3) Пауза по флагу
        PauseHook(outdir=Path(cfgB.OUTPUT_DIR), poll_every=10),

        # 4) Безопасный финальный чекпоинт
        SafeCheckpointHook(outdir=Path(cfgB.OUTPUT_DIR)),
    ])

    print("[PIPE] Stage B: training FCOS R34-FPN …", flush=True)
    try:
        trainerB.train()
        if args.freeze_backbone_iters > 0:
            print(
                f"[CFG] Freeze backbone for the first {args.freeze_backbone_iters} iters → then unfreeze & rebuild optimizer.",
                flush=True)

    except KeyboardInterrupt:
        print("\n[PIPE][B] INTERRUPTED -> saving checkpoint …")
        try: trainerB.checkpointer.save("model_interrupted")
        except: pass
        sys.exit(130)
    except Exception:
        print("[PIPE][B] crash -> saving last checkpoint…")
        try: trainerB.checkpointer.save("model_crash")
        except: pass
        raise

    # ---------- Валидация (финальная модель = Stage B) ----------
    evaluator = COCOEvaluator("psy_val", cfgB, True, output_dir=os.path.join(cfgB.OUTPUT_DIR, "inference", "psy_val"))
    val_loader = build_detection_test_loader(cfgB, "psy_val")
    results = inference_on_dataset(trainerB.model, val_loader, evaluator)
    print("\n[VAL] COCO metrics (Stage B):\n", results)

    # ---------- Тест ----------
    evaluator_te = COCOEvaluator("psy_test", cfgB, True, output_dir=os.path.join(cfgB.OUTPUT_DIR, "inference", "psy_test"))
    test_loader = build_detection_test_loader(cfgB, "psy_test")
    results_te = inference_on_dataset(trainerB.model, test_loader, evaluator_te)
    print("\n[TEST] COCO metrics (Stage B):\n", results_te)

    # ---------- Замер P95 / FPS ----------
    final_path = Path(cfgB.OUTPUT_DIR) / "model_final.pth"
    if final_path.exists():
        cfgB.MODEL.WEIGHTS = str(final_path)
    else:
        last_safe = Path(cfgB.OUTPUT_DIR) / "model_final_safe.pth"
        if last_safe.exists():
            cfgB.MODEL.WEIGHTS = str(last_safe)

    predictor = DefaultPredictorLite(cfgB)
    few_imgs = list_some_images(args.val_img, k=30)
    p95_ms, fps = measure_latency_fps(predictor, few_imgs, warmup=20, runs=200)
    print(f"\n[Runtime] P95 latency: {p95_ms:.1f} ms; FPS(mean): {fps:.1f}  @ {args.min_size_test}p on {cfgB.MODEL.DEVICE}")

    try:
        with (Path(cfgB.OUTPUT_DIR) / "runtime_report.txt").open("w", encoding="utf-8") as f:
            f.write(f"P95_ms={p95_ms:.2f}\nFPS_mean={fps:.2f}\nMIN_SIZE_TEST={args.min_size_test}\nDEVICE={cfgB.MODEL.DEVICE}\n")
    except Exception:
        pass

    print("\nDone. Artifacts:")
    print("  Stage A:", outA)
    print("  Stage B:", outB)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# проверяет работу аугументации на датасете, смотрит как обсчитываются боксы
"""
Запуск
python R-CNN/FCOS/tools/sanity_dataset.py \
  --train-json /home/user/mrgv2/tiny50/tiny_train.json \
  --train-img  /home/user/mrgv2/tiny50 \
  --limit 40 \
  --save-vis /home/user/mrgv2/exp_fcos_tiny50/sanity_vis
"""
import os
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

# --- Pillow shim: совместимость с detectron2, где ждут Image.LINEAR
try:
    from PIL import Image as _PIL_Image
    if not hasattr(_PIL_Image, "LINEAR"):
        if hasattr(_PIL_Image, "BILINEAR"):
            _PIL_Image.LINEAR = _PIL_Image.BILINEAR
        else:
            # очень старые Pillow: подстрахуемся через Resampling
            from PIL.Image import Resampling as _Resampling
            _PIL_Image.LINEAR = _Resampling.BILINEAR
except Exception:
    pass


from detectron2.data.datasets import load_coco_json
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode, Boxes

# ---------- аугментации (щадящие) ----------
def build_augs():
    return [
        T.RandomFlip(horizontal=True, vertical=False),
        T.RandomBrightness(0.95, 1.05),
        T.RandomContrast(0.95, 1.05),
        T.RandomSaturation(0.95, 1.05),
    ]

# ---------- маппер в духе вашего train_fcos.py ----------
def mapper_with_augs_like_train(dataset_dict, image_root):
    """
    Возвращает:
      image_aug (H,W,3) BGR,
      inst_boxes_xyxy (Nx4, float32),
      inst_classes (N,)
    """
    rec = dataset_dict
    # абсолютный путь
    img_path = rec["file_name"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(image_root, img_path)

    image = utils.read_image(img_path, format="BGR")
    H0, W0 = image.shape[:2]

    augs = build_augs()
    aug_input = T.AugInput(image)
    tfms = T.AugmentationList(augs)(aug_input)
    image_aug = aug_input.image
    H, W = image_aug.shape[:2]

    annos_raw = rec.get("annotations", [])
    annos = []
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
        # чуть щадящая отсечка очень маленьких боксов
        if (x1 - x0) < 0.5 or (y1 - y0) < 0.5:
            continue

        a["bbox"] = [x0, y0, x1, y1]
        a["bbox_mode"] = BoxMode.XYXY_ABS
        annos.append(a)

    # соберём тензоры
    if len(annos):
        boxes = np.array([a["bbox"] for a in annos], dtype=np.float32)
        classes = np.array([a.get("category_id", 0) for a in annos], dtype=np.int64)
        # клип в границы
        b = Boxes(torch.as_tensor(boxes))
        b.clip((H, W))
        boxes = b.tensor.cpu().numpy()
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.int64)

    return image_aug, boxes, classes, (H0, W0), (H, W)

# ---------- визуализация ----------
def draw_boxes(img_bgr, boxes_xyxy, color=(0,255,0), thickness=2):
    img = img_bgr.copy()
    for x0, y0, x1, y1 in boxes_xyxy.astype(int):
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    return img

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True, help="COCO json для train")
    ap.add_argument("--train-img", required=True, help="Папка с изображениями")
    ap.add_argument("--limit", type=int, default=50, help="Сколько примеров проверить")
    ap.add_argument("--save-vis", type=str, default="", help="Папка для сохранения визуализаций (опционально)")
    args = ap.parse_args()

    json_path = Path(args.train_json)
    img_root = Path(args.train_img)
    assert json_path.is_file(), f"no file: {json_path}"
    assert img_root.is_dir(), f"no dir: {img_root}"

    records = load_coco_json(str(json_path), str(img_root), dataset_name="sanity_tmp")
    print(f"[SANITY] loaded {len(records)} records")

    if args.save_vis:
        outdir = Path(args.save_vis)
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[SANITY] will save visualizations to: {outdir.resolve()}")

    n = min(args.limit, len(records))
    empty_after = 0
    empty_before = 0
    total_before = 0
    total_after = 0

    for i, rec in enumerate(records[:n]):
        # до аугментаций
        n_before = len(rec.get("annotations", []) or [])
        total_before += n_before
        if n_before == 0:
            empty_before += 1

        # после аугментаций
        img_aug, boxes, classes, sz0, sz = mapper_with_augs_like_train(rec, str(img_root))
        n_after = boxes.shape[0]
        total_after += n_after
        if n_after == 0:
            empty_after += 1

        # печать краткой строки
        print(f"[{i:03d}] before={n_before:2d}  after={n_after:2d}  "
              f"orig={sz0[0]}x{sz0[1]} -> aug={sz[0]}x{sz[1]}  file={rec['file_name']}")

        # опциональная визуализация
        if args.save_vis:
            # до
            img_path = rec["file_name"]
            if not os.path.isabs(img_path):
                img_path = str(Path(args.train_img) / img_path)
            img0 = cv2.imread(img_path)  # BGR
            boxes0 = []
            for a in rec.get("annotations", []) or []:
                if "bbox" in a:
                    bb = np.array(a["bbox"], dtype=np.float32)
                    bm = a.get("bbox_mode", BoxMode.XYWH_ABS)
                    bb = BoxMode.convert(bb, bm, BoxMode.XYXY_ABS)
                    boxes0.append(bb)
            boxes0 = np.array(boxes0, dtype=np.float32) if len(boxes0) else np.zeros((0,4), np.float32)
            vis0 = draw_boxes(img0, boxes0, color=(0, 128, 255))
            vis1 = draw_boxes(img_aug, boxes, color=(0, 255, 0))
            cv2.imwrite(str(Path(args.save_vis) / f"{i:03d}_before.jpg"), vis0)
            cv2.imwrite(str(Path(args.save_vis) / f"{i:03d}_after.jpg"),  vis1)

    print("\n[SANITY] -------- summary --------")
    print(f"checked: {n} images")
    print(f"empty BEFORE augs: {empty_before}/{n}")
    print(f"empty AFTER  augs: {empty_after}/{n}")
    print(f"total boxes BEFORE: {total_before}")
    print(f"total boxes AFTER : {total_after}")
    if total_after == 0:
        print("⚠️  после аугментаций исчезают все боксы — либо аугментации агрессивны, либо json/пути/box-mode некорректны.")

if __name__ == "__main__":
    main()

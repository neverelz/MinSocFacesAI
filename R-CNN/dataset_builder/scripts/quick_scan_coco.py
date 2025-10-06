#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрый аудит _extracted: находит сплиты, аннотации, считает images/annotations,
показывает существование файлов и список категорий.
"""
import os, json, sys
from pathlib import Path

ANN_CAND = [
    "annotations.coco.json","_annotations.coco.json",
    "annotation.coco.json","anotation.coco.json","coco.json"
]
SPLITS = {"train","val","validation","valid","test"}
IMG_EXT = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff",".gif"}

def find_ann(split_dir: Path):
    for n in ANN_CAND:
        p = split_dir / n
        if p.is_file():
            return p
    ann_dir = split_dir / "annotations"
    if ann_dir.is_dir():
        # возьмём первый json
        for q in ann_dir.glob("*.json"):
            return q
    return None

def is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMG_EXT

def main(root):
    root = Path(root).expanduser().absolute()
    any_found = False
    print("DS_KEY\tSPLIT\tANN_JSON\tIMAGES(total)\tANNOTATIONS(total)\tIMAGES(existing)\tCATEGORIES(sample)")
    for ds_dir in sorted(root.rglob("*")):
        if not ds_dir.is_dir():
            continue
        # формат ключа, как в merge: parent/name
        ds_key = f"{ds_dir.parent.name}/{ds_dir.name}"
        for split in SPLITS:
            sd = ds_dir / split
            if not sd.is_dir():
                continue
            ann = find_ann(sd)
            if not ann:
                continue
            any_found = True
            try:
                data = json.loads(ann.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"{ds_key}\t{split}\t{ann}\tERR\tERR\tERR\tERR (read fail: {e})")
                continue
            imgs = data.get("images", [])
            anns = data.get("annotations", [])
            cats = [c.get("name", str(c.get('id'))) for c in data.get("categories", [])]
            cats_show = ",".join(cats[:5])

            # где картинки
            img_dir = sd / "images"
            if not img_dir.is_dir():
                img_dir = sd

            exist_count = 0
            for im in imgs[:1000]:  # ускорение: до 1000
                fn = Path(im.get("file_name","")).name
                if not is_image_name(fn):
                    continue
                if (img_dir / fn).exists():
                    exist_count += 1

            print(f"{ds_key}\t{split}\t{ann}\t{len(imgs)}\t{len(anns)}\t{exist_count}\t{cats_show}")

    if not any_found:
        print("\n[!] Не найдено ни одного сплита/аннотаций. Проверь структуру папок в --extracted.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_scan_coco.py /path/to/_extracted")
        sys.exit(1)
    main(sys.argv[1])

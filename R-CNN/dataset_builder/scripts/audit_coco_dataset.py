#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_coco_dataset.py — полный аудит COCO-датасета (train/val/test).

Пример:
  python audit_coco_dataset.py --root /home/user/mrg --split-names train val test
  # если у тебя val называется 'valid':
  python audit_coco_dataset.py --root /home/user/mrg --split-names train valid test
"""

import argparse, json, os, math, csv, statistics as stats
from pathlib import Path

ANN_NAMES = [
    "annotations.coco.json",
    "_annotations.coco.json",
    "annotation.coco.json",
    "anotation.coco.json",
    "coco.json",
]

IMG_EXT = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff",".gif"}

def is_img_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMG_EXT

def read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def find_ann_file(split_dir: Path) -> Path|None:
    # ищем в split/
    for n in ANN_NAMES:
        p = split_dir / n
        if p.is_file():
            return p
    # ищем в split/annotations/*.json
    ann_dir = split_dir / "annotations"
    if ann_dir.is_dir():
        for q in sorted(ann_dir.glob("*.json")):
            return q
    return None

def list_basenames_safe(img_dir: Path) -> set[str]:
    # только basename — не конструируем длинных абсолютных путей
    try:
        return set(os.listdir(str(img_dir)))
    except Exception:
        return set()

def safe_join_exists(img_dir: Path, basename: str) -> bool:
    # проверяем существование по списку имен, без os.stat длинного пути
    return basename in list_basenames_safe(img_dir)

def write_csv(out: Path, header: list[str], rows: list[list]):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def audit_split(root: Path, split: str, reports_dir: Path, small_frac: float = 0.0002):
    split_dir = root / split
    if not split_dir.is_dir():
        print(f"[WARN] split '{split}' not found at {split_dir}")
        return

    ann_file = find_ann_file(split_dir)
    if not ann_file:
        print(f"[WARN] annotations not found in {split_dir}")
        return

    data = read_json(ann_file)
    images = data.get("images", []) or []
    anns = data.get("annotations", []) or []
    cats = data.get("categories", []) or []

    # где лежат картинки
    img_dir = split_dir / "images" if (split_dir / "images").is_dir() else split_dir
    present = list_basenames_safe(img_dir)

    id2img = {im["id"]: im for im in images}
    id2cat = {c["id"]: c.get("name", str(c["id"])) for c in cats}
    cat_name2count = {}
    per_image_anncnt = {}
    missing_files = []
    orphan_anns = []
    images_without_anns = []
    bbox_issues = []   # (image_id, ann_id, reason, bbox, img_w, img_h)
    seg_stats = {"with_seg": 0, "without_seg": 0}

    # существование картинок
    exist_count = 0
    for im in images:
        fn = Path(im.get("file_name","")).name
        if fn in present:
            exist_count += 1
        else:
            missing_files.append([split, im.get("id"), fn])

    # «висячие» аннотации и подсчёты по классам
    for a in anns:
        iid = a.get("image_id")
        if iid not in id2img:
            orphan_anns.append([split, a.get("id"), iid, a.get("category_id")])
            continue
        per_image_anncnt[iid] = per_image_anncnt.get(iid, 0) + 1
        cname = id2cat.get(a.get("category_id"), str(a.get("category_id")))
        cat_name2count[cname] = cat_name2count.get(cname, 0) + 1

        # сегментация (для информации, даже если это detection)
        seg = a.get("segmentation")
        if seg:
            seg_stats["with_seg"] += 1
        else:
            seg_stats["without_seg"] += 1

        # bbox sanity
        bbox = a.get("bbox")
        imeta = id2img[iid]
        w = imeta.get("width") or 0
        h = imeta.get("height") or 0
        if not bbox or len(bbox) != 4:
            bbox_issues.append([split, iid, a.get("id"), "bbox_missing_or_bad", bbox, w, h])
        else:
            x, y, bw, bh = bbox
            issue = None
            if bw is None or bh is None:
                issue = "bbox_none_size"
            elif bw <= 0 or bh <= 0:
                issue = "bbox_nonpositive"
            elif w and (x < -1 or y < -1 or x + bw > w + 1 or y + bh > h + 1):
                issue = "bbox_out_of_bounds"
            # очень маленькие боксы (по площади относительно изображения)
            if not issue and w and h:
                frac = (bw * bh) / float(w * h)
                if frac < small_frac:
                    issue = f"bbox_too_small(<{small_frac:.5f})"
            if issue:
                bbox_issues.append([split, iid, a.get("id"), issue, bbox, w, h])

    # изображения без аннотаций
    images_with_anns = set(per_image_anncnt.keys())
    for im in images:
        if im["id"] not in images_with_anns:
            images_without_anns.append([split, im["id"], Path(im.get("file_name","")).name])

    # сводка
    n_img = len(images)
    n_ann = len(anns)
    n_cat = len(cats)
    n_exist = exist_count
    n_missing = len(missing_files)
    n_orphan = len(orphan_anns)
    n_noann = len(images_without_anns)
    n_bbox_bad = len(bbox_issues)

    # пер-картинке статистика
    img_ann_counts = list(per_image_anncnt.values()) or [0]
    min_ann = min(img_ann_counts)
    max_ann = max(img_ann_counts)
    mean_ann = stats.mean(img_ann_counts)
    median_ann = stats.median(img_ann_counts)

    # частоты классов (CSV)
    class_rows = [[split, name, cnt] for name, cnt in sorted(cat_name2count.items(), key=lambda kv: (-kv[1], kv[0]))]
    write_csv(reports_dir / f"{split}_class_counts.csv", ["split","class","count"], class_rows)

    # отсутствующие файлы
    write_csv(reports_dir / f"{split}_missing_files.csv", ["split","image_id","file_name"], missing_files)

    # висячие аннотации
    write_csv(reports_dir / f"{split}_orphan_annotations.csv", ["split","ann_id","image_id","category_id"], orphan_anns)

    # изображения без аннотаций
    write_csv(reports_dir / f"{split}_images_without_annotations.csv", ["split","image_id","file_name"], images_without_anns)

    # проблемы bbox
    write_csv(reports_dir / f"{split}_bbox_issues.csv",
              ["split","image_id","ann_id","issue","bbox","img_w","img_h"], bbox_issues)

    # пер-изображению
    per_img_rows = [[split, iid, per_image_anncnt.get(iid, 0)] for iid in sorted(id2img.keys())]
    write_csv(reports_dir / f"{split}_per_image_ann_counts.csv", ["split","image_id","ann_count"], per_img_rows)

    # печать краткой сводки
    print(f"\n[{split}] {ann_file}")
    print(f" images: {n_img} (exist on disk: {n_exist}, missing: {n_missing})")
    print(f" annotations: {n_ann} (orphan: {n_orphan}, bbox_issues: {n_bbox_bad})")
    print(f" categories: {n_cat}")
    print(f" images without annotations: {n_noann}")
    print(f" per-image ann count: min={min_ann} mean={mean_ann:.2f} median={median_ann} max={max_ann}")
    print(f" segmentation presence: with={seg_stats['with_seg']} without={seg_stats['without_seg']}")
    print(f" reports: {reports_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Корень датасета (с подпапками train/ val|valid|validation/ test)")
    ap.add_argument("--split-names", nargs="+", default=["train","val","test"], help="Имена сплитов, напр. train val test")
    ap.add_argument("--small-box-frac", type=float, default=0.0002, help="Порог доли площади маленького bbox относительно изображения")
    args = ap.parse_args()

    root = Path(args.root).expanduser().absolute()
    reports_dir = root / "reports_audit"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # совместимость: если пользователь указал 'val', но папка называется 'valid'/'validation'
    split_alias = {
        "val": ["val","valid","validation"],
        "valid": ["valid","validation","val"],
        "validation": ["validation","valid","val"]
    }

    # прогон по каждому указанному сплиту (с поиском синонимов)
    for sp in args.split_names:
        candidates = [sp]
        if sp in split_alias:
            candidates = split_alias[sp]
        found = None
        for c in candidates:
            if (root / c).is_dir():
                found = c
                break
        if not found:
            print(f"[WARN] split '{sp}' not found (checked: {candidates})")
            continue
        audit_split(root, found, reports_dir, small_frac=args.small_box_frac)

if __name__ == "__main__":
    main()

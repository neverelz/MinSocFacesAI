# -*- coding: utf-8 -*-
import json, hashlib, re, os, shutil
from pathlib import Path
import re

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

def md5sum(p: Path, chunk=1<<20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for piece in iter(lambda: f.read(chunk), b""):
            h.update(piece)
    return h.hexdigest()

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-zА-Яа-я0-9_\-]+", "_", s)

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

def find_annotation_json(split_dir: Path):
    candidates = [
        "anotation.coco.json", "annotation.coco.json",
        "_annotations.coco.json", "annotations.coco.json", "coco.json"
    ]
    for name in candidates:
        p = split_dir / name
        if p.exists():
            return p
    for p in split_dir.glob("*.coco.json"):
        return p
    ann_dir = split_dir / "annotations"
    if ann_dir.is_dir():
        for p in ann_dir.glob("*.json"):
            if "coco" in p.name.lower() or "annot" in p.name.lower():
                return p
    return None

def detect_split_dirs(root: Path):
    # поддерживаем train / val|valid|validation / test
    res = {"train": None, "val": None, "test": None}
    # ищем на глубину 2-3
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        n = p.name.lower()
        if n == "train": res["train"] = p
        if n in ("val", "valid", "validation"): res["val"] = p
        if n == "test": res["test"] = p
    return res

def ensure_image_wh(img_path: Path, im_meta: dict):
    w = im_meta.get("width"); h = im_meta.get("height")
    if (w is None or h is None) and PIL_OK and img_path.exists():
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                im_meta["width"], im_meta["height"] = im.size
        except Exception:
            pass

def hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def canon(s: str) -> str:
    """Привести имя класса к канону: lowercase, пробелы/дефисы -> '_', убрать нестандарт."""
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)       # пробелы и дефисы -> _
    s = re.sub(r"_+", "_", s)            # схлопнуть повторные _
    s = re.sub(r"[^a-z0-9_]", "", s)     # убрать экзотику
    return s

def normalize_class_map(raw_map: dict) -> dict:
    """
    Нормализуем ключи в class_map.json для сопоставления по имени.
    В секциях оставляем как есть (кроме лишних слешей), а правила нормализуем по canon(key).
    """
    nm = {}
    for section, rules in raw_map.items():
        sec = section.strip().strip("/")  # "__global__" оставим как есть
        if section == "__global__": sec = "__global__"
        nm[sec] = {}
        for k, v in (rules or {}).items():
            nm[sec][canon(str(k))] = v  # ключи приводим к канону
    return nm

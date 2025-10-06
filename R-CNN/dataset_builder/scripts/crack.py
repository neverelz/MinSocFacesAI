#!/usr/bin/env python3
# shrink_filenames_in_place.py
import hashlib, json, re, shutil
from pathlib import Path

MAX_BASENAME_LEN = 128

def sha10(s): return hashlib.sha1(s.encode("utf-8", 'ignore')).hexdigest()[:10]

def short_name(old):
    p = Path(old); stem = p.stem; suf = p.suffix
    h = sha10(old)
    keep = max(1, MAX_BASENAME_LEN - len(h) - len(suf) - 2)
    return f"{stem[:keep]}_{h}{suf}"

def find_ann(split_dir: Path):
    for name in ["annotations.coco.json","_annotations.coco.json","annotation.coco.json","anotation.coco.json","coco.json"]:
        p = split_dir / name
        if p.exists(): return p
    for p in (split_dir/"annotations").glob("*.json"):
        return p
    return None

root = Path("/home/user/_extracted")  # <-- ПУТЬ К ВАШЕМУ _extracted
count = 0

for ds in sorted(root.rglob("*")):
    if not ds.is_dir(): continue
    for split in ("train","val","validation","valid","test"):
        sd = ds / split
        if not sd.is_dir(): continue

        ann = find_ann(sd)
        if not ann: continue
        data = json.loads(ann.read_text(encoding="utf-8"))

        # где лежат картинки
        img_dir = sd / "images"
        if not img_dir.is_dir(): img_dir = sd

        # карта переименований
        ren = {}
        for im in data.get("images", []):
            old = im.get("file_name","")
            # файл может быть с подпапкой в имени — возьмём только basename для реального файла
            fn = Path(old).name
            new = short_name(fn)
            if new == fn: continue
            src = img_dir / fn
            if src.exists():
                dst = img_dir / new
                i=1
                while dst.exists():
                    dst = img_dir / f"{Path(new).stem}_{i}{Path(new).suffix}"
                    i+=1
                src.rename(dst)
                ren[fn] = dst.name

        if ren:
            for im in data.get("images", []):
                fn = Path(im.get("file_name","")).name
                if fn in ren:
                    im["file_name"] = ren[fn]
            ann.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            count += len(ren)

print("Renamed files:", count)

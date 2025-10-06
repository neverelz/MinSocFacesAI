#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_coco.py (safe-open) — слияние COCO-датасетов без обращения к длинным путям файлов.
Копирование исходных изображений выполняется через openat (dir_fd) по basename.

Пример:
  python -u merge_coco.py \
    --extracted /home/user/_extracted \
    --out /home/user/mrg \
    --mode unified --split 0.8 0.1 0.1 \
    --map R-CNN/dataset_builder/configs/class_map.json \
    --dedup --task detection --short-names --verbose
"""
import argparse, hashlib, json, os, random, sys
from collections import Counter
from pathlib import Path
import shutil

# ====== utils you already have in project; keep imports same ======
from utils_common import (
    load_json, save_json, detect_split_dirs, find_annotation_json, sanitize,
    md5sum, ensure_image_wh, hardlink_or_copy, canon, normalize_class_map
)

MAX_BASENAME_LEN = 128

def short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8","ignore")).hexdigest()[:n]

def make_safe_name(basename: str, prefix: str, use_short: bool) -> str:
    stem = Path(basename).stem
    suf  = Path(basename).suffix
    if use_short:
        h = short_hash(prefix + "::" + basename)
        keep = max(1, MAX_BASENAME_LEN - len(prefix) - len(h) - len(suf) - 6)
        return f"{prefix}__{stem[:keep]}__{h}{suf}"
    name = f"{prefix}__{basename}"
    if len(name) > MAX_BASENAME_LEN:
        h = short_hash(prefix + "::" + basename)
        keep = max(1, MAX_BASENAME_LEN - len(prefix) - len(h) - len(suf) - 6)
        name = f"{prefix}__{stem[:keep]}__{h}{suf}"
    return name

# ---------- FD helpers (avoid long absolute paths on source side) ----------
def copy_from_dirfd(img_dir: Path, basename: str, dst_path: Path):
    """
    Копируем файл basename из каталога img_dir в dst_path, используя openat.
    Не создаём длинных путей к источнику вообще.
    """
    # open dir
    dirfd = os.open(str(img_dir), os.O_RDONLY)
    try:
        srcfd = os.open(basename, os.O_RDONLY, dir_fd=dirfd)
        try:
            with os.fdopen(srcfd, "rb", closefd=False) as src, open(dst_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024*1024)
        finally:
            os.close(srcfd)
    finally:
        os.close(dirfd)

def list_basenames(img_dir: Path):
    """Безопасно получить список имён в каталоге img_dir (basename'ы)."""
    return os.listdir(str(img_dir))

# ------------------------------ debug logger ------------------------------ #
class DebugLog:
    def __init__(self, root: Path, enable: bool):
        self.enable = enable
        self.rows = []
        self.root = root
        if enable:
            (root / "debug").mkdir(parents=True, exist_ok=True)
    def add(self, **kwargs):
        if self.enable:
            self.rows.append(kwargs)
    def flush(self):
        if not self.enable:
            return None
        out = self.root / "debug" / "merge_debug.tsv"
        with out.open("w", encoding="utf-8") as f:
            keys = ["ds_key","split","image_id","orig_file_name","mapped_file_name","orig_class","mapped_class","reason"]
            f.write("\t".join(keys) + "\n")
            for r in self.rows[:2000]:
                f.write("\t".join(str(r.get(k,"")) for k in keys) + "\n")
        return out

# ------------------------------- mapping utils ---------------------------- #
def _lookup_scoped(cmap: dict, ds_key: str, key_can: str):
    if ds_key in cmap and key_can in cmap[ds_key]:
        return cmap[ds_key][key_can]
    for sec in cmap:
        if sec == "__global__": continue
        if ds_key.endswith(sec) and key_can in cmap[sec]:
            return cmap[sec][key_can]
    if "__global__" in cmap and key_can in cmap["__global__"]:
        return cmap["__global__"][key_can]
    return None

def apply_class_map_scoped(src_name, src_id, ds_key: str, cmap: dict):
    name_can = canon(str(src_name))
    m = _lookup_scoped(cmap, ds_key, name_can)
    if m is not None: return m
    id_can = canon(str(src_id))
    m = _lookup_scoped(cmap, ds_key, id_can)
    if m is not None: return m
    if name_can.endswith("s"):
        m = _lookup_scoped(cmap, ds_key, name_can[:-1])
        if m is not None: return m
    return src_name

# ---------------------------- core merge function -------------------------- #
def merge_datasets(datasets: list, out_split_dir: Path, class_map: dict, dedup: bool,
                   task: str, use_short_names: bool, dbg: DebugLog, verbose: bool):
    out_images = out_split_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    out_ann = out_split_dir / "annotations.coco.json"

    final = {"images": [], "annotations": [], "categories": [], "info": {"description": "merged"}, "licenses": []}
    catname2id = {}
    next_img_id = 1
    next_ann_id = 1
    seen_hash = {}     # md5 -> {"image_id": int, "file_name": str}
    used_names = set()
    stats_before = Counter()
    stats_after  = Counter()

    src_total_imgs = 0
    src_total_anns = 0
    copied_imgs    = 0

    for ds in datasets:
        img_dir: Path = ds["img_dir"]
        ann_path: Path = ds["ann_path"]
        prefix: str = ds["prefix"]
        ds_key: str = ds["ds_key"]
        split  : str = ds.get("split","?")

        if not ann_path or not ann_path.exists():
            if verbose: print(f"[WARN] no ann json: {ann_path}")
            continue

        try:
            cj = load_json(ann_path)
        except Exception as e:
            if verbose: print(f"[WARN] unreadable ann: {ann_path} :: {e}")
            continue

        id2name = {c["id"]: c.get("name", str(c["id"])) for c in cj.get("categories", [])}
        images_dict = {im["id"]: im for im in cj.get("images", [])}
        anns = cj.get("annotations", [])
        src_total_imgs += len(images_dict)
        src_total_anns += len(anns)

        # где лежат картинки: split/images или сам split
        real_img_dir = img_dir / "images" if (img_dir / "images").is_dir() else img_dir
        try:
            present_names = set(list_basenames(real_img_dir))
        except Exception as e:
            if verbose: print(f"[WARN] cannot list images dir {real_img_dir}: {e}")
            present_names = set()

        used_image_ids = set(a["image_id"] for a in anns)
        imgid_old2new = {}

        for old_img_id in used_image_ids:
            im = images_dict.get(old_img_id)
            if not im:
                dbg.add(ds_key=ds_key, split=split, image_id=old_img_id,
                        orig_file_name="", mapped_file_name="", orig_class="", mapped_class="", reason="image_meta_missing")
                continue

            # берём только basename из COCO
            orig_fn = Path(im.get("file_name","")).name

            if orig_fn not in present_names:
                # файл отсутствует в каталоге
                dbg.add(ds_key=ds_key, split=split, image_id=old_img_id,
                        orig_file_name=orig_fn, mapped_file_name="", orig_class="", mapped_class="", reason="image_missing_in_dir")
                continue

            # dedup по md5 — считаем хэш через dirfd чтение
            if dedup:
                # посчитаем md5, читая файл через dirfd
                dirfd = os.open(str(real_img_dir), os.O_RDONLY)
                try:
                    fd = os.open(orig_fn, os.O_RDONLY, dir_fd=dirfd)
                    try:
                        h = hashlib.md5()
                        with os.fdopen(fd, "rb", closefd=False) as fsrc:
                            for chunk in iter(lambda: fsrc.read(1024*1024), b""):
                                h.update(chunk)
                        file_md5 = h.hexdigest()
                    finally:
                        os.close(fd)
                finally:
                    os.close(dirfd)
                if file_md5 in seen_hash:
                    imgid_old2new[old_img_id] = seen_hash[file_md5]["image_id"]
                    continue
            else:
                file_md5 = None

            # генерируем безопасное имя назначения
            new_name = make_safe_name(orig_fn, prefix, use_short_names)
            i = 1
            while (out_images / new_name).exists() or new_name in used_names:
                stem = Path(new_name).stem
                suf  = Path(new_name).suffix
                new_name = f"{stem}_{i}{suf}"
                i += 1

            # копируем через openat
            dst_path = out_images / new_name
            try:
                copy_from_dirfd(real_img_dir, orig_fn, dst_path)
            except Exception as e:
                dbg.add(ds_key=ds_key, split=split, image_id=old_img_id,
                        orig_file_name=orig_fn, mapped_file_name=new_name, orig_class="", mapped_class="",
                        reason=f"copy_failed:{e}")
                continue

            # регистрируем изображение
            new_im = {
                "id": next_img_id,
                "file_name": new_name,
                "width": im.get("width"),
                "height": im.get("height"),
                "license": im.get("license"),
                "date_captured": im.get("date_captured")
            }
            ensure_image_wh(dst_path, new_im)
            final["images"].append(new_im)
            imgid_old2new[old_img_id] = next_img_id
            if file_md5:
                seen_hash[file_md5] = {"image_id": next_img_id, "file_name": new_name}
            used_names.add(new_name)
            next_img_id += 1
            copied_imgs += 1

        # теперь переносим аннотации, только для скопированных images
        for ann in anns:
            old_img_id = ann.get("image_id")
            if old_img_id not in imgid_old2new:
                continue

            src_id = ann["category_id"]
            src_name = id2name.get(src_id, str(src_id))
            stats_before[src_name] += 1

            mapped = apply_class_map_scoped(src_name, src_id, ds_key, class_map)
            if mapped == "__ignore__":
                dbg.add(ds_key=ds_key, split=split, image_id=old_img_id,
                        orig_file_name="", mapped_file_name="", orig_class=src_name, mapped_class="__ignore__",
                        reason="mapped_to_ignore")
                continue

            if mapped not in catname2id:
                catname2id[mapped] = len(catname2id) + 1
                final["categories"].append({"id": catname2id[mapped], "name": mapped, "supercategory": ""})

            if task == "seg":
                seg = ann.get("segmentation")
                if not seg:
                    dbg.add(ds_key=ds_key, split=split, image_id=old_img_id,
                            orig_file_name="", mapped_file_name="", orig_class=src_name, mapped_class=mapped,
                            reason="no_segmentation_for_seg_task")
                    continue
            else:
                seg = None

            new_ann = {
                "id": next_ann_id,
                "image_id": imgid_old2new[old_img_id],
                "category_id": catname2id[mapped],
                "bbox": ann.get("bbox"),
                "area": ann.get("area"),
                "iscrowd": ann.get("iscrowd", 0),
            }
            if seg is not None:
                new_ann["segmentation"] = seg

            final["annotations"].append(new_ann)
            stats_after[mapped] += 1
            next_ann_id += 1

    save_json(out_ann, final)

    if verbose:
        print(f"[INFO] source_images:{src_total_imgs} source_annotations:{src_total_anns} "
              f"copied_images:{len(final['images'])} final_annotations:{len(final['annotations'])}")

    return stats_before, stats_after, out_ann, out_images, final

# ------------------------------- build modes ------------------------------- #
def collect_split_items(ds_root: Path):
    res = {}
    splits = detect_split_dirs(ds_root)
    for split, split_dir in splits.items():
        if not split_dir:
            continue
        ann = find_annotation_json(split_dir)
        if not ann or not ann.exists():
            continue
        img_dir = split_dir  # ВАЖНО: реальное расположение выясняем внутри merge (images/ или сам split)
        res[split] = (img_dir, ann)
    return res

def build_by_split(extracted_root: Path, out_dir: Path, class_map: dict, dedup: bool,
                   task: str, use_short_names: bool, dbg: DebugLog, verbose: bool):
    buckets = {"train": [], "val": [], "test": []}
    for ds_dir in sorted(extracted_root.rglob("*")):
        if not ds_dir.is_dir():
            continue
        splits = collect_split_items(ds_dir)
        if not any(splits.values()):
            continue
        prefix = f"{sanitize(ds_dir.parent.name)}__{sanitize(ds_dir.name)}"
        ds_key = f"{ds_dir.parent.name}/{ds_dir.name}"
        for split, (img_dir, ann) in splits.items():
            buckets[split].append({"img_dir": img_dir, "ann_path": ann, "prefix": prefix, "ds_key": ds_key, "split": split})

    reports = {}
    for split in ["train", "val", "test"]:
        if not buckets[split]:
            continue
        before, after, ann_out, imgs_out, final = merge_datasets(
            buckets[split], out_dir / split, class_map, dedup, task, use_short_names, dbg, verbose
        )
        if len(final["annotations"]) == 0 and verbose:
            print(f"[WARN] no annotations produced for split {split}")
        reports[split] = (before, after, ann_out, imgs_out, final)
    return reports

def build_unified(extracted_root: Path, out_dir: Path, class_map: dict, ratios,
                  dedup: bool, task: str, use_short_names: bool, dbg: DebugLog, verbose: bool):
    tmp_all = out_dir / "_all"
    tmp_all.mkdir(parents=True, exist_ok=True)

    datasets = []
    for ds_dir in sorted(extracted_root.rglob("*")):
        if not ds_dir.is_dir():
            continue
        splits = collect_split_items(ds_dir)
        if not any(splits.values()):
            continue
        prefix = f"{sanitize(ds_dir.parent.name)}__{sanitize(ds_dir.name)}"
        ds_key = f"{ds_dir.parent.name}/{ds_dir.name}"
        for split, (img_dir, ann) in splits.items():
            datasets.append({"img_dir": img_dir, "ann_path": ann, "prefix": prefix, "ds_key": ds_key, "split": split})

    before, after, ann_all, imgs_all, final_all = merge_datasets(
        datasets, tmp_all, class_map, dedup, task, use_short_names, dbg, verbose
    )

    if len(final_all["annotations"]) == 0:
        dbg_file = dbg.flush()
        raise RuntimeError(
            f"[FATAL] 0 annotations after merge in unified pool. "
            f"Смотри детали в {dbg_file}."
        )

    # новый сплит
    cj = load_json(ann_all)
    img_ids = [im["id"] for im in cj.get("images", [])]
    random.shuffle(img_ids)
    n = len(img_ids)
    n_tr = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    set_tr  = set(img_ids[:n_tr])
    set_val = set(img_ids[n_tr:n_tr+n_val])
    set_te  = set(img_ids[n_tr+n_val:])

    def subset(name: str, ids_set: set):
        sub = out_dir / name
        sub_imgs = sub / "images"
        sub_imgs.mkdir(parents=True, exist_ok=True)

        # переносим изображения жёсткими ссылками/копиями
        for im in cj["images"]:
            if im["id"] in ids_set:
                src = imgs_all / im["file_name"]
                dst = sub_imgs / im["file_name"]
                if not dst.exists():
                    hardlink_or_copy(src, dst)

        # фильтруем аннотации/категории, пересобираем id
        anns = [a for a in cj["annotations"] if a["image_id"] in ids_set]
        cat_ids = sorted(set(a["category_id"] for a in anns))
        idmap_img = {old: i+1 for i, old in enumerate(sorted(ids_set))}
        idmap_cat = {old: i+1 for i, old in enumerate(cat_ids)}

        images_new, anns_new, cats_new = [], [], []
        for old, new in idmap_img.items():
            im = next(x for x in cj["images"] if x["id"] == old)
            im2 = dict(im); im2["id"] = new; images_new.append(im2)
        for i, a in enumerate(anns, 1):
            a2 = dict(a); a2["id"] = i
            a2["image_id"] = idmap_img[a["image_id"]]
            a2["category_id"] = idmap_cat[a["category_id"]]
            anns_new.append(a2)
        for c in cj["categories"]:
            if c["id"] in idmap_cat:
                cc = dict(c); cc["id"] = idmap_cat[c["id"]]; cats_new.append(cc)

        save_json(sub / "annotations.coco.json", {
            "images": images_new,
            "annotations": anns_new,
            "categories": cats_new,
            "info": {"description": f"subset {name}"},
            "licenses": []
        })

    subset("train", set_tr)
    subset("val", set_val)
    subset("test", set_te)

    return {"all": (before, after, ann_all, imgs_all, final_all)}

# ----------------------------------- CLI ---------------------------------- #
def write_reports(reports: dict, out_dir: Path):
    rep = out_dir / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    def dump_counts(p: Path, counts: Counter):
        with p.open("w", encoding="utf-8", newline="") as f:
            f.write("class,count\n")
            for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                f.write(f"{k},{v}\n")
    for split, tup in reports.items():
        before, after = tup[0], tup[1]
        dump_counts(rep / f"{split}_classes_before.csv", before)
        dump_counts(rep / f"{split}_classes_after.csv",  after)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", required=True, help="Путь к распакованным архивам (work/_extracted).")
    ap.add_argument("--out", required=True, help="Куда писать итоговый датасет.")
    ap.add_argument("--mode", choices=["by_split","unified"], default="by_split")
    ap.add_argument("--split", nargs=3, type=float, default=[0.8,0.1,0.1])
    ap.add_argument("--map", default=None, help="class_map.json")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--task", choices=["detection","seg"], default="detection")
    ap.add_argument("--short-names", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # .absolute() — чтобы короткие симлинки не разворачивать в длинные реальные пути
    extracted_root = Path(args.extracted).expanduser().absolute()
    out_dir = Path(args.out).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    # class map
    class_map_raw = {}
    if args.map:
        try:
            class_map_raw = json.loads(Path(args.map).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] cannot read class_map.json: {e}")
    class_map = normalize_class_map(class_map_raw)

    dbg = DebugLog(out_dir, enable=True)

    if args.mode == "by_split":
        reports = build_by_split(extracted_root, out_dir, class_map, args.dedup, args.task, args.short_names, dbg, args.verbose)
    else:
        reports = build_unified(extracted_root, out_dir, class_map, args.split, args.dedup, args.task, args.short_names, dbg, args.verbose)

    dbg_file = dbg.flush()
    if args.verbose and dbg_file:
        print(f"[DEBUG] detailed log: {dbg_file}")

    write_reports(reports, out_dir)

    print("\nDone:", out_dir)
    if args.mode == "unified":
        print("  - _all/images + annotations.coco.json (temp pool)")
    print("  - train/val/test/images + annotations.coco.json")
    print("  - reports/*classes_*.csv")
    if dbg_file:
        print("  - debug/merge_debug.tsv")

if __name__ == "__main__":
    main()

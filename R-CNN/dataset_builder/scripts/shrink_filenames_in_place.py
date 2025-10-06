#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shrink_filenames_in_place.py — укоротить длинные ИМЕНА файлов в _extracted
и обновить COCO-аннотации. Работает через файловые дескрипторы каталогов,
без длинных абсолютных путей.

Пример:
  python shrink_filenames_in_place.py --root /home/user/_extracted --max-len 128
"""
import argparse, hashlib, json, os, re
from pathlib import Path

IMG_EXT = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff",".gif"}
ANN_CAND = [
    "annotations.coco.json","_annotations.coco.json",
    "annotation.coco.json","anotation.coco.json","coco.json"
]
SPLITS = {"train","val","validation","valid","test"}

# ---------------------------- FD-based helpers ---------------------------- #

def fchdir_listdir(dirfd: int):
    """Безопасно получить список имён в каталоге по его FD (не используя длинный путь)."""
    oldfd = os.open(".", os.O_RDONLY)   # сохранить текущий CWD как FD
    try:
        os.fchdir(dirfd)
        names = os.listdir(".")
    finally:
        os.fchdir(oldfd)
        os.close(oldfd)
    return names

def open_at(dirfd: int, name: str, flags: int, mode: int = 0o644):
    return os.open(name, flags, mode, dir_fd=dirfd)

def read_json_fd(fd: int):
    f = os.fdopen(os.dup(fd), "r", encoding="utf-8")
    try:
        return json.load(f)
    finally:
        f.close()

def write_json_fd(fd: int, data):
    txt = json.dumps(data, ensure_ascii=False)
    f = os.fdopen(os.dup(fd), "w", encoding="utf-8")
    try:
        f.write(txt)
    finally:
        f.close()

def rename_in_dir(dirfd: int, old: str, new: str):
    os.rename(old, new, src_dir_fd=dirfd, dst_dir_fd=dirfd)

# ------------------------------- utilities -------------------------------- #

def sha10(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8","ignore")).hexdigest()[:10]

def short_name(orig: str, max_len: int) -> str:
    stem, suf = os.path.splitext(orig)
    h = sha10(orig)
    keep = max(1, max_len - len(h) - len(suf) - 2)  # "_" + hash + suffix
    return f"{stem[:keep]}_{h}{suf}"

def is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXT

def index_images_by_hash(filenames):
    """
    Вернёт:
      - by_name: set существующих имён
      - by_hash: (sha, ext_lower) -> [names...], где имя заканчивается на _<sha>(_<n>)?.ext
    """
    by_name = set()
    by_hash = {}
    for nm in filenames:
        if not is_image(nm):
            continue
        by_name.add(nm)
        stem, ext = os.path.splitext(nm)
        m = re.search(r"_([0-9a-f]{10})(?:_\d+)?$", stem)
        if m:
            h = m.group(1)
            by_hash.setdefault((h, ext.lower()), []).append(nm)
    return by_name, by_hash

def pick_best_candidate(cands, old_stem_prefix):
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    def score(nm):
        stem = os.path.splitext(nm)[0]
        pref = 0
        for a, b in zip(stem, old_stem_prefix):
            if a == b: pref += 1
            else: break
        return (pref, -len(nm))
    return sorted(cands, key=score, reverse=True)[0]

# --------------------------------- core ----------------------------------- #

def process_split(dirfd_split: int, split_path: str, max_len: int) -> (int, int):
    """
    Переименовывает длинные BASENAME в каталоге с изображениями этого сплита
    и обновляет COCO-аннотации. Возвращает (renamed_count, fixed_count).
    """
    # По умолчанию картинки лежат в самом split/
    img_dirfd = dirfd_split
    has_images_dir = False

    # Если есть подкаталог images — работаем внутри него
    try:
        imgs_fd = open_at(dirfd_split, "images", os.O_RDONLY)
        # получилось открыть — значит есть images/
        img_dirfd = imgs_fd
        has_images_dir = True
    except FileNotFoundError:
        pass

    # 1) Переименуем слишком длинные BASENAME в каталоге с картинками
    filenames = fchdir_listdir(img_dirfd)
    occupied = set(filenames)
    renamed = {}
    renamed_count = 0

    for nm in list(filenames):
        if not is_image(nm):
            continue
        if len(nm) <= max_len:
            continue

        new_nm = short_name(nm, max_len)
        base, suf = os.path.splitext(new_nm)
        i = 1
        while new_nm in occupied:
            new_nm = f"{base}_{i}{suf}"
            i += 1
        try:
            rename_in_dir(img_dirfd, nm, new_nm)
        except Exception as e:
            print(f"[WARN] rename failed: {split_path}: {nm} -> {new_nm}: {e}")
            continue

        occupied.discard(nm)
        occupied.add(new_nm)
        renamed[nm] = new_nm
        renamed_count += 1

    # 2) Обновим COCO-аннотации (split/ + split/annotations/)
    fixed_total = 0

    def fix_ann(dirfd: int, ann_name: str):
        nonlocal fixed_total
        try:
            afd = open_at(dirfd, ann_name, os.O_RDONLY)
            data = read_json_fd(afd); os.close(afd)
        except Exception as e:
            print(f"[WARN] read ann failed {split_path}/{ann_name}: {e}")
            return

        # список файлов после возможных переименований
        fnames_now = fchdir_listdir(img_dirfd)
        by_name, by_hash = index_images_by_hash(fnames_now)

        changed = 0
        for im in data.get("images", []):
            old = im.get("file_name","")
            old_bn = os.path.basename(old)
            if old_bn in by_name:
                continue
            # прямое соответствие «мы только что переименовали»
            if old_bn in renamed:
                im["file_name"] = renamed[old_bn]
                changed += 1
                continue
            # поиск по хэшу (как делал укоротитель)
            stem, ext = os.path.splitext(old_bn)
            h = sha10(old_bn)
            cands = list(by_hash.get((h, ext.lower()), []))
            if not cands:
                # попробуем любые расширения с тем же хэшем
                for (hh, eext), arr in by_hash.items():
                    if hh == h:
                        cands.extend(arr)
            if cands:
                best = pick_best_candidate(cands, stem[:50])
                if best:
                    im["file_name"] = best
                    changed += 1
            else:
                # не нашли — оставим как есть (сообщать не будем, чтобы не засорять лог)
                pass

        if changed:
            try:
                afd = open_at(dirfd, ann_name, os.O_WRONLY | os.O_TRUNC)
            except FileNotFoundError:
                afd = open_at(dirfd, ann_name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            write_json_fd(afd, data); os.close(afd)
            fixed_total += changed

    # основной json в split/
    split_files = fchdir_listdir(dirfd_split)
    ann_in_split = next((n for n in ANN_CAND if n in split_files), None)
    if ann_in_split:
        fix_ann(dirfd_split, ann_in_split)

    # json в split/annotations/
    if "annotations" in split_files:
        try:
            ann_dirfd = open_at(dirfd_split, "annotations", os.O_RDONLY)
            for nm in fchdir_listdir(ann_dirfd):
                if nm.lower().endswith(".json"):
                    fix_ann(ann_dirfd, nm)
            os.close(ann_dirfd)
        except Exception as e:
            print(f"[WARN] cannot open annotations dir in {split_path}: {e}")

    # закрыть img_dirfd, если открывали отдельный
    if has_images_dir:
        os.close(img_dirfd)

    return renamed_count, fixed_total

# --------------------------------- main ----------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Путь к work/_extracted")
    ap.add_argument("--max-len", type=int, default=128, help="Макс. длина ИМЕНИ файла (без пути)")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    total_renamed = 0
    total_fixed = 0
    any_split = False

    # Обходим только директории-сплиты через os.fwalk
    for dirpath, dirnames, filenames, dirfd in os.fwalk(root, topdown=True, follow_symlinks=False):
        base = os.path.basename(dirpath.rstrip(os.sep))
        if base not in SPLITS:
            continue
        any_split = True
        r, f = process_split(dirfd, dirpath, args.max_len)
        total_renamed += r
        total_fixed += f

    if not any_split:
        print("[!] Не найдено ни одного сплита (train/val/validation/valid/test). Проверь --root.")
    print(f"Renamed basenames: {total_renamed}")
    print(f"Fixed filenames in annotations: {total_fixed}")

if __name__ == "__main__":
    main()

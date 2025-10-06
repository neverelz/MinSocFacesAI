#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_map_vs_archive.py (robust)

Сверяет class_map_old.json с classes_by_archive.csv:
- печатает по каждому архиву классы, которых НЕТ в мапе (NO_RULE)
- отдельно сохраняет unmapped/ignored в CSV

Поддержка:
  --col-archive/--col-class/--col-count  (явное указание имён столбцов)
  авто-детект колонок (расширенный список синонимов)
  --ds-key-mode as_is|tail2|tail3        (нормализация ключа архива)
  --show-cols                             (показать заголовки CSV и выйти)

Пример:
  python compare_map_vs_archive.py \
    --map R-CNN/class_map_old.json \
    --csv R-CNN/dataset_builder/work/reports/classes_by_archive.csv \
    --out R-CNN/dataset_builder/work/reports/csv \
    --ds-key-mode tail2
"""

import argparse, csv, json, re, sys
from collections import defaultdict
from pathlib import Path

# ---------------- utils ----------------
def canon(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\\s\\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s

def load_class_map(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for sec, rules in (raw or {}).items():
        sec_key = "__global__" if sec == "__global__" else sec.strip().strip("/").replace("\\","/")
        out.setdefault(sec_key, {})
        for k, v in (rules or {}).items():
            out[sec_key][canon(str(k))] = v
    return out

def normalize_ds_key(ds_key: str, mode: str):
    ds_key = (ds_key or "").strip().replace("\\","/").strip("/")
    if not ds_key:
        return ds_key
    if mode == "as_is":
        return ds_key
    parts = ds_key.split("/")
    if mode == "tail2" and len(parts) >= 2:
        return "/".join(parts[-2:])
    if mode == "tail3" and len(parts) >= 3:
        return "/".join(parts[-3:])
    return ds_key

def lookup_scoped(cmap: dict, ds_key: str, key: str):
    c = canon(key)
    # точное совпадение секции
    if ds_key in cmap and c in cmap[ds_key]:
        return cmap[ds_key][c]
    # секция-хвост (ds_key заканчивается на секцию)
    for sec in cmap:
        if sec == "__global__":
            continue
        if ds_key.endswith(sec) and c in cmap[sec]:
            return cmap[sec][c]
    # глобальные правила
    if "__global__" in cmap and c in cmap["__global__"]:
        return cmap["__global__"][c]
    return None

# ------------- column detection -------------
ARCHIVE_SYNONYMS = [
    "archive","dataset","ds_key","folder","folder_name","subfolder","subdir",
    "path","folder_path","dataset_path","archive_path","package","source","repo",
]
CLASS_SYNONYMS = [
    "class","category","name","label","class_name","category_name","cls",
]
COUNT_SYNONYMS = ["count","qty","n","freq","frequency","total"]

def detect_col(header, want, synonyms):
    hc = {canon(h): h for h in header}
    for k in want:      # строгое имя сначала
        if k in hc: return hc[k]
    for k in synonyms:  # потом синонимы
        if k in hc: return hc[k]
    return None

def detect_columns(header, col_archive=None, col_class=None, col_count=None):
    if col_archive and col_archive not in header:
        # попробуем без регистра/пробелов
        hc = {canon(h): h for h in header}
        key = canon(col_archive)
        col_archive = hc.get(key, None)
    if col_class and col_class not in header:
        hc = {canon(h): h for h in header}
        col_class = hc.get(canon(col_class), None)
    if col_count and col_count not in header:
        hc = {canon(h): h for h in header}
        col_count = hc.get(canon(col_count), None)

    if not col_archive:
        col_archive = detect_col(header, ["archive","dataset","ds_key"], ARCHIVE_SYNONYMS)
    if not col_class:
        col_class = detect_col(header, ["class","category","name"], CLASS_SYNONYMS)
    if not col_count:
        col_count = detect_col(header, [], COUNT_SYNONYMS)

    missing = []
    if not col_archive: missing.append("archive/dataset/ds_key (или синонимы)")
    if not col_class:   missing.append("class/category/name (или синонимы)")
    if missing:
        raise ValueError("Не найден(ы) столбец(ы): " + ", ".join(missing) +
                         f". Заголовки CSV: {header}")
    return col_archive, col_class, col_count

# ------------------- main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="Путь к class_map_old.json")
    ap.add_argument("--csv", required=True, help="Путь к classes_by_archive.csv")
    ap.add_argument("--out", default="./report_compare", help="Папка для отчётов CSV")
    ap.add_argument("--ds-key-mode", choices=["as_is","tail2","tail3"], default="as_is",
                    help="Нормализация ключа архива: as_is (по CSV), tail2 (последние 2 папки), tail3 (последние 3)")
    ap.add_argument("--col-archive", help="Явное имя колонки архива в CSV")
    ap.add_argument("--col-class", help="Явное имя колонки класса в CSV")
    ap.add_argument("--col-count", help="Явное имя колонки количества в CSV (опционально)")
    ap.add_argument("--show-cols", action="store_true", help="Показать заголовки CSV и выйти")
    args = ap.parse_args()

    map_path = Path(args.map)
    csv_path = Path(args.csv)
    out_dir  = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # class map
    cmap = load_class_map(map_path)

    # CSV
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        header = r.fieldnames or []
        if args.show_cols:
            print("CSV columns:", header)
            sys.exit(0)
        try:
            col_ds, col_cls, col_cnt = detect_columns(
                header,
                col_archive=args.col_archive,
                col_class=args.col_class,
                col_count=args.col_count
            )
        except ValueError as e:
            print(str(e))
            print("Подсказка: укажи имена колонок явно через --col-archive/--col-class/--col-count "
                  "или запусти с --show-cols, чтобы увидеть доступные заголовки.")
            sys.exit(2)

        classes_by_ds = defaultdict(lambda: defaultdict(int))
        for row in r:
            ds_raw = (row.get(col_ds) or "").strip()
            cls    = (row.get(col_cls) or "").strip()
            if not ds_raw or not cls:
                continue
            ds_key = normalize_ds_key(ds_raw, args.ds_key_mode)
            cnt = 0
            if col_cnt:
                txt = (row.get(col_cnt) or "").strip()
                if txt != "":
                    try:
                        cnt = int(float(txt))
                    except Exception:
                        cnt = 0
            classes_by_ds[ds_key][cls] += cnt if cnt else 1

    unmapped = []  # NO_RULE
    ignored  = []  # mapped to __ignore__
    printed  = False

    for ds_key in sorted(classes_by_ds.keys()):
        missing_here = []
        for cls_name, cnt in sorted(classes_by_ds[ds_key].items(), key=lambda kv:(-kv[1], kv[0])):
            m = lookup_scoped(cmap, ds_key, cls_name)
            if m is None:
                missing_here.append((cls_name, cnt))
                unmapped.append([ds_key, cls_name, cnt])
            elif m == "__ignore__":
                ignored.append([ds_key, cls_name, cnt])
        if missing_here:
            printed = True
            print(f"\n[{ds_key}] нет правил для:")
            for name, cnt in missing_here:
                print(f"  - {name}  (count={cnt})")

    if not printed:
        print("Все классы из CSV покрыты правилами (с учётом __global__).")

    # save CSV
    def write_csv(path: Path, rows, header):
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)

    if unmapped:
        write_csv(out_dir / "unmapped_classes.csv", unmapped, ["archive(ds_key)","class_name","count"])
        print(f"\nСохранено: {out_dir / 'unmapped_classes.csv'}")
    if ignored:
        write_csv(out_dir / "ignored_classes.csv", ignored, ["archive(ds_key)","class_name","count"])
        print(f"Сохранено: {out_dir / 'ignored_classes.csv'}")

if __name__ == "__main__":
    main()

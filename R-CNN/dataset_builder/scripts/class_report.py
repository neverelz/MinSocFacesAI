# -*- coding: utf-8 -*-
import argparse, json, re
from pathlib import Path
from collections import Counter, defaultdict
from utils_common import load_json, find_annotation_json, detect_split_dirs, sanitize

# Опционально используем PyYAML и парсинг простых JSON/TXT для карт классов
try:
    import yaml
    YAML_OK = True
except Exception:
    YAML_OK = False

DOC_CANDIDATES = [
    "README", "readme", "README.md", "readme.md",
    "data.yaml", "dataset.yaml", "roboflow.yaml",
    "classes.txt", "classes.names", "labels.txt", "names.txt",
    "labelmap.json", "labels.json", "meta.json"
]

def try_parse_doc_labelmap(doc_path: Path):
    """
    Пытаемся извлечь список/словарь классов из документации.
    Возвращаем dict[int->str] или list[str] или {}.
    """
    name = doc_path.name.lower()
    try:
        txt = doc_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}

    # JSON с labelmap
    if name.endswith(".json"):
        try:
            obj = json.loads(txt)
            # варианты: {"0":"knife","1":"scissors"} или {"names":[...]}
            if isinstance(obj, dict) and all(k.isdigit() for k in obj.keys()):
                return {int(k): str(v) for k, v in obj.items()}
            if isinstance(obj, dict) and "names" in obj and isinstance(obj["names"], list):
                return {i: str(n) for i, n in enumerate(obj["names"])}
        except Exception:
            pass

    # YAML
    if name.endswith(".yaml") and YAML_OK:
        try:
            y = yaml.safe_load(txt)
            # варианты: names: [..]  или names: {0: "...", 1: "..."}
            if isinstance(y, dict) and "names" in y:
                names = y["names"]
                if isinstance(names, list):
                    return {i: str(n) for i, n in enumerate(names)}
                if isinstance(names, dict):
                    return {int(k): str(v) for k, v in names.items()}
        except Exception:
            pass

    # TXT/NAMES — по строке на класс
    if name.endswith(".txt") or name.endswith(".names") or name.endswith(".md") or "readme" in name:
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        # очень грубо: если строк < 200 и они без markdown-разметки — считаем перечислением классов
        pure = [ln for ln in lines if len(ln) < 64]
        if 1 <= len(pure) <= 200:
            return {i: pure[i] for i in range(len(pure))}

    return {}

def find_docs(root: Path):
    docs = []
    for p in root.rglob("*"):
        if p.is_file() and p.name in DOC_CANDIDATES:
            docs.append(p)
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", required=True, help="Путь к work/_extracted (после распаковки).")
    ap.add_argument("--out", required=True, help="Папка для отчётов.")
    args = ap.parse_args()
    extracted = Path(args.extracted).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    meta_rows = []

    # перебираем распакованные наборы
    for ds_root in sorted(extracted.rglob("*")):
        # считаем датасетом только листовые папки, где есть какой-то из сплитов
        if not ds_root.is_dir():
            continue
        splits = detect_split_dirs(ds_root)
        if not any(splits.values()):
            continue

        # попробуем достать документацию
        doc_map = {}
        docs = find_docs(ds_root)
        doc_src = None
        for d in docs:
            dm = try_parse_doc_labelmap(d)
            if dm:
                doc_map = dm
                doc_src = d
                break

        # для каждого сплита — считаем по классам
        for split_name, split_dir in splits.items():
            if not split_dir:
                continue
            ann = find_annotation_json(split_dir)
            if not ann or not ann.exists():
                continue
            cj = load_json(ann)
            id2name = {c["id"]: c.get("name", str(c["id"])) for c in cj.get("categories", [])}
            counts = Counter()
            for a in cj.get("annotations", []):
                counts[a["category_id"]] += 1

            for cat_id, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                cat_name = id2name.get(cat_id, str(cat_id))
                doc_name = None
                # если в документации имена по индексу
                if isinstance(doc_map, dict):
                    if cat_name.isdigit() and int(cat_name) in doc_map:
                        doc_name = doc_map[int(cat_name)]
                    elif cat_id in doc_map:
                        doc_name = doc_map[cat_id]
                rows.append({
                    "archive_dir": f"{ds_root.parent.name}/{ds_root.name}",
                    "split": split_name,
                    "category_id": cat_id,
                    "category_name_from_coco": cat_name,
                    "doc_name_guess": doc_name or "",
                    "count": cnt,
                    "annotations_path": str(ann)
                })

        # мета-строка по документации
        meta_rows.append({
            "archive_dir": f"{ds_root.parent.name}/{ds_root.name}",
            "doc_found": "yes" if doc_map else "no",
            "doc_path": str(doc_src) if doc_map else "",
            "doc_preview": json.dumps(doc_map)[:400]
        })

    # сохраняем отчёты
    import csv
    classes_csv = out_dir / "classes_by_archive.csv"
    with classes_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "archive_dir","split","category_id","category_name_from_coco","doc_name_guess","count","annotations_path"
        ])
        w.writeheader()
        for r in rows: w.writerow(r)

    meta_csv = out_dir / "docs_detected.csv"
    with meta_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["archive_dir","doc_found","doc_path","doc_preview"])
        w.writeheader()
        for r in meta_rows: w.writerow(r)

    print("Создан отчёт:")
    print(" -", classes_csv)
    print(" -", meta_csv)
    print("Откройте classes_by_archive.csv, чтобы спроектировать class_map.json.")

if __name__ == "__main__":
    main()

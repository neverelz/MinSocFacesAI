# -*- coding: utf-8 -*-
import zipfile, argparse
from pathlib import Path
from utils_common import sanitize

def extract_all_zips(root: Path, work_dir: Path):
    zips = list(root.rglob("*.zip"))
    out_dirs = []
    for z in zips:
        cat = sanitize(z.parent.name)
        base = sanitize(z.stem)
        dst = work_dir / "_extracted" / cat / base
        if dst.exists():
            out_dirs.append(dst); continue
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(dst)
        out_dirs.append(dst)
    return out_dirs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Корень с папками (канцелярия/мед/удушающие) и ZIP-архивами.")
    ap.add_argument("--work", required=True, help="Рабочая папка (куда распаковывать).")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    work = Path(args.work).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    out = extract_all_zips(root, work)
    print(f"Готово. Распаковано наборов: {len(out)}")
    print(f"Путь: {work/'_extracted'}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_subset_coco.py — формирует подвыборку COCO:
  - Размер фиксированный (--n) ИЛИ цель по кадрам на класс (--per-class).
  - Баланс по классам (жадный set-cover).
  - (Опц.) oversample редких классов: дублирует изображения с новыми ID/файлами.
  - (Опц.) доля пустых кадров (--neg-frac).
  - Стратифицированный сплит train/val.

Примеры:
  # 50 кадров для sanity (как раньше), без баланса/оверсэмпла
  python R-CNN/FCOS/tool/smake_subset_coco.py  \
    --ann /home/user/mrgv2/train/annotations.coco.json \
    --img-root /home/user/mrgv2/train/images \
    --out /home/user/mrgv2/tiny50 \
    --n 50 --seed 42

  # Пилот ≈1000 кадров, максимально ровно по классам (без дублей)
  python R-CNN/FCOS/tools/make_subset_coco.py \
    --ann /home/user/mrgv2/train/annotations.coco.json \
    --img-root /home/user/mrgv2/train/images \
    --out /home/user/mrgv2/pilot_balanced_1000 \
    --n 1000 --balance --seed 42

  # Пилот «по 30 кадров на класс», с oversample до цели (разрешаем дубли)
  python R-CNN/FCOS/tools/make_subset_coco.py \
    --ann /home/user/mrgv2/train/annotations.coco.json \
    --img-root /home/user/mrgv2/train/images \
    --out /home/user/mrgv2/pilot_per_class30_os \
    --per-class 150 --balance --oversample --seed 42
"""
import argparse, json, os, random, shutil, hashlib
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

def _read_coco(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    imgs = {im["id"]: im for im in data["images"]}
    cats = {c["id"]: c for c in data.get("categories", [])}
    anns_by_img: Dict[int, List[dict]] = defaultdict(list)
    cats_by_img: Dict[int, Set[int]] = defaultdict(set)
    for a in data.get("annotations", []):
        anns_by_img[a["image_id"]].append(a)
        if "category_id" in a:
            cats_by_img[a["image_id"]].add(a["category_id"])
    return data, imgs, cats, anns_by_img, cats_by_img

def _image_src(img_root: Path, file_name: str) -> Path:
    p = Path(file_name)
    if not p.is_absolute():
        p = img_root / file_name
    if not p.exists():
        alt = img_root / os.path.basename(file_name)
        if alt.exists():
            p = alt
    return p

def _copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)

def _greedy_cover_balance(
    imgs: Dict[int, dict],
    cats_by_img: Dict[int, Set[int]],
    target_per_class: Dict[int, int],
) -> List[int]:
    """
    Жадная подборка изображений, чтобы приблизиться к target_per_class.
    На каждом шаге берём кадр, который лучше всего уменьшает недобор.
    """
    remaining = target_per_class.copy()
    # Предотвратить зацикливание: игнорируем классы с нулевой целью
    for k, v in list(remaining.items()):
        if v <= 0:
            remaining[k] = 0
    chosen: List[int] = []
    pool: Set[int] = set(imgs.keys())

    def gain(img_id: int) -> int:
        # Сколько «очков» недобора закроет этот кадр
        g = 0
        for c in cats_by_img.get(img_id, set()):
            if remaining.get(c, 0) > 0:
                g += 1
        return g

    # Пока есть классы с недобором и есть кадры
    while any(v > 0 for v in remaining.values()) and pool:
        # Выбираем кадр с макс. gain; при равенстве — тот, где больше редких классов
        best = None
        best_gain = -1
        for iid in list(pool):
            g = gain(iid)
            if g > best_gain:
                best, best_gain = iid, g
        if best is None or best_gain <= 0:
            # Больше нечем покрывать недобор (нет подходящих кадров)
            break
        chosen.append(best)
        pool.remove(best)
        for c in cats_by_img.get(best, set()):
            if remaining.get(c, 0) > 0:
                remaining[c] -= 1

    return chosen

def _stratified_split(ids: List[int], cats_by_img: Dict[int, Set[int]], val_frac=0.2, seed=42) -> Tuple[Set[int], Set[int]]:
    """
    Стратифицированный сплит по классам (жадный, стабильный).
    """
    rnd = random.Random(seed)
    ids = list(ids)
    rnd.shuffle(ids)

    # Считаем частоты классов в наборе
    cls_counts = Counter()
    for iid in ids:
        cls_counts.update(cats_by_img.get(iid, set()))
    val_target = {c: int(round(n * val_frac)) for c, n in cls_counts.items()}
    val_got = Counter()
    val_ids: Set[int] = set()
    train_ids: Set[int] = set()

    for iid in ids:
        # Если этот кадр помогает заполнить "дыру" в валидации — отдаём его в val
        helps = sum(1 for c in cats_by_img.get(iid, set()) if val_got[c] < val_target[c])
        if helps > 0 and rnd.random() < 0.7:  # мягкий приоритет в val
            val_ids.add(iid)
            for c in cats_by_img.get(iid, set()):
                if val_got[c] < val_target[c]:
                    val_got[c] += 1
        else:
            train_ids.add(iid)

    # если недобор в val — докинем случайные
    deficit = sum(max(0, val_target[c] - val_got[c]) for c in val_target)
    if deficit > 0:
        for iid in ids:
            if iid in val_ids:
                continue
            val_ids.add(iid)
            if len(val_ids) >= int(round(len(ids) * val_frac)):
                break
        train_ids = set(ids) - val_ids

    return train_ids, val_ids

def _new_id(seed_text: str, base: int) -> int:
    # стабильный «новый id» для дубликатов (чтобы избежать пересечений)
    h = hashlib.sha1(f"{seed_text}-{base}".encode()).hexdigest()[:12]
    return int(h, 16) % (2**31 - 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Путь к исходному COCO JSON (train)")
    ap.add_argument("--img-root", required=True, help="Корень изображений (train/images)")
    ap.add_argument("--out", required=True, help="Куда положить subset")
    # Режим выбора размера:
    ap.add_argument("--n", type=int, default=None, help="Желаемое число изображений всего")
    ap.add_argument("--per-class", type=int, default=None, help="Целевое число изображений на класс (до oversample)")
    # Баланс и oversample:
    ap.add_argument("--balance", action="store_true", help="Балансировать по классам (жадный подбор)")
    ap.add_argument("--oversample", action="store_true", help="Дозаполнить редкие классы дубликатами изображений")
    ap.add_argument("--neg-frac", type=float, default=0.0, help="Доля пустых изображений без аннотаций (0..1)")
    # Сплит:
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    img_out = out / "images"
    out.mkdir(parents=True, exist_ok=True)
    img_out.mkdir(parents=True, exist_ok=True)

    data, imgs, cats, anns_by_img, cats_by_img = _read_coco(Path(args.ann))
    all_img_ids = list(imgs.keys())
    pos_ids = [iid for iid in all_img_ids if len(anns_by_img.get(iid, [])) > 0]
    neg_ids = [iid for iid in all_img_ids if len(anns_by_img.get(iid, [])) == 0]

    if args.n is None and args.per_class is None and not args.balance:
        print("[INFO] Ни --n, ни --per-class, ни --balance не заданы — по умолчанию выберу N=50 (как sanity).")
        args.n = 50

    # Цели по классам
    cat_ids = sorted(cats.keys())
    target_per_class = {c: 0 for c in cat_ids}

    if args.per_class is not None:
        # Явная цель на класс
        for c in cat_ids:
            target_per_class[c] = max(0, args.per_class)
    elif args.n is not None and args.balance:
        # Размазать по классам
        per = max(1, int(round(args.n / max(1, len(cat_ids)))))
        for c in cat_ids:
            target_per_class[c] = per
    else:
        # Без балансировки: просто возьмём N случайных (с опцией пустых)
        n_total = args.n if args.n is not None else 50
        n_neg = int(n_total * max(0.0, min(1.0, args.neg_frac)))
        n_pos = max(1, n_total - n_neg)
        random.shuffle(pos_ids); random.shuffle(neg_ids)
        chosen = pos_ids[:n_pos] + neg_ids[:n_neg]
        random.shuffle(chosen)
        train_ids, val_ids = _stratified_split(chosen, cats_by_img, val_frac=args.val_frac, seed=args.seed)
        # Сформируем JSON, скопируем файлы и выходим
        _emit_subset(out, img_out, data, imgs, anns_by_img, train_ids, val_ids, Path(args.img_root))
        return

    # BALANCED режим (жадный подбор)
    # 1) Собираем кадры, максимально покрывая таргеты по классам
    chosen_balanced = _greedy_cover_balance(imgs, cats_by_img, target_per_class)

    # 2) Добавим пустые кадры по желанию
    n_total = len(chosen_balanced)
    add_neg = int(((args.n or n_total) * max(0.0, min(1.0, args.neg_frac))) if args.n else int(n_total * args.neg_frac))
    random.shuffle(neg_ids)
    chosen = list(chosen_balanced) + neg_ids[:add_neg]

    # 3) Если задан --n и не дотянули — докинем случайных позитивных (без разрыва баланса слишком сильно)
    if args.n is not None and len(chosen) < args.n:
        pool = [iid for iid in pos_ids if iid not in chosen]
        random.shuffle(pool)
        need = args.n - len(chosen)
        chosen += pool[:max(0, need)]

    # 4) Oversample (создание дублей изображений и аннотаций), если запрошено
    #    Дубли создаются с НОВЫМИ image_id/annotation_id и копированием файла с суффиксом.
    #    Цель: достичь target_per_class.
    oversampled_images = {}  # new_id -> (src_img_id, dst_path)
    if args.oversample:
        # Текущие покрытия по классам
        cover = Counter()
        for iid in chosen:
            cover.update(cats_by_img.get(iid, set()))

        # Список кандидатов (из выбранных позитивных)
        pos_chosen = [iid for iid in chosen if iid in pos_ids]
        new_img_id_base = max(imgs.keys()) + 1
        new_ann_id_base = max([a["id"] for a in data.get("annotations", [])] or [0]) + 1
        dup_idx = 0

        # Пока есть классы с недобором — клонируем подходящие кадры
        def lack_classes():
            return [c for c in cat_ids if cover[c] < target_per_class.get(c, 0)]

        while True:
            lacking = lack_classes()
            if not lacking:
                break
            # Найдём изображение, которое покрывает как можно больше недостающих классов
            best, best_gain = None, -1
            for iid in pos_chosen:
                g = sum(1 for c in cats_by_img.get(iid, set()) if c in lacking)
                if g > best_gain:
                    best, best_gain = iid, g
            if best is None or best_gain <= 0:
                break  # нечем докрывать

            # Сгенерируем новый image_id и имя файла с суффиксом
            src_im = imgs[best]
            src_name = Path(src_im["file_name"]).name
            stem, ext = os.path.splitext(src_name)
            new_file_name = f"{stem}_dup{dup_idx:04d}{ext}"
            dup_idx += 1

            new_image_id = _new_id(new_file_name, new_img_id_base + dup_idx)
            # Зарегистрируем новый image
            new_im = dict(src_im)
            new_im["id"] = new_image_id
            new_im["file_name"] = f"images/{new_file_name}"

            imgs[new_image_id] = new_im
            cats_by_img[new_image_id] = set(cats_by_img.get(best, set()))
            # Аннотации-клоны
            for a in list(anns_by_img.get(best, [])):
                new_ann = dict(a)
                new_ann["id"] = _new_id(f"{a['id']}_{new_image_id}", new_ann_id_base + new_ann["id"])
                new_ann["image_id"] = new_image_id
                anns_by_img[new_image_id].append(new_ann)

            chosen.append(new_image_id)
            cover.update(cats_by_img.get(new_image_id, set()))
            oversampled_images[new_image_id] = (best, new_file_name)

    # 5) Сплит
    train_ids, val_ids = _stratified_split(chosen, cats_by_img, val_frac=args.val_frac, seed=args.seed)

    # 6) Выхлоп
    _emit_subset(out, img_out, data, imgs, anns_by_img, train_ids, val_ids, Path(args.img_root),
                 oversampled_map=oversampled_images)

def _emit_subset(out_dir: Path, img_out: Path, data, imgs, anns_by_img, train_ids: Set[int], val_ids: Set[int],
                 img_root: Path, oversampled_map=None):
    oversampled_map = oversampled_map or {}

    def build_json(image_ids: Set[int]):
        images = [imgs[i] for i in image_ids if i in imgs]
        anns = []
        for iid in image_ids:
            anns.extend(anns_by_img.get(iid, []))
        return {
            "images": images,
            "annotations": anns,
            "categories": data.get("categories", []),
            "licenses": data.get("licenses", []),
            "info": data.get("info", {}),
        }

    tiny_train = build_json(train_ids)
    tiny_val   = build_json(val_ids)

    # Копируем файлы
    used_ids = train_ids | val_ids
    for iid in used_ids:
        im = imgs[iid]
        fn = im["file_name"]
        # Если это новосозданный дубликат — имя уже images/<dupname>
        if iid in oversampled_map:
            src_iid, dup_name = oversampled_map[iid]
            src = _image_src(img_root, imgs[src_iid]["file_name"])
            dst = img_out / dup_name
            _copy_image(src, dst)
            im["file_name"] = f"images/{dup_name}"
        else:
            src = _image_src(img_root, fn)
            dst = img_out / Path(fn).name  # уплощаем в одну папку
            _copy_image(src, dst)
            im["file_name"] = f"images/{dst.name}"

    (out_dir / "subset_train.json").write_text(json.dumps(tiny_train, ensure_ascii=False), encoding="utf-8")
    (out_dir / "subset_val.json").write_text(json.dumps(tiny_val, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] out: {out_dir}")
    print(f"  train: {len(tiny_train['images'])} imgs, {len(tiny_train['annotations'])} anns")
    print(f"  val:   {len(tiny_val['images'])} imgs, {len(tiny_val['annotations'])} anns")
    print(f"  images dir: {img_out}")
    print(f"  json: {out_dir/'subset_train.json'}, {out_dir/'subset_val.json'}")

if __name__ == "__main__":
    main()

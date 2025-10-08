#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_tiny_coco.py — создать мини-датасет COCO из 50 изображений (40 train / 10 val)

Пример:
  python R-CNN/FCOS/tools/make_tiny_coco.py \
    --ann /home/user/mrgv2/train/annotations.coco.json \
    --img-root /home/user/mrgv2/train/images \
    --out /home/user/mrgv2/tiny50 \
    --n 50 --seed 42
"""
import argparse, json, os, random, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Путь к исходному COCO JSON (train)")
    ap.add_argument("--img-root", required=True, help="Корень изображений (train/images)")
    ap.add_argument("--out", required=True, help="Куда положить tiny датасет")
    ap.add_argument("--n", type=int, default=50, help="Сколько изображений выбрать всего")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg-frac", type=float, default=0.0,
                    help="Доля пустых изображений без аннотаций (0..1), по умолчанию 0.0")
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    img_out = out / "images"
    out.mkdir(parents=True, exist_ok=True)
    img_out.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.ann).read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data["images"]}
    anns_by_img = {}
    for a in data.get("annotations", []):
        anns_by_img.setdefault(a["image_id"], []).append(a)

    pos_ids = [iid for iid, _ in images.items() if iid in anns_by_img and len(anns_by_img[iid]) > 0]
    neg_ids = [iid for iid, _ in images.items() if iid not in anns_by_img or len(anns_by_img[iid]) == 0]

    if len(pos_ids) == 0:
        raise SystemExit("В исходном train нет изображений с аннотациями.")

    n_total = min(args.n, len(pos_ids) + len(neg_ids))
    n_neg = int(n_total * args.neg_frac)
    n_pos = max(1, n_total - n_neg)

    random.shuffle(pos_ids)
    random.shuffle(neg_ids)
    chosen_pos = pos_ids[:n_pos]
    chosen_neg = neg_ids[:n_neg]
    chosen = chosen_pos + chosen_neg
    random.shuffle(chosen)

    # Сплит 80/20 ≈ 40/10 для n=50
    k_train = max(1, int(len(chosen) * 0.8))
    train_ids = set(chosen[:k_train])
    val_ids   = set(chosen[k_train:])

    def subset_json(image_ids):
        imgs = []
        anns = []
        for iid in image_ids:
            im = images[iid]
            imgs.append(im)
            for a in anns_by_img.get(iid, []):
                anns.append(a)
        tiny = {
            "images": imgs,
            "annotations": anns,
            "categories": data.get("categories", []),
            "licenses": data.get("licenses", []),
            "info": data.get("info", {}),
        }
        return tiny

    tiny_train = subset_json(train_ids)
    tiny_val   = subset_json(val_ids)

    # Копируем изображения, повторяя относительный путь file_name, если он относительный
    def copy_images(image_list):
        for im in image_list:
            fn = im["file_name"]
            src = Path(fn)
            if not src.is_absolute():
                src = Path(args.img_root) / fn
            if not src.exists():
                # иногда file_name без подпапок — пробуем basename
                alt = Path(args.img_root) / os.path.basename(fn)
                if alt.exists():
                    src = alt
                else:
                    print(f"[WARN] Не найден файл: {src}")
                    continue
            # кладём в ту же относительную структуру
            dst = img_out / Path(fn).name  # уплощаем (в одной папке), чтобы просто
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            # правим путь в json на новый
            im["file_name"] = str(dst.relative_to(out))  # 'images/<name>'

    copy_images(tiny_train["images"])
    copy_images(tiny_val["images"])

    (out / "tiny_train.json").write_text(json.dumps(tiny_train, ensure_ascii=False), encoding="utf-8")
    (out / "tiny_val.json").write_text(json.dumps(tiny_val, ensure_ascii=False), encoding="utf-8")

    print(f"Готово: {out}")
    print(f"  train: {len(tiny_train['images'])} изображений, {len(tiny_train['annotations'])} аннотаций")
    print(f"  val:   {len(tiny_val['images'])} изображений, {len(tiny_val['annotations'])} аннотаций")
    print(f"  images dir: {img_out}")
    print(f"  json: {out/'tiny_train.json'}, {out/'tiny_val.json'}")

if __name__ == "__main__":
    main()

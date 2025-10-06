#!/usr/bin/env python3
"""
Convert filtered bbox annotations + downloaded images to COCO format.
Пример вызова:
python convert_to_coco.py --filtered_csv filtered_train.csv --images_dir images_train \
    --class_csv class-descriptions-boxable.csv --out_json coco_train.json
"""
import argparse
import pandas as pd
import os
import json
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--filtered_csv", required=True)
parser.add_argument("--images_dir", required=True)
parser.add_argument("--class_csv", required=True)
parser.add_argument("--out_json", required=True)
parser.add_argument("--min_area_px", type=int, default=4, help="min bbox area in px to keep")
args = parser.parse_args()

df = pd.read_csv(args.filtered_csv)
df_classes = pd.read_csv(args.class_csv, header=None, names=["label_id","label_name"])
id_to_name = dict(zip(df_classes["label_id"], df_classes["label_name"]))

# build category mapping
unique_label_ids = df["LabelName"].unique().tolist()
categories = []
catid_map = {}
for i, lid in enumerate(unique_label_ids, start=1):
    name = id_to_name.get(lid, lid)
    categories.append({"id": i, "name": name, "supercategory": "none"})
    catid_map[lid] = i

# collect images metadata
images = []
annotations = []
imgid_to_cocoid = {}
coco_img_id = 1
anno_id = 1

# group by image
grouped = df.groupby("ImageID")

for imgid, g in tqdm(grouped, desc="Converting to COCO"):
    image_path = os.path.join(args.images_dir, f"{imgid}.jpg")
    if not os.path.exists(image_path):
        continue
    try:
        with Image.open(image_path) as im:
            w, h = im.size
    except Exception as e:
        continue

    img_dict = {"id": coco_img_id, "file_name": f"{imgid}.jpg", "width": w, "height": h}
    images.append(img_dict)
    imgid_to_cocoid[imgid] = coco_img_id

    for _, row in g.iterrows():
        xmin = float(row["XMin"]) * w
        xmax = float(row["XMax"]) * w
        ymin = float(row["YMin"]) * h
        ymax = float(row["YMax"]) * h
        box_w = max(0, xmax - xmin)
        box_h = max(0, ymax - ymin)
        area = box_w * box_h
        if area < args.min_area_px:
            continue
        bbox = [xmin, ymin, box_w, box_h]
        category_id = catid_map[row["LabelName"]]
        ann = {"id": anno_id, "image_id": coco_img_id, "category_id": category_id,
               "bbox": [round(x,2) for x in bbox], "area": round(area,2),
               "iscrowd": 0}
        # No segmentation provided here — if you have polygon column, add "segmentation": [...]
        annotations.append(ann)
        anno_id += 1

    coco_img_id += 1

coco = {"images": images, "annotations": annotations, "categories": categories}
with open(args.out_json, "w", encoding="utf-8") as f:
    json.dump(coco, f, ensure_ascii=False, indent=2)
print(f"Wrote COCO JSON: {args.out_json} with {len(images)} images and {len(annotations)} annotations.")

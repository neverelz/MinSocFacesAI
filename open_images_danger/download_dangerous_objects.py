import os
import json
import csv
import pandas as pd
import requests
import shutil
from tqdm import tqdm
import argparse

# -------------------------------
# 1. Целевые родительские классы
# -------------------------------
TARGET_PARENTS = {
    "Office supplies",
    "Medical equipment",
    "Weapon",
    "Telephone",
    "Kitchen utensil",
    "Bathroom accessory"
}

# -------------------------------
# 2. Скачивание вспомогательных файлов
# -------------------------------
def download_file(url, dest):
    if os.path.exists(dest):
        print(f"[INFO] Already exists: {dest}")
        return
    print(f"[INFO] Downloading {url} → {dest}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        raise

# -------------------------------
# 3. Извлечение целевых label_name
# -------------------------------
def extract_target_labels():
    # Скачиваем файлы иерархии и описаний
    base_meta_url = "https://storage.googleapis.com/openimages/2018_04/"
    class_desc_file = "class-descriptions-boxable.csv"
    hierarchy_file = "bbox_labels_600_hierarchy.json"

    download_file(base_meta_url + class_desc_file, class_desc_file)
    download_file(base_meta_url + hierarchy_file, hierarchy_file)

    # Маппинг label_name → display_name (файл использует ; как разделитель!)
    label_to_name = {}
    with open(class_desc_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(';', 1)
            if len(parts) == 2:
                label_name = parts[0]
                display_name = parts[1]
                label_to_name[label_name] = display_name
            else:
                print(f"[WARN] Skipping invalid line: {line}")

    name_to_label = {v: k for k, v in label_to_name.items()}

    # Находим label_name для родителей
    target_parent_labels = set()
    for parent in TARGET_PARENTS:
        if parent in name_to_label:
            target_parent_labels.add(name_to_label[parent])
        else:
            print(f"[WARN] Parent '{parent}' not found!")

    # Загружаем иерархию (это ОДИН корневой JSON-объект!)
    with open(hierarchy_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            raise ValueError("Hierarchy file is empty!")
        root_node = json.loads(content)

    # Рекурсивный сбор ВСЕХ узлов из иерархии
    all_nodes = []
    def collect_all_nodes(node):
        all_nodes.append(node)
        if "Subcategory" in node:
            for child in node["Subcategory"]:
                collect_all_nodes(child)

    collect_all_nodes(root_node)

    # Теперь ищем нужные родительские узлы среди всех
    target_labels = set()
    def get_descendants(node, all_labels):
        label = node["LabelName"]
        all_labels.add(label)
        if "Subcategory" in node:
            for child in node["Subcategory"]:
                get_descendants(child, all_labels)

    for node in all_nodes:
        if node["LabelName"] in target_parent_labels:
            get_descendants(node, target_labels)

    target_labels.update(target_parent_labels)

    # Сохраняем
    with open("target_labels.txt", "w", encoding="utf-8") as f:
        for label in sorted(target_labels):
            f.write(f"{label}\n")

    print(f"[INFO] Found {len(target_labels)} target classes")
    return target_labels, label_to_name
# -------------------------------
# 4. Скачивание изображений и аннотаций
# -------------------------------
def download_dataset(split, target_labels, output_dir):
    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/annotations", exist_ok=True)

    # ПРАВИЛЬНЫЕ URL для Open Images V7
    base_v7_url = "https://storage.googleapis.com/openimages/v7/"
    base_meta_url = "https://storage.googleapis.com/openimages/2018_04/"

    # Скачиваем аннотации bbox
    bbox_file = f"{split}-annotations-bbox.csv"
    bbox_path = f"{output_dir}/annotations/{bbox_file}"
    if not os.path.exists(bbox_path):
        download_file(base_v7_url + bbox_file, bbox_path)
    else:
        print(f"[INFO] BBox annotations already exist: {bbox_path}")

    # Скачиваем аннотации масок (только для train)
    mask_path = None
    if split == "train":
        mask_file = f"{split}-annotations-object-segmentation.csv"
        mask_path = f"{output_dir}/annotations/{mask_file}"
        if not os.path.exists(mask_path):
            download_file(base_v7_url + mask_file, mask_path)
        else:
            print(f"[INFO] Mask annotations already exist: {mask_path}")

    # Скачиваем метаданные изображений
    img_meta_file = f"{split}-images-boxable.csv"
    img_meta_path = f"{output_dir}/annotations/{img_meta_file}"
    if not os.path.exists(img_meta_path):
        download_file(base_meta_url + img_meta_file, img_meta_path)
    else:
        print(f"[INFO] Image metadata already exist: {img_meta_path}")

    # Загружаем и фильтруем аннотации
    print("[INFO] Filtering annotations...")
    bbox_df = pd.read_csv(bbox_path)
    filtered_bbox = bbox_df[bbox_df["LabelName"].isin(target_labels)]
    image_ids = set(filtered_bbox["ImageID"].unique())

    img_meta = pd.read_csv(img_meta_path)
    filtered_images = img_meta[img_meta["ImageID"].isin(image_ids)]

    # Сохраняем фильтрованные файлы
    filtered_bbox.to_csv(f"{output_dir}/annotations/{split}-annotations-bbox-filtered.csv", index=False)
    filtered_images.to_csv(f"{output_dir}/annotations/{split}-images-boxable-filtered.csv", index=False)

    print(f"[INFO] Found {len(image_ids)} images with target objects")

    # Скачиваем изображения
    print(f"[INFO] Downloading {len(filtered_images)} images...")
    for _, row in tqdm(filtered_images.iterrows(), total=len(filtered_images)):
        image_id = row["ImageID"]
        out_path = f"{output_dir}/images/{split}/{image_id}.jpg"
        if os.path.exists(out_path):
            continue
        try:
            url = row["OriginalURL"]
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(out_path, "wb") as f:
                    shutil.copyfileobj(response.raw, f)
        except Exception as e:
            print(f"\n[WARN] Failed to download {image_id}: {e}")

    # Фильтрация масок
    if mask_path and os.path.exists(mask_path):
        print("[INFO] Filtering mask annotations...")
        mask_df = pd.read_csv(mask_path)
        filtered_mask = mask_df[mask_df["LabelName"].isin(target_labels)]
        filtered_mask.to_csv(f"{output_dir}/annotations/{split}-annotations-mask-filtered.csv", index=False)
        print(f"[INFO] Mask annotations saved for {len(filtered_mask)} objects")

# -------------------------------
# 5. Основная функция
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--output", default="open_images_danger")
    args = parser.parse_args()

    print(f"[INFO] Starting download for split: {args.split}")
    target_labels, label_to_name = extract_target_labels()

    print("\n[INFO] Sample target classes:")
    for label in list(target_labels)[:10]:
        print(f"  {label} → {label_to_name.get(label, 'UNKNOWN')}")

    download_dataset(args.split, target_labels, args.output)
    print(f"\n✅ Done! Data saved to: {args.output}")

if __name__ == "__main__":
    main()
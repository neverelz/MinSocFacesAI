# check_classmap_coverage.py
import json, sys
from pathlib import Path

extracted = Path("R-CNN/dataset_builder/work/_extracted")
cmap = json.loads(Path("/home/user/Documents/PycharmProjects/MinSocFacesAI/R-CNN/dataset_builder/configs/class_map.json").read_text(encoding="utf-8"))

# собрать все имена категорий по архивам
cats_by_ds = {}
for ds in extracted.rglob("*"):
    if not ds.is_dir(): continue
    for split in ("train","val","validation","valid","test"):
        sd = ds / split
        if not sd.is_dir(): continue
        for ann_name in ["annotations.coco.json","_annotations.coco.json","annotation.coco.json","anotation.coco.json","coco.json"]:
            ann = sd / ann_name
            if ann.exists():
                data = json.loads(ann.read_text(encoding="utf-8"))
                names = {c.get("name", str(c["id"])) for c in data.get("categories", [])}
                key = f"{ds.parent.name}/{ds.name}"
                cats_by_ds.setdefault(key, set()).update(names)

# проверить покрытие по секциям class_map
unmatched = []
def sec_keys(sec): return set(cmap.get(sec, {}).keys())

for sec in cmap:
    if sec == "__global__":
        # глобальные ключи проверим против объединённого множества
        all_names = set().union(*cats_by_ds.values()) if cats_by_ds else set()
        for k in sec_keys("__global__"):
            if k not in all_names:
                unmatched.append(("__global__", k))
    else:
        # секция по архиву: ищем точное совпадение ключа
        targets = []
        for ds_key in cats_by_ds:
            if ds_key == sec or ds_key.endswith(sec):
                targets.append(ds_key)
        present = set().union(*(cats_by_ds[t] for t in targets)) if targets else set()
        for k in sec_keys(sec):
            if k not in present:
                unmatched.append((sec, k))

print("UNMATCHED KEYS (section, key):")
for sec, k in unmatched[:200]:
    print(f" - {sec}: {k}")
print(f"Total unmatched: {len(unmatched)}")

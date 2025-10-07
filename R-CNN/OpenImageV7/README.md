Запуск фильтрации изображений

### python R-CNN/datasets/filter_openimages.py   --class_list R-CNN/datasets/expanded_classes.txt   --class_csv R-CNN/datasets/oidv6-class-descriptions.csv   --bbox_csv R-CNN/datasets/train-annotations-bbox.csv   --out_csv R-CNN/datasets/debug.csv

Запуск сбора изображений
### python R-CNN/datasets/download_images.py --image_csv R-CNN/datasets/train-images-boxable-with-rotation.csv --filter_csv R-CNN/datasets/debug.csv --out_dir R-CNN/datasets/images --max_images_per_class 3000

Запуск конфертации в COCO
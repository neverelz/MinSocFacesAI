# config.py
# Конфигурация проекта — меняйте тут или через CLI аргументы

# Камера: 0 - системная фронтальная, 1 - внешняя USB (по желанию)
CAMERA_INDEX = 0

# Путь к ONNX-модели RTMDet / YOLOX-совместимого экспорта
ONNX_MODEL_PATH = "models/rtmdet_medium.onnx"

# Размер входа модели (обычно 640 для многих экспоротов). Если ваша модель 320/512/640 — установите соответствующее.
INPUT_SIZE = 640

# Confidence threshold для отображения детекций (0..1)
CONFIDENCE_THRESHOLD = 0.25

# NMS IoU threshold
NMS_IOU_THRESHOLD = 0.45

# Максимум боксов после NMS
MAX_DETECTIONS = 200

# Включить/выключить использование GPU провайдера (onnxruntime будет пробовать Tensorrt/CUDA)
USE_GPU = True

# Список классов, которые следует исключить (люди и животные), указываются в файле classes_coco.py
EXCLUDE_CLASSES = ["person", "dog", "cat", "horse", "sheep", "cow", "elephant",
                   "bear", "zebra", "giraffe", "toilet"]

import cv2
import numpy as np
from typing import List, Dict, Tuple

# Detectron2 imports
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog


COCO_EXCLUDE_CLASSES = {
    "person",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


def build_detectron2_predictor(
    config_name: str = "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    score_threshold: float = 0.5,
    device: str = "cpu",
    weights_path: str | None = None,
):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_name))
    if not weights_path:
        raise ValueError(
            "weights_path is required to ensure commercial-usable weights. Provide path to your trained .pth/.pkl."
        )
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = score_threshold
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) if len(cfg.DATASETS.TRAIN) else MetadataCatalog.get("coco_2017_train")
    class_names = list(metadata.get("thing_classes", []))
    return predictor, class_names


def run_detectron2_inference(
    predictor: DefaultPredictor,
    frame_bgr: np.ndarray,
    class_names: List[str],
    exclude_class_names: set = COCO_EXCLUDE_CLASSES,
) -> List[Tuple[np.ndarray, float, int]]:
    outputs = predictor(frame_bgr)
    instances = outputs.get("instances")
    if instances is None or len(instances) == 0:
        return []

    boxes_xyxy = instances.pred_boxes.tensor.cpu().numpy().astype(np.float32)
    scores = instances.scores.cpu().numpy().astype(float)
    class_ids = instances.pred_classes.cpu().numpy().astype(int)

    results: List[Tuple[np.ndarray, float, int]] = []
    for box, score, cid in zip(boxes_xyxy, scores, class_ids):
        name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        if name in exclude_class_names:
            continue
        x1, y1, x2, y2 = box
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = max(x1 + 1.0, x2)
        y2 = max(y1 + 1.0, y2)
        results.append((np.array([x1, y1, x2, y2], dtype=np.float32), float(score), int(cid)))
    return results


def draw_detections_bgr(image_bgr: np.ndarray, detections: List[Tuple[np.ndarray, float, int]], class_names: List[str]) -> np.ndarray:
    vis = image_bgr.copy()
    for (box, score, class_id) in detections:
        x1, y1, x2, y2 = map(int, box)
        label_name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        label = f"{label_name} {score:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(vis, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return vis



import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class SizeEstimatorConfig:
    pixels_per_meter: float = 0.0  # Глобальный PPM (пиксели/метр)
    per_camera_ppm: Optional[Dict[str, float]] = None  # Не используется в упрощённой версии


class SizeEstimator:
    def __init__(self, cfg: SizeEstimatorConfig) -> None:
        # Хранит конфигурацию для перевода пикселей в метры
        self.cfg = cfg

    def estimate(self, bbox: List[int], camera_key: Optional[str] = None) -> Dict[str, float]:
        # Оценивает ширину/высоту объекта в метрах по PPM
        x1, y1, x2, y2 = bbox
        width_px = max(1, x2 - x1)
        height_px = max(1, y2 - y1)
        ppm = float(self.cfg.pixels_per_meter)
        if ppm <= 0:
            return {"width_m": 0.0, "height_m": 0.0}
        return {"width_m": float(width_px / ppm), "height_m": float(height_px / ppm)}


def draw_detections(image: np.ndarray, detections: List[Dict], sizes: Optional[List[Dict]] = None) -> np.ndarray:
    # Рисует детекции с подписями (класс, вероятность, размеры)
    if sizes is None:
        sizes = [{} for _ in detections]
    for det, sz in zip(detections, sizes):
        x1, y1, x2, y2 = det["bbox"]
        cls = det.get("class_name", str(det.get("class_id", "?")))
        score = det.get("score", 0.0)
        label = f"{cls} {score:.2f}"
        w_m = sz.get("width_m", 0.0)
        h_m = sz.get("height_m", 0.0)
        if w_m > 0 or h_m > 0:
            label += f" [{w_m:.2f}m x {h_m:.2f}m]"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def append_json_lines(path: str, records: List[Dict]) -> None:
    # Добавляет записи в локальный JSONL-файл
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")




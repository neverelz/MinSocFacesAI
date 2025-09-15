import os
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import openvino as ov
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenVINO is required. Please install 'openvino'.") from exc


class OpenVINOObjectDetector:
    """Обёртка над моделью детекции объектов OpenVINO (лицензии OMZ — для коммерции)."""

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        labels: Optional[List[str]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        custom_postprocess=None,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.core = ov.Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, device)
        self.infer_request = self.compiled_model.create_infer_request()

        # Determine input name and expected layout
        inputs = self.compiled_model.inputs
        if not inputs:
            raise RuntimeError("Model has no inputs")
        self.input_port = inputs[0]
        input_shape = list(self.input_port.get_shape())
        if len(input_shape) != 4:
            raise RuntimeError(f"Unsupported input rank {len(input_shape)}; expected NCHW")
        self.input_height = int(input_shape[2])
        self.input_width = int(input_shape[3])

        # Output
        outputs = self.compiled_model.outputs
        if not outputs:
            raise RuntimeError("Model has no outputs")
        self.output_port = outputs[0]

        self.labels = labels
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.custom_postprocess = custom_postprocess

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """Подгоняет кадр под вход модели (letterbox, нормализация)."""
        h, w = self.input_height, self.input_width
        img = image_bgr
        # Letterbox to preserve aspect ratio
        scale = min(w / img.shape[1], h / img.shape[0])
        new_w, new_h = int(img.shape[1] * scale), int(img.shape[0] * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        canvas = np.full((h, w, 3), 114, dtype=np.uint8)
        canvas[top : top + new_h, left : left + new_w] = resized
        # BGR -> RGB and HWC -> NCHW, normalize to 0..1
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        meta = {
            "scale": scale,
            "pad_top": top,
            "pad_left": left,
            "orig_w": img.shape[1],
            "orig_h": img.shape[0],
        }
        return blob, meta

    def postprocess_default(self, raw: np.ndarray, meta: Dict) -> List[Dict]:
        """Декодирует стандартный вывод детектора в список bbox/score/class."""
        # Handle shapes [N, 1, 200, 7] or [N, 200, 7] or [200, 7]
        detections = raw
        detections = np.squeeze(detections)
        if detections.ndim == 3:
            detections = detections[0]

        results: List[Dict] = []
        for det in detections:
            if det.shape[-1] != 7:
                # Unexpected format; skip
                continue
            _, label, conf, x_min, y_min, x_max, y_max = det.tolist()
            if conf < self.confidence_threshold:
                continue
            # Convert from letterboxed coordinates back to original image size
            pad_left = meta["pad_left"]
            pad_top = meta["pad_top"]
            scale = meta["scale"]

            x1 = max(0, int((x_min * self.input_width - pad_left) / scale))
            y1 = max(0, int((y_min * self.input_height - pad_top) / scale))
            x2 = min(meta["orig_w"], int((x_max * self.input_width - pad_left) / scale))
            y2 = min(meta["orig_h"], int((y_max * self.input_height - pad_top) / scale))

            class_id = int(label)
            class_name = self.labels[class_id] if self.labels and 0 <= class_id < len(self.labels) else str(class_id)

            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "score": float(conf),
                    "class_id": class_id,
                    "class_name": class_name,
                }
            )
        return results

    def infer(self, image_bgr: np.ndarray) -> List[Dict]:
        """Полный цикл инференса: препроцесс → модель → постпроцесс."""
        blob, meta = self.preprocess(image_bgr)
        outputs = self.infer_request.infer({self.input_port.any_name: blob})
        # take first (and usually only) output
        raw = next(iter(outputs.values()))
        if self.custom_postprocess:
            return self.custom_postprocess(raw, meta)
        return self.postprocess_default(raw, meta)


# Local import to avoid hard dependency at module import time
import cv2  # noqa: E402




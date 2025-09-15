import argparse
import os
import json
from datetime import datetime
from typing import Optional

import cv2

from ov_detector import OpenVINOObjectDetector
from processing import SizeEstimator, SizeEstimatorConfig, draw_detections, append_json_lines


def build_estimator(ppm: float = 0.0) -> SizeEstimator:
    """Простая оценка физических размеров по глобальному PPM (пикс/метр)."""
    return SizeEstimator(SizeEstimatorConfig(pixels_per_meter=float(ppm), per_camera_ppm=None))


def process_stream(source: str, model: str, labels: Optional[str], device: str, out_dir: str,
                   ppm: float, display: bool, save_video: bool) -> None:
    """Захват видео (IVcam/файл/индекс), детекция и сохранение результатов локально."""
    class_names = None
    if labels and os.path.exists(labels):
        with open(labels, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]

    detector = OpenVINOObjectDetector(model_path=model, device=device, labels=class_names)
    estimator = build_estimator(ppm)

    os.makedirs(out_dir, exist_ok=True)
    is_index = str(source).isdigit()
    cap = cv2.VideoCapture(int(source) if is_index else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        out_path = os.path.join(out_dir, "output.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    log_path = os.path.join(out_dir, "detections.jsonl")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.infer(frame)
        sizes = [estimator.estimate(det["bbox"]) for det in detections]

        ts = datetime.utcnow().isoformat()
        records = []
        for det, sz in zip(detections, sizes):
            x1, y1, x2, y2 = det["bbox"]
            records.append(
                {
                    "timestamp": ts,
                    "source": source,
                    "class_id": det.get("class_id"),
                    "class_name": det.get("class_name"),
                    "score": det.get("score"),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "size_m": sz,
                }
            )
        if records:
            append_json_lines(log_path, records)

        vis = draw_detections(frame.copy(), detections, sizes)
        if writer is not None:
            writer.write(vis)
        if display:
            cv2.imshow("Detections", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()


def main():
    """CLI: без аргументов читает конфиг config.json и запускается автоматически."""
    parser = argparse.ArgumentParser(description="Object Detection (OpenVINO) CLI")
    parser.add_argument("--source", help="Путь к видео/камере: индекс (0), IVcam URL, файл")
    parser.add_argument("--model", help="Путь к модели OpenVINO (.xml или .onnx)")
    parser.add_argument("--labels", default=None, help="Путь к labels.txt (по строке на класс)")
    parser.add_argument("--device", default=None, help="OpenVINO устройство: CPU/GPU")
    parser.add_argument("--out_dir", default=None, help="Папка для логов и видео")
    parser.add_argument("--ppm", type=float, default=None, help="Пиксели в метре для оценки размеров")
    parser.add_argument("--display", action="store_true", help="Показывать окно с результатом")
    parser.add_argument("--save_video", action="store_true", help="Сохранять видео с аннотациями")
    args = parser.parse_args()

    # Загружаем конфиг из корня проекта
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cfg_path = os.path.join(base_dir, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}

    source = args.source if args.source is not None else str(cfg.get("SOURCE", "0"))
    model = args.model if args.model is not None else cfg.get("MODEL_PATH")
    labels = args.labels if args.labels is not None else cfg.get("LABELS_PATH")
    device = args.device if args.device is not None else cfg.get("DEVICE", "CPU")
    out_dir = args.out_dir if args.out_dir is not None else cfg.get("OUT_DIR", os.path.join(base_dir, "outputs"))
    ppm = args.ppm if args.ppm is not None else float(cfg.get("PPM", 0.0))
    display = bool(args.display or cfg.get("DISPLAY", False))
    save_video = bool(args.save_video or cfg.get("SAVE_VIDEO", False))

    if not model:
        raise SystemExit("Не задан путь к модели. Укажите --model или ключ MODEL_PATH в config.json")

    process_stream(
        source=source,
        model=model,
        labels=labels,
        device=device,
        out_dir=out_dir,
        ppm=ppm,
        display=display,
        save_video=save_video,
    )


if __name__ == "__main__":
    main()




import cv2
import onnxruntime as ort
import numpy as np
import argparse
import os
from pathlib import Path
from camera import get_ivcam_stream
from utils import preprocess, postprocess, draw_boxes

def _resolve_model_path(model_path: str) -> str:
    p = Path(model_path)
    if p.exists():
        return str(p)

    script_dir = Path(__file__).resolve().parent
    candidates = []
    # Если путь относительный — пробуем относительно структуры проекта
    if not p.is_absolute():
        # 1) src/../models/<filename>
        candidates.append(script_dir.parent / "models" / p.name)
        # 2) src/../<provided_relative>
        candidates.append(script_dir.parent / p)
    for c in candidates:
        if c.exists():
            return str(c)
    return str(p)


def run_camera_inference(model_path, source=0, conf=0.5, iou=0.5, input_size="640,640", mode="auto"):
    # Подключение к IVCam/камере
    cap = get_ivcam_stream(source)

    # Загружаем модель
    resolved_model = _resolve_model_path(model_path)
    session = ort.InferenceSession(resolved_model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Небольшой прогрев камеры и автоэкспозиции
    for _ in range(5):
        ret, _ = cap.read()
        if not ret:
            break

    print("Starting inference from camera. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Препроцессинг
        # Парсим размер входа
        try:
            w_str, h_str = str(input_size).split(",")
            in_w, in_h = int(w_str), int(h_str)
        except Exception:
            in_w, in_h = 640, 640

        blob, scale = preprocess(frame, input_shape=(in_h, in_w))

        # Инференс
        outputs = session.run(None, {input_name: blob})

        # Постпроцессинг с настраиваемыми порогами
        results = postprocess(outputs, scale, conf_thres=conf, iou_thres=iou, input_size=(in_h, in_w), mode=mode)

        # Отрисовка боксов
        frame_vis = draw_boxes(frame.copy(), results)

        # Показ результата
        cv2.imshow("YOLOX Camera Inference", frame_vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Path to ONNX model")
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0/1) or URL (rtsp/http)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--input-size", default="640,640", help="Model input size 'W,H' (e.g. 640,640)")
    parser.add_argument("--decode", choices=["auto", "raw", "decoded"], default="auto", help="YOLOX output decode mode")
    args = parser.parse_args()

    src = args.source
    # Преобразуем числовую строку в int для корректной работы бэкендов устройств
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    run_camera_inference(args.model, src, conf=args.conf, iou=args.iou, input_size=args.input_size, mode=args.decode)

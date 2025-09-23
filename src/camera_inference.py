import cv2
import numpy as np
import argparse
from camera import get_ivcam_stream
from d2_predictor import build_detectron2_predictor, run_detectron2_inference, draw_detections_bgr

def _noop(*args, **kwargs):
    return None


def run_camera_inference(source=0, conf=0.5, device="cpu", config_name="COCO-Detection/retinanet_R_50_FPN_3x.yaml", weights_path: str | None = None):
    # Подключение к IVCam/камере
    cap = get_ivcam_stream(source)

    # Загружаем предиктор Detectron2
    predictor, class_names = build_detectron2_predictor(config_name=config_name, score_threshold=conf, device=device, weights_path=weights_path)

    # Небольшой прогрев камеры и автоэкспозиции
    for _ in range(5):
        ret, _ = cap.read()
        if not ret:
            break

    print("Starting Detectron2 inference from camera. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Инференс Detectron2 + фильтрация (без людей и животных)
        results = run_detectron2_inference(predictor, frame, class_names)

        # Отрисовка боксов
        frame_vis = draw_detections_bgr(frame, results, class_names)

        # Показ результата
        cv2.imshow("Detectron2 Camera Inference", frame_vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0/1) or URL (rtsp/http)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Inference device")
    parser.add_argument("--config", default="COCO-Detection/retinanet_R_50_FPN_3x.yaml", help="Detectron2 config name from model zoo")
    parser.add_argument("--weights", required=True, help="Path to Detectron2 weights (.pth/.pkl) with commercial rights")
    args = parser.parse_args()

    src = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    run_camera_inference(src, conf=args.conf, device=args.device, config_name=args.config, weights_path=args.weights)

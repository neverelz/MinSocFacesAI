import cv2
import onnxruntime as ort
import numpy as np
import argparse
from camera import get_ivcam_stream
from utils import preprocess, postprocess, draw_boxes

def run_camera_inference(model_path, source_url=0):
    # Подключение к IVcam
    cap = get_ivcam_stream(cv2.VideoCapture(source_url))

    # Загружаем модель
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    print("Starting inference from camera. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Препроцессинг
        blob, scale = preprocess(frame)

        # Инференс
        outputs = session.run(None, {input_name: blob})

        # Постпроцессинг
        results = postprocess(outputs, scale)

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
    parser.add_argument("--url", default=cv2.VideoCapture(0), help="IVcam source URL")
    args = parser.parse_args()

    run_camera_inference(args.model, args.url)

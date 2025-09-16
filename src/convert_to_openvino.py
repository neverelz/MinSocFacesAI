import argparse
import os
import cv2
import numpy as np
import openvino as ov

from utils import preprocess, postprocess, draw_boxes


def convert_model(onnx_path, xml_path, fp16=False):
    """Конвертация ONNX -> OpenVINO IR"""
    core = ov.Core()
    print(f"[INFO] Converting {onnx_path} to IR...")
    model = ov.convert_model(onnx_path)
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    ov.save_model(model, xml_path, compress_to_fp16=fp16)
    print(f"[INFO] Saved IR model at {xml_path} (+.bin)")


def run_inference_image(xml_path, img_path, output_dir):
    """Инференс на статичном изображении"""
    core = ov.Core()
    model = core.read_model(xml_path)
    compiled = core.compile_model(model, "CPU")
    input_layer = compiled.input(0)

    # Загружаем картинку
    img = cv2.imread(img_path)
    blob, scale = preprocess(img)

    # Запуск инференса
    result = compiled([blob])[compiled.output(0)]

    # Постпроцессинг и отрисовка
    results = postprocess([result], scale)
    vis = draw_boxes(img.copy(), results)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "result_openvino.jpg")
    cv2.imwrite(out_path, vis)
    print(f"[INFO] Saved result to {out_path}")


def run_inference_camera(xml_path, source_url="http://127.0.0.1:8090/video"):
    """Инференс с IVcam"""
    core = ov.Core()
    model = core.read_model(xml_path)
    compiled = core.compile_model(model, "CPU")
    input_layer = compiled.input(0)

    cap = cv2.VideoCapture(source_url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera stream: {source_url}")

    print("[INFO] Starting camera inference (press 'q' to exit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            break

        blob, scale = preprocess(frame)
        result = compiled([blob])[compiled.output(0)]
        results = postprocess([result], scale)
        vis = draw_boxes(frame.copy(), results)

        cv2.imshow("OpenVINO Camera Inference", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and run YOLOX with OpenVINO")
    parser.add_argument("-i", "--input", required=True, help="Path to ONNX model or IR (.xml)")
    parser.add_argument("-o", "--output", default="models/yolox_s.xml", help="Output IR path (.xml)")
    parser.add_argument("--img", help="Run inference on a single image")
    parser.add_argument("--cam", action="store_true", help="Run inference on IVcam stream")
    parser.add_argument("--outdir", default="outputs", help="Directory for results")
    parser.add_argument("--fp16", action="store_true", help="Save IR in FP16 format")

    args = parser.parse_args()

    if args.input.endswith(".onnx"):
        convert_model(args.input, args.output, fp16=args.fp16)
        xml_path = args.output
    else:
        xml_path = args.input

    if args.img:
        run_inference_image(xml_path, args.img, args.outdir)

    if args.cam:
        run_inference_camera(xml_path)

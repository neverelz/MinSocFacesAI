import onnxruntime as ort
import numpy as np
import cv2
import os
import argparse

def preprocess(img_path, input_shape=(640, 640)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_resized = cv2.resize(img, input_shape)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)
    return img_input, img

def run_inference(model_path, img_path, output_dir):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    img_input, orig_img = preprocess(img_path)

    outputs = session.run(None, {input_name: img_input})
    print("Inference completed. Outputs:", [o.shape for o in outputs])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "result.jpg")
    cv2.imwrite(out_path, orig_img)
    print(f"Saved result image to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Path to ONNX model")
    parser.add_argument("-i", "--image", required=True, help="Path to input image")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    args = parser.parse_args()

    run_inference(args.model, args.image, args.output)

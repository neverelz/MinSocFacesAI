# main.py
import cv2
import argparse
from config import CAMERA_INDEX, ONNX_MODEL_PATH, EXCLUDE_CLASSES, CONFIDENCE_THRESHOLD
from classes_coco import COCO_CLASSES
from model import ONNXModel
from camera import Camera

def draw_detections(frame, detections, class_names, color=(0,255,0)):
    for det in detections:
        x1,y1,x2,y2 = det["bbox"]
        cls_id = det["class_id"]
        score = det["score"]
        label = f"{class_names[cls_id]} {score:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(frame, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 6, y1), color, -1)
        cv2.putText(frame, label, (x1+3, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None, help="camera index override (0 or 1)")
    parser.add_argument("--model", type=str, default=None, help="path to onnx model")
    args = parser.parse_args()

    cam_idx = CAMERA_INDEX if args.camera is None else args.camera
    model_path = ONNX_MODEL_PATH if args.model is None else args.model

    print(f"Using camera {cam_idx}, model {model_path}")

    cam = Camera(cam_idx)
    model = ONNXModel(model_path)

    exclude_set = set(EXCLUDE_CLASSES)
    names = COCO_CLASSES

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("Camera read failed, exiting")
                break
            detections = model.infer(frame)
            # Filter exclude classes
            filtered = []
            for d in detections:
                cls_name = names[d["class_id"]] if d["class_id"] < len(names) else str(d["class_id"])
                if cls_name in exclude_set:
                    continue
                filtered.append(d)
            draw_detections(frame, filtered, names)
            cv2.imshow("Detections (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# object_in_hand.py
import cv2
import numpy as np
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

class DangerousObjectInHandDetector:
    def __init__(self, confidence_threshold=0.4, use_gpu=False):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold

        # Загружаем предобученную модель на COCO
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights, score_thresh=confidence_threshold)
        self.model.to(self.device).eval()
        self.transform = self.weights.transforms()

        # Классы COCO, которые считаются "потенциально опасными бытовыми предметами"
        self.dangerous_classes = {
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            76: "scissors",
            57: "chair",  # только если поднят (проверяется через IoU с рукой)
        }

        # Полный список имён классов COCO
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

        print(f"✅ DangerousObjectInHandDetector инициализирован на {self.device.type.upper()}")

    def detect_all(self, frame):
        # Уменьшаем размер для скорости
        small = cv2.resize(frame, (320, 320))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        tensor = self.transform(torch.from_numpy(rgb).permute(2, 0, 1))
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(tensor)[0]

        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy().astype(int)

        # Масштабируем обратно на оригинальный размер
        h_orig, w_orig = frame.shape[:2]
        boxes[:, [0, 2]] *= w_orig / 320
        boxes[:, [1, 3]] *= h_orig / 320

        return boxes, scores, labels

    def get_dangerous_objects_in_hand(self, frame, hand_boxes, iou_threshold=0.1):
        obj_boxes, obj_scores, obj_labels = self.detect_all(frame)
        dangerous_objects = []

        for i, (box, score, label) in enumerate(zip(obj_boxes, obj_scores, obj_labels)):
            if score < self.confidence_threshold:
                continue
            if label not in self.dangerous_classes:
                continue

            in_hand = False
            for hand_box in hand_boxes:
                if self._iou(box, hand_box) > iou_threshold:
                    in_hand = True
                    break

            if in_hand:
                obj_name = self.class_names[label] if label < len(self.class_names) else f"cls{label}"
                dangerous_objects.append({
                    'bbox': box,
                    'score': score,
                    'label': label,
                    'name': obj_name,
                    'type': 'lifted' if label == 57 else 'handheld'
                })

        return dangerous_objects

    @staticmethod
    def _iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0
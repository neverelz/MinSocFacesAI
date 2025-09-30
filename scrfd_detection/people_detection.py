# people_detection.py

import cv2
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import numpy as np
import time


class HybridPeopleDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu = (self.device.type == 'cuda')

        if self.use_gpu:
            print("🚀 Используется GPU — запускаем гибридный детектор с трекингом")
            self._init_gpu_detector()
        else:
            print("💻 Используется CPU — запускаем упрощённый SSD-детектор")
            self._init_cpu_detector()

    def _init_gpu_detector(self):
        """Гибридная версия: нейросеть + простой трекинг (для GPU)"""
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights, score_thresh=0.4)
        self.model.to(self.device)
        self.model.eval()
        self.transform = self.weights.transforms()
        self.detection_interval = 10
        self.frame_count = 0
        self.trackers = []

    def _init_cpu_detector(self):
        """Простая версия без трекинга (для CPU)"""
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights, score_thresh=0.4)
        self.model.to(self.device)
        self.model.eval()
        self.transform = self.weights.transforms()

    def detect(self, frame):
        if self.use_gpu:
            return self._detect_gpu(frame)
        else:
            return self._detect_cpu(frame)

    # =============== GPU ВЕРСИЯ (гибридная) ===============
    def _detect_gpu(self, frame):
        self.frame_count += 1

        if self.frame_count % self.detection_interval == 0:
            return self._neural_detect(frame)
        else:
            return self._track_people(frame)

    def _neural_detect(self, frame):
        small_img = cv2.resize(frame, (320, 320))
        image_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(torch.from_numpy(image_rgb).permute(2, 0, 1))
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        people_indices = (labels == 1) & (scores > 0.3)
        people_boxes = boxes[people_indices]
        people_scores = scores[people_indices]

        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 320
        people_boxes[:, [0, 2]] *= scale_x
        people_boxes[:, [1, 3]] *= scale_y
        people_boxes = people_boxes.astype(int)

        self._update_trackers(people_boxes, people_scores)
        return people_boxes, people_scores

    def _update_trackers(self, new_boxes, new_scores):
        self.trackers = []
        for box, score in zip(new_boxes, new_scores):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            self.trackers.append({
                'box': [x1, y1, w, h],
                'score': score,
                'missed_frames': 0
            })

    def _track_people(self, frame):
        if not self.trackers:
            return np.array([]), np.array([])

        current_boxes = []
        current_scores = []

        for tracker in self.trackers:
            x, y, w, h = tracker['box']
            current_boxes.append([x, y, x + w, y + h])
            current_scores.append(tracker['score'])

        return np.array(current_boxes), np.array(current_scores)

    # =============== CPU ВЕРСИЯ (простая) ===============
    def _detect_cpu(self, frame):
        small_img = cv2.resize(frame, (320, 320))
        image_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(torch.from_numpy(image_rgb).permute(2, 0, 1))
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        people_indices = (labels == 1) & (scores > 0.3)
        people_boxes = boxes[people_indices]
        people_scores = scores[people_indices]

        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 320
        people_boxes[:, [0, 2]] *= scale_x
        people_boxes[:, [1, 3]] *= scale_y
        people_boxes = people_boxes.astype(int)

        return people_boxes, people_scores
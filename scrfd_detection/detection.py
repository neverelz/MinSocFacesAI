# detection.py

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, model_name='scrfd_10g_kps', device_id=0):
        """
        Инициализация детектора лиц SCRFD.
        model_name: 'scrfd_2.5g', 'scrfd_10g_kps' и т.д.
        device_id: 0 — GPU (если доступно), -1 — CPU
        """
        self.app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=device_id, det_size=(640, 640))

    def detect(self, frame):
        """
        Обнаружение лиц на кадре.
        Возвращает: список лиц с bounding box, ключевыми точками и уверенностью.
        """
        faces = self.app.get(frame)
        return faces

    def draw_faces(self, frame, faces):
        """
        Рисует bounding box и ключевые точки на кадре.
        """
        for face in faces:
            # Bounding box
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # Confidence
            conf = face.det_score
            cv2.putText(frame, f'{conf:.2f}', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Key points (eyes, nose, etc.)
            if hasattr(face, 'kps'):
                kps = face.kps.astype(int)
                for i in range(kps.shape[0]):
                    cv2.circle(frame, (kps[i][0], kps[i][1]), 3, (255, 0, 0), -1)
        return frame
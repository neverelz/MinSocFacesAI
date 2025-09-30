# detection.py

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, model_name='scrfd_10g_kps', use_gpu=True, det_size=(640, 640)):
        """
        use_gpu: True — попытка использовать CUDA, False — только CPU
        det_size: разрешение для детекции (меньше = быстрее на CPU)
        """
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.app = FaceAnalysis(name=model_name, providers=providers)
        ctx_id = 0 if use_gpu else -1
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.use_gpu = use_gpu
        self.det_size = det_size

    def detect(self, frame):
        return self.app.get(frame)
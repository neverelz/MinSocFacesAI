# camera.py — асинхронный захват с логирование

import cv2
import threading
import time
import os
import logging
from queue import Queue, Full, Empty
from platform_utils import get_optimal_camera_backends, safe_makedirs

# Гарантируем, что папка logs существует (на случай, если запущен отдельно)
LOG_DIR = safe_makedirs("logs", exist_ok=True)

class AsyncCameraReader:
    def __init__(self, camera_index: int, width: int = 640, height: int = 480, fps: int = 15, queue_size: int = 3):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.queue_size = queue_size
        self.frame_queue = Queue(maxsize=queue_size)
        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()

        # Настройка логгера для этой камеры
        self.logger = logging.getLogger(f"camera_reader_{camera_index}")
        # Убираем дублирование, если логгер уже существует
        if not self.logger.handlers:
            handler = logging.FileHandler(os.path.join(LOG_DIR, f"camera_{camera_index}.log"), encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def open(self):
        backends = get_optimal_camera_backends()
        for backend in backends:
            self.cap = cv2.VideoCapture(self.camera_index, backend)
            if not self.cap.isOpened():
                continue

            for _ in range(3):
                ret, _ = self.cap.read()
                if ret:
                    self.logger.info(f"✅ Открыта через backend {backend}")
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                    return True
                time.sleep(0.1)

            self.cap.release()

        self.logger.error("❌ Не удалось открыть")
        return False

    def start(self):
        if not self.open():
            return False

        def run():
            frame_count = 0
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame_count += 1
                if frame_count % 30 == 0:
                    self.logger.info(f"📹 Захвачено {frame_count} кадров")

                try:
                    self.frame_queue.put_nowait(frame)
                except Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except (Empty, Full):
                        pass

            if self.cap and self.cap.isOpened():
                self.cap.release()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self.logger.info("▶️ Поток захвата запущен")
        return True

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def get_frame(self, timeout=0.5):
        try:
            return self.frame_queue.get(timeout=timeout)
        except Exception:
            return None
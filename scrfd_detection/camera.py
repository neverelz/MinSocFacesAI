# camera.py

import cv2
import threading
import time
from queue import Queue, Full, Empty


class AsyncCameraReader:
    def __init__(self, camera_index: int, width: int = 1280, height: int = 720, fps: int = 30, queue_size: int = 3):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.queue_size = queue_size
        self.frame_queue = Queue(maxsize=queue_size)
        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()

    def open(self):
        # Для iVCam (индекс 1) пробуем другие бэкенды
        if self.camera_index == 1:
            backends = [cv2.CAP_MSMF, cv2.CAP_ANY, cv2.CAP_DSHOW]
            attempts = 5
            delay = 0.3
        else:
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            attempts = 3
            delay = 0.1

        for backend in backends:
            self.cap = cv2.VideoCapture(self.camera_index, backend)
            if not self.cap.isOpened():
                continue

            # Пробуем прочитать кадр несколько раз
            for attempt in range(attempts):
                ret, _ = self.cap.read()
                if ret:
                    print(f"✅ [Камера {self.camera_index}] Открыта через backend {backend} (попытка {attempt + 1})")
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                    return True
                time.sleep(delay)

            self.cap.release()

        print(f"❌ [Камера {self.camera_index}] Не удалось открыть ни через один backend")
        return False

    def start(self):
        if not self.open():
            return False

        def run():
            frame_count = 0
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.05 if self.camera_index == 1 else 0.01)
                    continue

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"📹 [Камера {self.camera_index}] Захвачено {frame_count} кадров")

                try:
                    self.frame_queue.put_nowait(frame)
                except Full:
                    # Для iVCam — не сбрасываем агрессивно
                    if self.camera_index == 1:
                        time.sleep(0.02)
                        continue
                    else:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except (Empty, Full):
                            pass

            if self.cap.isOpened():
                self.cap.release()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        print(f"▶️  [Камера {self.camera_index}] Поток захвата запущен")
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
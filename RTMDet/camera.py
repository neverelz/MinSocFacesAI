# camera.py
import cv2

class Camera:
    def __init__(self, cam_index=0, width=1280, height=720):
        self.cam_index = cam_index
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW if hasattr(cv2,'CAP_DSHOW') else 0)
        # задаём размер захвата (можно менять)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cam_index}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def release(self):
        self.cap.release()

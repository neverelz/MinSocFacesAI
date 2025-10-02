import cv2
from ultralytics import YOLO
import numpy as np

class FaceDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.7, device_index=1):
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_path)
        # Пытаемся открыть iVCam через DirectShow (Windows)
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)

        # Если не открылась, пробуем перебрать несколько индексов и бэкендов
        if not self.cap.isOpened():
            try:
                self.cap.release()
            except Exception:
                pass
            opened = False
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
                for idx in (1, 2, 3, 0, 4, 5):
                    cap_try = cv2.VideoCapture(idx, backend)
                    if cap_try.isOpened():
                        ret, _ = cap_try.read()
                        if ret:
                            self.cap = cap_try
                            opened = True
                            break
                        cap_try.release()4
                if opened:
                    break
            if not opened:
                raise Exception("Не удалось открыть iVCam. Убедитесь, что iVCam запущен.")
    
    def detect_faces(self, frame):
        results = self.model(frame, verbose=False)
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        
        filtered_boxes = []
        filtered_confidences = []
        
        for i, confidence in enumerate(confidences):
            if confidence >= self.confidence_threshold:
                filtered_boxes.append(boxes[i])
                filtered_confidences.append(confidence)
        
        return np.array(filtered_boxes), np.array(filtered_confidences)
    
    def draw_boxes(self, frame, boxes, confidences):
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            confidence = confidences[i]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Face: {confidence:.3f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        print("Запуск детектора лиц. Нажмите 'q' для выхода.")
        print(f"Порог уверенности: {self.confidence_threshold}")
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Не удалось получить кадр с камеры (iVCam)")
                break
            
            boxes, confidences = self.detect_faces(frame)
            
            if len(boxes) > 0:
                frame = self.draw_boxes(frame, boxes, confidences)
                print(f"Обнаружено лиц: {len(boxes)} с уверенностью > {self.confidence_threshold}")
            
            cv2.imshow('iVCam Face Detection (Confidence > 0.85)', frame)
            
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # По умолчанию пытаемся открыть индекс 1, где часто находится iVCam
        detector = FaceDetector(confidence_threshold=0.7, device_index=1)
        detector.run()
    
    except Exception as e:
        print(f"Ошибка: {e}")
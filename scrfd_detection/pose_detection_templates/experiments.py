import cv2
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import numpy as np
import time

class HybridPeopleDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights, score_thresh=0.4)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = self.weights.transforms()
        self.last_detection_time = 0
        self.detection_interval = 10
        self.trackers = []
        self.frame_count = 0

        self.prev_gray = None
        
        print("Гибридный детектор загружен!")
    
    def detect(self, frame):
        self.frame_count += 1
        
        if self.frame_count % self.detection_interval == 0:
            return self.neural_detect(frame)
        else:
            return self.track_people(frame)
    
    def neural_detect(self, frame):
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
        
        self.update_trackers(people_boxes, people_scores)
        
        return people_boxes, people_scores
    
    def update_trackers(self, new_boxes, new_scores):
        self.trackers = []
        for box, score in zip(new_boxes, new_scores):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            self.trackers.append({
                'box': [x1, y1, w, h],
                'score': score,
                'missed_frames': 0
            })
    
    def track_people(self, frame):
        if not self.trackers:
            return np.array([]), np.array([])
        
        current_boxes = []
        current_scores = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for tracker in self.trackers:
            x, y, w, h = tracker['box']
            current_boxes.append([x, y, x + w, y + h])
            current_scores.append(tracker['score'])
        
        return np.array(current_boxes), np.array(current_scores)

def main():
    detector = HybridPeopleDetector()
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    fps_time = time.time()
    frame_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        boxes, scores = detector.detect(frame)
        detection_time = time.time() - start_time
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{score:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()
        
        mode = "NEURAL" if detector.frame_count % detector.detection_interval == 0 else "TRACKING"
        info = f"FPS: {fps} | Mode: {mode} | People: {len(boxes)}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Hybrid People Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#Гибрид для работы на цпу

'''import cv2
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms import functional as F
import numpy as np
import time

class FastPeopleDetector:
    def __init__(self, confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        
        # Используем легкую SSD модель
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights, score_thresh=confidence_threshold)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = self.weights.transforms()
        print("Быстрая SSD модель загружена!")
    
    def detect(self, image):
        # Уменьшаем размер изображения для скорости
        small_img = cv2.resize(image, (320, 320))
        image_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        
        image_tensor = self.transform(torch.from_numpy(image_rgb).permute(2, 0, 1))
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        return self.process_predictions(predictions[0], image.shape)
    
    def process_predictions(self, prediction, original_shape):
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        
        # Фильтруем людей (класс 1)
        people_indices = (labels == 1) & (scores > 0.3)
        people_boxes = boxes[people_indices]
        people_scores = scores[people_indices]
        
        # Масштабируем координаты обратно к оригинальному размеру
        scale_x = original_shape[1] / 320
        scale_y = original_shape[0] / 320
        
        people_boxes[:, [0, 2]] *= scale_x
        people_boxes[:, [1, 3]] *= scale_y
        people_boxes = people_boxes.astype(int)
        
        return people_boxes, people_scores

def main():
    detector = FastPeopleDetector(confidence_threshold=0.4)
    cap = cv2.VideoCapture(0)
    
    # Устанавливаем меньший размер для скорости
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Включаем оптимизации CUDA если доступно
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    fps_time = time.time()
    frame_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        boxes, scores = detector.detect(frame)
        detection_time = time.time() - start_time
        
        # Рисуем результаты
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{score:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Считаем FPS
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()
        
        # Выводим информацию
        info = f"FPS: {fps} | Detection: {detection_time*1000:.1f}ms | People: {len(boxes)}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Fast People Detector (SSD)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()'''

#Faster-R CNN для CUDA
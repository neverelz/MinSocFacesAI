import cv2
import mediapipe as mp
import numpy as np

class MultiPersonDetector:
    def __init__(self, max_people=6):
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.max_people = max_people
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_people*2,  # До 2 рук на человека
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )

        self.colors = [
            (0, 255, 0),    # Зеленый
            (0, 0, 255),    # Красный
            (255, 0, 0),    # Синий
            (255, 255, 0),  # Голубой
            (255, 0, 255),  # Розовый
            (0, 255, 255),  # Желтый
            (255, 165, 0),  # Оранжевый
            (128, 0, 128),  # Фиолетовый
            (0, 128, 0),    # Темно-зеленый
            (128, 128, 0)   # Оливковый
        ]
    
    def get_person_bbox(self, pose_landmarks, hand_landmarks_list, frame_shape):
        """Рассчитывает bounding box для всего человека"""
        h, w, _ = frame_shape
        all_points = []
        
        # Добавляем точки позы (если есть)
        if pose_landmarks:
            for landmark in pose_landmarks.landmark:
                if landmark.visibility > 0.5:  # Только видимые точки
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    all_points.append((x, y))
        
        # Добавляем точки рук (если есть)
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    all_points.append((x, y))
        
        if not all_points:
            return None
        
        # Рассчитываем bounding box
        x_coords = [point[0] for point in all_points]
        y_coords = [point[1] for point in all_points]
        
        # Добавляем отступы
        padding = 20
        x_min = max(0, min(x_coords) - padding)
        y_min = max(0, min(y_coords) - padding)
        x_max = min(w, max(x_coords) + padding)
        y_max = min(h, max(y_coords) + padding)
        
        return (x_min, y_min, x_max, y_max)
    
    def get_hand_bbox(self, hand_landmarks, frame_shape):
        """Рассчитывает bounding box для отдельной руки"""
        h, w, _ = frame_shape
        
        landmarks_array = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_array.append((x, y))
        
        x_coords = [point[0] for point in landmarks_array]
        y_coords = [point[1] for point in landmarks_array]
        
        padding = 15
        x_min = max(0, min(x_coords) - padding)
        y_min = max(0, min(y_coords) - padding)
        x_max = min(w, max(x_coords) + padding)
        y_max = min(h, max(y_coords) + padding)
        
        return (x_min, y_min, x_max, y_max)
    
    def detect_people(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Детекция позы человека
        holistic_results = self.holistic.process(rgb_frame)
        
        # Детекция рук
        hands_results = self.hands.process(rgb_frame)
        
        person_count = 0
        person_info = []
        hand_info = []
        
        # Если обнаружена поза, считаем что есть хотя бы один человек
        if holistic_results.pose_landmarks:
            person_count = 1  # MediaPipe Holistic обнаруживает одного человека
            
            # Получаем bounding box для человека
            hand_landmarks_list = []
            if hands_results.multi_hand_landmarks:
                hand_landmarks_list = hands_results.multi_hand_landmarks
            
            person_bbox = self.get_person_bbox(
                holistic_results.pose_landmarks, 
                hand_landmarks_list, 
                frame.shape
            )
            
            if person_bbox:
                color = self.colors[0]  # Первый цвет для первого человека
                x_min, y_min, x_max, y_max = person_bbox
                
                # Рисуем bounding box для человека (толстый)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
                
                # Рисуем заполненный прямоугольник для подписи человека
                cv2.rectangle(frame, (x_min, y_min-35), (x_min+150, y_min), color, -1)
                cv2.putText(frame, f'Person 1', (x_min+5, y_min-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Рисуем landmarks позы
                self.mp_drawing.draw_landmarks(
                    frame,
                    holistic_results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                person_info.append({
                    'id': 0,
                    'bbox': person_bbox,
                    'color': color
                })
        
        # Обрабатываем руки отдельно
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for i, (hand_landmarks, handedness) in enumerate(zip(
                hands_results.multi_hand_landmarks, 
                hands_results.multi_handedness
            )):
                color = self.colors[(i+1) % len(self.colors)]  # Смещаем цвета для рук
                hand_label = handedness.classification[0].label
                
                # Рисуем landmarks и соединения для руки
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=color, thickness=2, circle_radius=3
                    ),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=color, thickness=2
                    )
                )
                
                # Получаем и рисуем bounding box для руки
                hand_bbox = self.get_hand_bbox(hand_landmarks, frame.shape)
                if hand_bbox:
                    x_min, y_min, x_max, y_max = hand_bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    # Подпись для руки
                    cv2.rectangle(frame, (x_min, y_min-25), (x_min+120, y_min), color, -1)
                    cv2.putText(frame, f'{hand_label} {i+1}', (x_min+5, y_min-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Центр ладони
                h, w, _ = frame.shape
                cx = int(hand_landmarks.landmark[0].x * w)
                cy = int(hand_landmarks.landmark[0].y * h)
                cv2.circle(frame, (cx, cy), 8, color, -1)
                
                hand_info.append({
                    'id': i,
                    'label': hand_label,
                    'position': (cx, cy),
                    'bbox': hand_bbox,
                    'color': color
                })
        
        return frame, person_count, person_info, hand_info

def main():
    detector = MultiPersonDetector(max_people=6)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Распознавание людей и рук запущено")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        processed_frame, person_count, person_info, hand_info = detector.detect_people(frame)

        # Отображаем общую информацию
        cv2.putText(processed_frame, f'People: {person_count}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f'Hands: {len(hand_info)}', 
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f'+/-/q', 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 25, 25), 2)
        
        # Отображаем информацию о bounding boxes
        y_offset = 130
        for i, person in enumerate(person_info):
            bbox = person['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            bbox_info = f"Person {i+1}: {width}x{height}"
            cv2.putText(processed_frame, bbox_info, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, person['color'], 1)
            y_offset += 25
        
        for i, hand in enumerate(hand_info):
            if hand['bbox']:
                bbox = hand['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                bbox_info = f"Hand {i+1}: {width}x{height}"
                cv2.putText(processed_frame, bbox_info, 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand['color'], 1)
                y_offset += 25
        
        cv2.imshow('Multi-Person Detection', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') and detector.max_people < 10:
            detector.max_people += 1
            detector.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=detector.max_people*2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.3
            )
        elif key == ord('-') and detector.max_people > 1:
            detector.max_people -= 1
            detector.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=detector.max_people*2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.3
            )
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
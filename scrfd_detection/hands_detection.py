import cv2
import mediapipe as mp
import numpy as np

class MultiHandDetector:
    def __init__(self, max_hands=6, mirror_view=True):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.max_hands = max_hands
        self.mirror_view = mirror_view
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )

        self.colors = [
            (0, 255, 0),    
            (0, 0, 255),  
            (255, 0, 0),    
            (255, 255, 0), 
            (255, 0, 255), 
            (0, 255, 255),  
            (255, 165, 0),  
            (128, 0, 128),  
            (0, 128, 0),    
            (128, 128, 0)   
        ]
    
    def detect_hands(self, frame):
        # Если режим зеркала включен, обрабатываем на перевёрнутой копии
        work_frame = cv2.flip(frame, 1) if self.mirror_view else frame
        rgb_frame = cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_count = 0
        hand_info = []
        
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            
            for i, (hand_landmarks, handedness) in enumerate(zip(
                results.multi_hand_landmarks, 
                results.multi_handedness
            )):
                color = self.colors[i % len(self.colors)]

                hand_label = handedness.classification[0].label
                # При зеркальном отображении меняем метку для соответствия ожиданиям пользователя
                if self.mirror_view:
                    hand_label = 'Right' if hand_label == 'Left' else 'Left'

                self.mp_drawing.draw_landmarks(
                    work_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=color, thickness=2, circle_radius=3
                    ),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=color, thickness=2
                    )
                )

                h, w, _ = work_frame.shape
                cx = int(hand_landmarks.landmark[0].x * w)
                cy = int(hand_landmarks.landmark[0].y * h)

                hand_info.append({
                    'id': i,
                    'label': hand_label,
                    'position': (cx, cy),
                    'color': color
                })

                cv2.circle(work_frame, (cx, cy), 8, color, -1)
                cv2.putText(work_frame, f'{hand_label} {i+1}', (cx-40, cy-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Возвращаем кадр обратно в оригинальном виде, если он был зеркален
        out_frame = cv2.flip(work_frame, 1) if self.mirror_view else work_frame
        return out_frame, hand_count, hand_info

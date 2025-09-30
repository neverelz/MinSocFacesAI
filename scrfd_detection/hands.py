import cv2
import mediapipe as mp
import numpy as np

class MultiHandDetector:
    def __init__(self, max_hands=6):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.max_hands = max_hands
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                h, w, _ = frame.shape
                cx = int(hand_landmarks.landmark[0].x * w)
                cy = int(hand_landmarks.landmark[0].y * h)

                hand_info.append({
                    'id': i,
                    'label': hand_label,
                    'position': (cx, cy),
                    'color': color
                })

                cv2.circle(frame, (cx, cy), 8, color, -1)
                cv2.putText(frame, f'{hand_label} {i+1}', (cx-40, cy-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame, hand_count, hand_info

def main():
    detector = MultiHandDetector(max_hands=6)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Распознавание запущено")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        processed_frame, hand_count, hand_info = detector.detect_hands(frame)

        cv2.putText(processed_frame, f'Hands: {hand_count}/{detector.max_hands}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f'+/-/q', 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 25, 25), 2)
        
        cv2.imshow('Multi-Hand Detection', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') and detector.max_hands < 10:
            detector.max_hands += 1
            detector.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=detector.max_hands,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.3
            )
        elif key == ord('-') and detector.max_hands > 1:
            detector.max_hands -= 1
            detector.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=detector.max_hands,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.3
            )
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
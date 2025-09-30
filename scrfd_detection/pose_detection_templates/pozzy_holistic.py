import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def detect_multiple_people(image, pose_model, face_mesh_model, hands_model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = {}
    
    pose_results = pose_model.process(image_rgb)
    results['pose_landmarks'] = pose_results.pose_landmarks
    results['pose_world_landmarks'] = pose_results.pose_world_landmarks
    
    face_results = face_mesh_model.process(image_rgb)
    results['face_landmarks'] = face_results.multi_face_landmarks or []
    
    hand_results = hands_model.process(image_rgb)
    results['left_hand_landmarks'] = []
    results['right_hand_landmarks'] = []
    
    if hand_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            handedness = hand_results.multi_handedness[i].classification[0].label if hand_results.multi_handedness else 'Unknown'
            if handedness == 'Left':
                results['left_hand_landmarks'].append(hand_landmarks)
            else:
                results['right_hand_landmarks'].append(hand_landmarks)
    
    return results

def draw_styled_multiple_landmarks(image, results):

    pose_spec = mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4)
    pose_connection_spec = mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    
    face_spec = mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1)
    face_connection_spec = mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    
    left_hand_spec = mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4)
    left_hand_connection_spec = mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    
    right_hand_spec = mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4)
    right_hand_connection_spec = mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    
    if results['pose_landmarks']:
        mp_drawing.draw_landmarks(
            image, results['pose_landmarks'], mp_pose.POSE_CONNECTIONS,
            pose_spec, pose_connection_spec
        )
    
    if results['face_landmarks']:
        for i, face_landmarks in enumerate(results['face_landmarks']):
            mp_drawing.draw_landmarks(
                image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                face_spec, face_connection_spec
            )
    
    if results['left_hand_landmarks']:
        for hand_landmarks in results['left_hand_landmarks']:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                left_hand_spec, left_hand_connection_spec
            )
    
    if results['right_hand_landmarks']:
        for hand_landmarks in results['right_hand_landmarks']:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                right_hand_spec, right_hand_connection_spec
            )

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
    smooth_landmarks=True
) as pose_model, \
mp_face_mesh.FaceMesh(
    max_num_faces=6,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh_model, \
mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands_model:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        results = detect_multiple_people(frame, pose_model, face_mesh_model, hands_model)
        
        num_faces = len(results['face_landmarks'])
        num_left_hands = len(results['left_hand_landmarks'])
        num_right_hands = len(results['right_hand_landmarks'])
        has_pose = results['pose_landmarks'] is not None
        
        print(f"Pose: {has_pose}, Faces: {num_faces}, Left hands: {num_left_hands}, Right hands: {num_right_hands}")
        
        draw_styled_multiple_landmarks(frame, results)
        
        info_text = f'Faces: {num_faces} | Hands: {num_left_hands + num_right_hands}'
        if has_pose:
            info_text += ' | Pose detected'
        
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Multi-Person Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
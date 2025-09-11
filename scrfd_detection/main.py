# main.py — обновлённый

import cv2
from detection import FaceDetector
from recognizer import FaceRecognizer

def main():
    # Инициализация детектора лиц
    detector = FaceDetector(model_name='scrfd_10g_kps', device_id=0)  # GPU

    # Инициализация распознавателя
    recognizer = FaceRecognizer()  # автоматически загрузит модель и базу

    # Открываем камеру
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Не удалось открыть видеопоток.")
        return

    print("✅ Видеопоток запущен. Нажми 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Ошибка чтения кадра.")
            break

        frame = cv2.flip(frame, 1)  # зеркало
        faces = detector.detect(frame)

        # Отрисовка лиц и распознавание
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Вырезаем лицо
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            # Распознавание
            name, sim = recognizer.recognize(face_img)

            # Цвет рамки
            color = (0, 255, 0) if name != "Неизвестно" else (0, 0, 255)

            # Рамка
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Подпись
            label = f"{name} ({sim:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Ключевые точки (если есть)
            if hasattr(face, 'kps'):
                kps = face.kps.astype(int)
                for pt in kps:
                    cv2.circle(frame, tuple(pt), 3, (255, 0, 0), -1)

        # Общее количество лиц
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Face Recognition System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
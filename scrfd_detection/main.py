# main.py

import cv2
from detection import FaceDetector

def main():
    # Инициализация детектора
    detector = FaceDetector(model_name='scrfd_10g_kps', device_id=0)  # ctx_id=0 — GPU, -1 — CPU

    # Открываем видеопоток (обычно камера телефона — это 0 или 1)
    cap = cv2.VideoCapture(1)  # Попробуй 0, 1, 2... если не работает

    # Увеличим разрешение, если нужно (iVCam может поддерживать)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Не удалось открыть видеопоток. Проверь подключение камеры.")
        return

    print("✅ Видеопоток запущен. Нажми 'q', чтобы выйти.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Ошибка чтения кадра.")
            break

        # Зеркальное отражение (если нужно — как в большинстве веб-камер)
        frame = cv2.flip(frame, 1)

        # Детекция лиц
        faces = detector.detect(frame)

        # Отрисовка лиц
        frame = detector.draw_faces(frame, faces)

        # Отображение количества лиц
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Показ кадра
        cv2.imshow('Face Detection (SCRFD)', frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Очистка
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
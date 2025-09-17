# main.py — обновлённый

import cv2
import time
import os
import numpy as np
from detection import FaceDetector
from recognizer import FaceRecognizer
import sys


def align_face_by_kps(frame, kps, output_size=(112, 112)):
    """
    Выравнивает лицо по 5 ключевым точкам SCRFD до размера 112x112.
    kps: np.ndarray формы (5,2)
    """
    # Опорные точки из InsightFace для 112x112
    ref = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    src = kps.astype(np.float32)
    # Оценка преобразования похожести (similarity)
    transform = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)[0]
    if transform is None:
        return None
    aligned = cv2.warpAffine(frame, transform, output_size, flags=cv2.INTER_LINEAR)
    return aligned


class FaceTrack:
    def __init__(self, person_id: str):
        self.person_id = person_id
        self.last_saved_ts = 0.0
        self.last_bbox_center = None


class FaceSaver:
    def __init__(self, recognizer: FaceRecognizer, save_interval_sec: float = 2.0):
        self.recognizer = recognizer
        self.save_interval = save_interval_sec
        self.tracks = []  # список FaceTrack

    @staticmethod
    def _center_of_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _centers_close(c1, c2, max_dist=80):
        if c1 is None or c2 is None:
            return False
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return (dx * dx + dy * dy) <= (max_dist * max_dist)

    def _get_or_create_track(self, bbox_center, person_id_hint: str) -> FaceTrack:
        # Ищем существующий трек рядом с центром
        for tr in self.tracks:
            if self._centers_close(tr.last_bbox_center, bbox_center):
                # Обновим person_id, если появился известный вместо неизвестного
                if tr.person_id != person_id_hint and person_id_hint != "Неизвестно":
                    tr.person_id = person_id_hint
                tr.last_bbox_center = bbox_center
                return tr
        # Не нашли — создаём новый
        if person_id_hint == "Неизвестно":
            new_id = "Неизвестно"  # не генерируем числовой ID для чужих
        else:
            new_id = person_id_hint
        tr = FaceTrack(new_id)
        tr.last_bbox_center = bbox_center
        self.tracks.append(tr)
        return tr

    def maybe_save(self, bbox, face_img_bgr, person_id_hint: str, similarity: float):
        # Не сохраняем чужих и очень уверенные совпадения
        if person_id_hint == "Неизвестно":
            return None
        if similarity >= 0.9:
            return None
        now = time.time()
        center = self._center_of_bbox(bbox)
        tr = self._get_or_create_track(center, person_id_hint)
        if now - tr.last_saved_ts < self.save_interval:
            return None
        # Сохраняем и обновляем эмбеддинги онлайн
        saved_path = self.recognizer.add_image_to_person(tr.person_id, face_img_bgr)
        tr.last_saved_ts = now
        return saved_path


def main():
    # Выбор автоочистки перед запуском
    auto_clear = False
    try:
        ans = input("Очистить базу лиц и кэш эмбеддингов перед запуском? (y/N): ").strip().lower()
        auto_clear = ans in ("y", "yes", "д", "да")
    except Exception:
        pass

    # Инициализация детектора лиц
    detector = FaceDetector(model_name='scrfd_10g_kps', device_id=0)  # GPU

    # Инициализация распознавателя
    recognizer = FaceRecognizer(force_rebuild=auto_clear)  # автоматически загрузит модель и базу
    if auto_clear:
        # Дополнительно гарантируем очистку на диске и в памяти
        recognizer.clear_database_and_cache()
        # После очистки база пустая; при первом сохранении ID начнётся с 1
    saver = FaceSaver(recognizer, save_interval_sec=2.0)

    # Открываем камеру
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Не удалось открыть видеопоток.")
        return

    WINDOW_NAME = 'Face Recognition System'
    print("✅ Видеопоток запущен. Нажми 'q' для выхода. Кликни по боксу 'Неизвестно' чтобы добавить в базу.")

    status_text = ""
    status_until = 0
    current_faces = []  # список словарей: {bbox, name, sim, face_img}

    # Обработчик кликов мыши
    def on_mouse(event, x, y, flags, userdata=None):
        nonlocal status_text, status_until, current_faces
        if event == cv2.EVENT_LBUTTONDOWN:
            # Найти бокс, по которому кликнули (сверху вниз)
            for item in current_faces:
                (bx1, by1, bx2, by2) = item['bbox']
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    if item['name'] == "Неизвестно":
                        new_id = recognizer.get_next_person_id()
                        recognizer.add_image_to_person(new_id, item['face_img'])
                        status_text = f"Добавлен ID {new_id}"
                        status_until = time.time() + 2.0
                        print(f"✅ Добавлен новый человек с ID {new_id}")
                    else:
                        status_text = f"Уже в базе: {item['name']}"
                        status_until = time.time() + 1.5
                    break

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Ошибка чтения кадра.")
            break

        frame = cv2.flip(frame, 1)  # зеркало
        faces = detector.detect(frame)

        # Сброс списка текущих лиц
        current_faces = []

        # Собираем лица для батча
        face_patches = []
        face_bboxes = []
        face_kps = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            raw_face = frame[y1:y2, x1:x2]
            if raw_face.size == 0:
                continue
            patch = raw_face
            if hasattr(face, 'kps') and face.kps is not None and np.array(face.kps).shape == (5, 2):
                aligned = align_face_by_kps(frame, face.kps)
                if aligned is not None:
                    patch = aligned
            face_patches.append(patch)
            face_bboxes.append((x1, y1, x2, y2))

        # Батч-распознавание
        names = []
        sims = []
        if face_patches:
            names, sims = recognizer.recognize_batch(face_patches)

        # Отрисовка результатов
        for i, (x1, y1, x2, y2) in enumerate(face_bboxes):
            face_img = face_patches[i]
            name = names[i]
            sim = sims[i]

            current_faces.append({
                'bbox': (x1, y1, x2, y2),
                'name': name,
                'sim': sim,
                'face_img': face_img.copy(),
            })

            color = (0, 255, 0) if name != "Неизвестно" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({sim:.2f})"
            if name == "Неизвестно":
                label += " — кликните по рамке, чтобы добавить"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            saved_path = saver.maybe_save((x1, y1, x2, y2), face_img, name, sim)
            if saved_path:
                cv2.putText(frame, "Saved", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Ключевые точки (если есть)
            if hasattr(face, 'kps'):
                kps = face.kps.astype(int)
                for pt in kps:
                    cv2.circle(frame, tuple(pt), 3, (255, 0, 0), -1)

        # Общее количество лиц
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Статусная строка (временная)
        if status_text and time.time() < status_until:
            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif time.time() >= status_until:
            status_text = ""

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
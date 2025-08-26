import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Скрипт запускает работу камер
import cv2
import pickle
from datetime import datetime, timedelta
from insightface.app import FaceAnalysis
from utils.logger import log_event, log_error
from shared import (
    SIMILARITY_THRESHOLD_KNOWN,
    SIMILARITY_THRESHOLD_UNKNOWN,
    SCREENSHOT_DELAY,
    KNOWN_DB_PATH,
    UNKNOWN_DB_PATH,
    SAVE_DIR,
    UNKNOWN_SAVE_DIR,
    cosine_similarity,
    get_mean_embedding,
    get_new_unknown_id,
    find_best_unknown_match,
    ensure_dir
)

def camera_loop(source, queue, cam_id, stop_flag):
    print(f"[INFO] Камера {cam_id} запущена с источником {source}")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 480))

    embeddings_db = {}
    if os.path.exists(KNOWN_DB_PATH):
        with open(KNOWN_DB_PATH, "rb") as f:
            embeddings_db = pickle.load(f)

    unknown_db = {}
    if os.path.exists(UNKNOWN_DB_PATH):
        with open(UNKNOWN_DB_PATH, "rb") as f:
            unknown_db = pickle.load(f)

    last_screenshot_time = {}
    last_saved = datetime.now()

    try:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if isinstance(source, str) else cv2.VideoCapture(source)
        if not cap.isOpened():
            log_error(f"Камера {cam_id} не открылась: {source}")
            return
    except Exception as e:
        log_error(f"Ошибка при открытии камеры {cam_id}: " + str(e))
        return

    skip_frames = 2

    while not stop_flag.value:
        for _ in range(skip_frames):
            cap.grab()

        ret, frame = cap.read()
        if not ret:
            log_error(f"[CAMERA {cam_id}] Не удалось прочитать кадр.")
            print(f"[WARNING] Камера {cam_id} не читает.")
            continue

        faces = app.get(frame)
        now = datetime.now()

        if (now - last_saved).total_seconds() > 30:
            with open(UNKNOWN_DB_PATH, "wb") as f:
                pickle.dump(unknown_db, f)
            last_saved = now

        for face in faces:
            emb = face.embedding
            label = "Неизвестный"
            person_key = None

            best_score = 0
            best_match = None
            for name, db_embs in embeddings_db.items():
                for emb_vector in db_embs:
                    score = cosine_similarity(emb, emb_vector)
                    if score > best_score:
                        best_score = score
                        best_match = name

            if best_score >= SIMILARITY_THRESHOLD_KNOWN:
                label = best_match
                person_key = best_match
                log_event(label)
                print(f"[KNOWN] Узнан: {label} (score: {best_score:.3f})")
            else:
                matched_unknown = find_best_unknown_match(emb, unknown_db, SIMILARITY_THRESHOLD_UNKNOWN)
                if matched_unknown:
                    label = matched_unknown
                    person_key = matched_unknown
                    unknown_db[matched_unknown].append(emb)
                    print(f"[UNKNOWN] Найден похожий: {matched_unknown}")
                else:
                    person_key = get_new_unknown_id(unknown_db)
                    unknown_db[person_key] = [emb]
                    label = person_key
                    print(f"[UNKNOWN] Новый ID: {person_key}")
                log_event(label)

            last_time = last_screenshot_time.get(person_key, datetime.min)
            if now - last_time > SCREENSHOT_DELAY:
                x1, y1, x2, y2 = map(int, face.bbox)
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                timestamp = now.strftime('%Y%m%d_%H%M%S')
                save_dir = os.path.join(
                    UNKNOWN_SAVE_DIR if person_key.startswith("unknown_") else SAVE_DIR,
                    now.strftime('%d_%m_%Y') if not person_key.startswith("unknown_") else "",
                    person_key
                )
                ensure_dir(save_dir)
                filepath = os.path.join(save_dir, f"{label}_{timestamp}.jpg")
                cv2.imwrite(filepath, face_img)
                last_screenshot_time[person_key] = now
                print(f"[CAMERA {cam_id}] Сохранено: {filepath}")

        queue.put((cam_id, frame))

    cap.release()
    with open(UNKNOWN_DB_PATH, "wb") as f:
        pickle.dump(unknown_db, f)

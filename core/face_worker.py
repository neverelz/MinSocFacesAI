import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import cv2
import pickle
from multiprocessing import Process, Value
from camera_worker import camera_loop
from utils.logger import log_event, log_error
from utils.telegram_alert import send_telegram_alert
from insightface.app import FaceAnalysis
from datetime import datetime
from shared import (
    SIMILARITY_THRESHOLD_KNOWN,
    SIMILARITY_THRESHOLD_UNKNOWN,
    SCREENSHOT_DELAY,
    KNOWN_DB_PATH,
    UNKNOWN_DB_PATH,
    SAVE_DIR,
    UNKNOWN_SAVE_DIR,
    CAMERA_SOURCES,
    cosine_similarity,
    get_mean_embedding,
    get_new_unknown_id,
    find_best_unknown_match,
    ensure_dir,
    load_config
)

unknown_tracker = {}
last_alert_sent = {}
ALERT_COOLDOWN = 600  # 10 минут 

def face_loop(frame_queue, result_queue, latest_frames, stop_flag):
    print("[FACE] Запущен анализатор")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 480))

    config = load_config()
    SIMILARITY_THRESHOLD_KNOWN = config.get("SIMILARITY_THRESHOLD_KNOWN", 0.65)
    SIMILARITY_THRESHOLD_UNKNOWN = config.get("SIMILARITY_THRESHOLD_UNKNOWN", 0.55)

    embeddings_db = {}
    if os.path.exists(KNOWN_DB_PATH):
        with open(KNOWN_DB_PATH, "rb") as f:
            embeddings_db = pickle.load(f)

    unknown_db = {}
    if os.path.exists(UNKNOWN_DB_PATH):
        with open(UNKNOWN_DB_PATH, "rb") as f:
            unknown_db = pickle.load(f)

    while not stop_flag.value:
        try:
            cam_id, frame = frame_queue.get(timeout=1)
        except:
            continue

        faces = app.get(frame)

        for face in faces:
            # ===== ФИЛЬТР ПО УГЛУ =====
            yaw, pitch = face.pose[0], face.pose[1]
            if abs(yaw) > 45 or abs(pitch) > 45:
                print(f"[DEBUG] Пропуск лица: угол yaw={yaw:.1f}, pitch={pitch:.1f}")
                continue

            emb = face.embedding
            label = "Неизвестный"
            person_key = None

            # Поиск среди известных
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
            else:
                matched_unknown = find_best_unknown_match(emb, unknown_db, SIMILARITY_THRESHOLD_UNKNOWN)
                if matched_unknown:
                    label = matched_unknown
                    person_key = matched_unknown
                    unknown_db[matched_unknown].append(emb)
                else:
                    person_key = get_new_unknown_id(unknown_db)
                    unknown_db[person_key] = [emb]
                    label = person_key

                log_event(label)

                # ===== ТРЕКИНГ UNKNOWN И ТРЕВОГА =====
                if person_key.startswith("unknown_"):
                    now = time.time()
                    last_time, count = unknown_tracker.get(person_key, (0, 0))
                    if now - last_time <= 30:
                        count += 1
                    else:
                        count = 1
                    unknown_tracker[person_key] = (now, count)

                    if count >= 35:
                        last_sent_time = last_alert_sent.get(person_key, 0)
                        if now - last_sent_time >= ALERT_COOLDOWN:
                            print(f"[DEBUG] Отправляю тревогу в Telegram: {person_key}")
                            if frame is not None:
                                send_telegram_alert(frame, person_key)
                                last_alert_sent[person_key] = now
                            else:
                                print("[DEBUG] frame is None")
                        else:
                            print(f"[DEBUG] Пропускаю тревогу — {person_key} недавно уже отправлен (ждём cooldown)")

            # Подпись и сохранение
            x1, y1, x2, y2 = map(int, face.bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_img = frame[y1:y2, x1:x2]
            if face_img.size > 0:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_dir = os.path.join(
                    UNKNOWN_SAVE_DIR if person_key.startswith("unknown_") else SAVE_DIR,
                    datetime.now().strftime('%d_%m_%Y') if not person_key.startswith("unknown_") else "",
                    person_key
                )
                ensure_dir(save_dir)
                filepath = os.path.join(save_dir, f"{label}_{timestamp}.jpg")
                cv2.imwrite(filepath, face_img)
                # === AI daily stats hooks ===
                try:
                    from core import shared  # absolute when running from project root
                except Exception:
                    import shared
                try:
                    if isinstance(label, str) and not label.startswith('unknown_'):
                        shared.register_known(label, filepath)
                    else:
                        shared.register_unknown(person_key, filepath)
                except Exception:
                    pass

            color = (0, 255, 0) if not person_key.startswith("unknown_") else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        latest_frames[cam_id] = frame.copy()

    with open(UNKNOWN_DB_PATH, "wb") as f:
        pickle.dump(unknown_db, f)

def launch_all(latest_frames, frame_queue, result_queue):
    processes = []
    stop_flags = []

    for cam_id, source in enumerate(CAMERA_SOURCES):
        flag = Value('b', False)
        p = Process(target=camera_loop, args=(source, frame_queue, cam_id, flag))
        p.start()
        processes.append(p)
        stop_flags.append(flag)

    face_flag = Value('b', False)
    face_proc = Process(target=face_loop, args=(frame_queue, result_queue, latest_frames, face_flag))
    face_proc.start()
    processes.append(face_proc)
    stop_flags.append(face_flag)

    return processes, stop_flags

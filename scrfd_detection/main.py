# main.py — исправленная версия с поддержкой Windows/Linux
import cv2
import time
import numpy as np
import threading
import os
import logging
from queue import Queue, Full, Empty
from detection import FaceDetector
from recognizer import FaceRecognizer
from camera import AsyncCameraReader
from hardware_detection import estimate_hardware_level, get_optimal_settings
from hands_detection import MultiHandDetector
from people_detection import HybridPeopleDetector
from fire import EarlyFireDetector
from platform_utils import get_font_candidates, normalize_path, safe_makedirs, get_platform_info
from PIL import Image, ImageDraw

# === ЛОГИРОВАНИЕ ===
LOG_DIR = safe_makedirs("logs", exist_ok=True)
for f in os.listdir(LOG_DIR):
    if f.endswith(".log"):
        os.remove(os.path.join(LOG_DIR, f))

system_logger = logging.getLogger("system")
system_handler = logging.FileHandler(os.path.join(LOG_DIR, "system.log"), encoding='utf-8')
system_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
system_handler.setFormatter(system_formatter)
system_logger.addHandler(system_handler)
system_logger.setLevel(logging.INFO)

_FONT_CACHE = {}

def get_font_cached(font_path, font_size):
    key = (font_path, font_size)
    if key not in _FONT_CACHE:
        try:
            from PIL import ImageFont
            if font_path is None:
                _FONT_CACHE[key] = ImageFont.load_default()
            else:
                _FONT_CACHE[key] = ImageFont.truetype(font_path, font_size)
        except:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]

def put_text_russian(img, text, org, font_path="arial.ttf", font_size=24, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_font_cached(font_path, font_size)
    draw.text(org, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def get_font_path():
    candidates = get_font_candidates()
    for path in candidates:
        try:
            from PIL import ImageFont
            ImageFont.truetype(path, 10)
            return path
        except:
            continue
    try:
        from PIL import ImageFont
        ImageFont.load_default()
        return None
    except:
        return "arial.ttf"

def align_face_by_kps(frame, kps, output_size=(112, 112)):
    ref = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    src = kps.astype(np.float32)
    transform = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)[0]
    if transform is None:
        return None
    return cv2.warpAffine(frame, transform, output_size, flags=cv2.INTER_LINEAR)

# =============== КЛАССЫ ТРЕКИНГА ===============
class GlobalFaceTrack:
    def __init__(self, person_id: str):
        self.person_id = person_id
        self.last_seen = time.time()
        self.last_saved_ts = 0.0
        self.last_centers = {}

    def update_center(self, camera_index: int, bbox_center):
        self.last_centers[camera_index] = bbox_center
        self.last_seen = time.time()

class GlobalFaceSaver:
    def __init__(self, recognizer: FaceRecognizer, save_interval_sec: float = 2.0):
        self.recognizer = recognizer
        self.save_interval = save_interval_sec
        self.tracks = []

    @staticmethod
    def _center_of_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _distance(c1, c2):
        if c1 is None or c2 is None:
            return float('inf')
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return (dx * dx + dy * dy) ** 0.5

    def _find_track_by_id(self, person_id: str) -> GlobalFaceTrack:
        for tr in self.tracks:
            if tr.person_id == person_id:
                return tr
        return None

    def _find_track_by_proximity(self, camera_index: int, bbox_center, max_dist=100) -> GlobalFaceTrack:
        for tr in self.tracks:
            if camera_index in tr.last_centers:
                if self._distance(tr.last_centers[camera_index], bbox_center) <= max_dist:
                    return tr
        return None

    def _get_or_create_track(self, camera_index: int, bbox_center, person_id_hint: str) -> GlobalFaceTrack:
        if person_id_hint != "Неизвестно":
            tr = self._find_track_by_id(person_id_hint)
            if tr:
                tr.update_center(camera_index, bbox_center)
                return tr
        tr = self._find_track_by_proximity(camera_index, bbox_center)
        if tr:
            if tr.person_id == "Неизвестно" and person_id_hint != "Неизвестно":
                tr.person_id = person_id_hint
                system_logger.info(f"🔗 Обновлён ID: теперь {person_id_hint}")
            tr.update_center(camera_index, bbox_center)
            return tr
        if person_id_hint == "Неизвестно":
            new_id = "Неизвестно"
        else:
            new_id = person_id_hint
        tr = GlobalFaceTrack(new_id)
        tr.update_center(camera_index, bbox_center)
        self.tracks.append(tr)
        system_logger.info(f"🆕 Создан трек для {new_id} с камеры {camera_index}")
        return tr

    def maybe_save(self, camera_index: int, bbox, face_img_bgr, person_id_hint: str, similarity: float):
        if person_id_hint == "Неизвестно":
            return None
        if similarity >= 0.9:
            return None
        now = time.time()
        center = self._center_of_bbox(bbox)
        tr = self._get_or_create_track(camera_index, center, person_id_hint)
        if now - tr.last_saved_ts < self.save_interval:
            return None
        saved_path = self.recognizer.add_image_to_person(tr.person_id, face_img_bgr)
        tr.last_saved_ts = now
        system_logger.info(f"💾 Сохранено изображение для {tr.person_id} (камера {camera_index})")
        return saved_path

# =============== ОБРАБОТЧИК КАМЕР ===============
class AsyncFaceProcessor:
    def __init__(self, camera_index: int, detector, recognizer, saver, settings):
        self.camera_index = camera_index
        self.detector = detector
        self.recognizer = recognizer
        self.saver = saver
        self.process_interval = settings['process_interval_sec']
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=1)
        self.show_keypoints = True
        self.show_hands = False
        self.show_people = False
        self.show_fire = False
        self._hand_detector = None
        self._people_detector = None
        self._fire_detector = None
        self.logger = logging.getLogger(f"camera_{camera_index}")
        handler = logging.FileHandler(os.path.join(LOG_DIR, f"camera_{camera_index}.log"), encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        blank = put_text_russian(blank, f"Источник {camera_index + 1} ИНИЦИАЛИЗАЦИЯ...", (50, 240),
                                 font_path=get_font_path(), font_size=24, color=(255, 255, 0))
        self.last_result = {'frame': blank, 'faces': [], 'camera_index': camera_index, 'original_size': (640, 480)}
        self.stop_event = threading.Event()
        self.thread = None
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.processed_count = 0

    def set_show_keypoints(self, show: bool):
        self.show_keypoints = show

    def set_show_hands(self, show: bool):
        self.show_hands = show

    def set_show_people(self, show: bool):
        self.show_people = show

    def set_show_fire(self, show: bool):
        self.show_fire = show

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        print(f"🧠 Поток обработки для камеры {self.camera_index} запущен")

    def run(self):
        self.logger.info("▶️ Запущен")
        last_process_time = 0
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue
            now = time.time()
            if now - last_process_time < self.process_interval:
                continue
            last_process_time = now
            self.processed_count += 1
            if self.processed_count % 5 == 0:
                elapsed = now - self.last_fps_time
                self.fps = 5 / elapsed if elapsed > 0 else 0
                self.last_fps_time = now
            try:
                faces = self.detector.detect(frame)
                face_patches, face_bboxes = [], []
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

                names, sims = self.recognizer.recognize_batch(face_patches) if face_patches else ([], [])

                current_faces = []
                for i, (x1, y1, x2, y2) in enumerate(face_bboxes):
                    face_img = face_patches[i]
                    name, sim = names[i], sims[i]
                    current_faces.append(
                        {'bbox': (x1, y1, x2, y2), 'name': name, 'sim': sim, 'face_img': face_img.copy()})
                    color = (0, 255, 0) if name != "Неизвестно" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{name} ({sim:.2f})" + (" — кликните" if name == "Неизвестно" else "")
                    frame = put_text_russian(frame, label, (x1, y1 - 30), font_path=get_font_path(), font_size=26,
                                             color=color)
                    saved_path = self.saver.maybe_save(self.camera_index, (x1, y1, x2, y2), face_img, name, sim)
                    if saved_path:
                        frame = put_text_russian(frame, "Сохранено", (x1, y2 + 20), font_path=get_font_path(),
                                                 font_size=22, color=(255, 255, 0))

                if self.show_keypoints:
                    for face in faces:
                        if hasattr(face, 'kps') and face.kps is not None:
                            kps = np.array(face.kps)
                            if kps.shape == (5, 2):
                                for pt in kps.astype(int):
                                    if 0 <= pt[0] < frame.shape[1] and 0 <= pt[1] < frame.shape[0]:
                                        cv2.circle(frame, (pt[0], pt[1]), 3, (255, 0, 0), -1)

                if self.show_people:
                    if self._people_detector is None:
                        self._people_detector = HybridPeopleDetector()
                    try:
                        boxes, scores = self._people_detector.detect(frame)
                        for (x1, y1, x2, y2), score in zip(boxes, scores):
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
                            cv2.putText(frame, f'{float(score):.2f}', (int(x1), int(y1) - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
                    except Exception as e:
                        self.logger.error(f"People overlay error: {e}")

                if self.show_hands:
                    if self._hand_detector is None:
                        self._hand_detector = MultiHandDetector(max_hands=6)
                    try:
                        frame, _, _ = self._hand_detector.detect_hands(frame)
                    except Exception as e:
                        self.logger.error(f"Hand overlay error: {e}")

                if self.show_fire:
                    if self._fire_detector is None:
                        self._fire_detector = EarlyFireDetector()
                    try:
                        f_boxes, f_scores, f_labels = self._fire_detector.detect(frame)
                        for (x1, y1, x2, y2), s, lab in zip(f_boxes, f_scores, f_labels):
                            color = (0, 140, 255) if lab == "FIRE" else (200, 200, 200)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, f"{lab} {float(s):.2f}", (int(x1), max(15, int(y1) - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception as e:
                        self.logger.error(f"Fire overlay error: {e}")

                frame = put_text_russian(frame, f"FPS: {self.fps:.1f}", (10, 30), font_path=get_font_path(),
                                         font_size=24, color=(255, 255, 255))
                frame = put_text_russian(frame, f"Источник {self.camera_index + 1}", (10, 70),
                                         font_path=get_font_path(), font_size=28, color=(255, 255, 255))
                self.last_result = {
                    'frame': frame,
                    'faces': current_faces,
                    'camera_index': self.camera_index,
                    'original_size': (frame.shape[1], frame.shape[0])
                }
                if not self.result_queue.full():
                    self.result_queue.put_nowait(self.last_result)
            except Exception as e:
                self.logger.error(f"❌ Ошибка: {e}")

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def submit_frame(self, frame):
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            self.frame_queue.put_nowait(frame)
        except (Full, Empty):
            pass

    def get_result(self, timeout=0.01):
        try:
            result = self.result_queue.get(timeout=timeout)
            self.last_result = result
            return result
        except Empty:
            return self.last_result

# =============== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===============
def find_available_cameras(max_tested=10):
    from platform_utils import get_optimal_camera_backends
    available = []
    backends = get_optimal_camera_backends()
    for i in range(max_tested):
        success = False
        for backend in backends:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                for _ in range(3):
                    ret, _ = cap.read()
                    if ret:
                        available.append(i)
                        success = True
                        break
                    time.sleep(0.1)
                cap.release()
                if success:
                    break
            cap.release()
        if success:
            continue
    return available

# =============== ОСНОВНАЯ ФУНКЦИЯ ===============
def main():
    platform_info = get_platform_info()
    if platform_info['is_linux']:
        display_var = os.environ.get('DISPLAY')
        if not display_var:
            print("⚠️ Переменная DISPLAY не установлена. Убедитесь что запущена X11 или Wayland.")
            print("💡 Для headless режима можно использовать SSH с X11 forwarding")
        print("🐧 Linux система обнаружена - убедитесь что установлены необходимые библиотеки.")

    estimated_level, details = estimate_hardware_level()
    if estimated_level == "nvidia_gpu":
        print("✅ Обнаружен NVIDIA GPU — будет использоваться GPU")
    else:
        print("⚠️ NVIDIA GPU не обнаружен — используется CPU")

    settings = get_optimal_settings(estimated_level)
    print(f"\n🚀 Настройки: разрешение {settings['camera_width']}x{settings['camera_height']}, "
          f"обработка раз в {settings['process_interval_sec']} сек")

    auto_clear = input("Очистить базу лиц? (y/N): ").strip().lower() in ("y", "yes", "д", "да")
    system_logger.info(f"Запуск: уровень={estimated_level}, очистка={auto_clear}")

    detector = FaceDetector(
        model_name='scrfd_10g_kps',
        use_gpu=settings['use_gpu'],
        det_size=settings['det_size']
    )
    if auto_clear:
        print("🧹 Очистка базы лиц и кэша...")
        from recognizer import DATABASE_DIR, EMBEDDINGS_FILE
        import shutil
        if os.path.exists(DATABASE_DIR):
            shutil.rmtree(DATABASE_DIR)
        if os.path.exists(EMBEDDINGS_FILE):
            os.remove(EMBEDDINGS_FILE)
        print("✅ База и кэш удалены")

    # Теперь создаём распознаватель БЕЗ force_rebuild — он сам создаст пустую базу
    recognizer = FaceRecognizer(force_rebuild=False, use_gpu=settings['use_gpu'])

    saver = GlobalFaceSaver(recognizer, save_interval_sec=2.0)

    print("🔍 Поиск камер...")
    camera_indices = find_available_cameras()
    print(f"🎥 Найдены камеры: {camera_indices}")
    if not camera_indices:
        print("❌ Камеры не найдены")
        return

    camera_readers, processors = [], []
    for idx in camera_indices:
        reader = AsyncCameraReader(idx, settings['camera_width'], settings['camera_height'], settings['camera_fps'])
        if reader.start():
            camera_readers.append(reader)
            time.sleep(1.0)  # увеличена пауза для стабильности
            processor = AsyncFaceProcessor(idx, detector, recognizer, saver, settings)
            processor.start()
            processors.append(processor)

    if not processors:
        print("❌ Не удалось запустить ни одну камеру")
        return

    platform_info = get_platform_info()
    print("✅ Система запущена. Горячие клавиши: q — выход, m — точки лица, h — руки, e — люди, f — пожар.")
    print(f"📍 Обнаружена платформа: {platform_info['system'].upper()}")

    WINDOW_NAME = "Система распознавания лиц"
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame = cv2.putText(test_frame, "Инициализация окна...", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, test_frame)
    cv2.waitKey(100)  # даём время на инициализацию
    cv2.resizeWindow(WINDOW_NAME, 800, 600)
    print("✅ GUI окно инициализировано")

    status_text, status_until = "", 0
    current_faces_per_cam = {}
    show_keypoints = True
    show_hands = False
    show_people = False
    show_fire = False

    def on_mouse(event, x, y, flags, userdata=None):
        nonlocal status_text, status_until
        if event == cv2.EVENT_LBUTTONDOWN:
            for cam_idx, data in current_faces_per_cam.items():
                for item in data['faces']:
                    bx1, by1, bx2, by2 = item['bbox']
                    if (bx1 + data['offset_x'] <= x <= bx2 + data['offset_x'] and
                            by1 + data['offset_y'] <= y <= by2 + data['offset_y']):
                        if item['name'] == "Неизвестно":
                            new_id = recognizer.get_next_person_id()
                            recognizer.add_image_to_person(new_id, item['face_img'])
                            status_text = f"✅ Добавлен ID {new_id}"
                            status_until = time.time() + 2.0
                            system_logger.info(f"Добавлен {new_id} с камеры {cam_idx}")
                        else:
                            status_text = f"ℹ️ Уже в базе: {item['name']}"
                            status_until = time.time() + 1.5
                        return

    mouse_callback_set = False

    try:
        while True:
            n = len(processors)
            if n == 0:
                break

            for p in processors:
                p.set_show_keypoints(show_keypoints)
                p.set_show_hands(show_hands)
                p.set_show_people(show_people)
                p.set_show_fire(show_fire)

            if n == 1:
                rows, cols = 1, 1
            elif n <= 2:
                rows, cols = 1, 2
            elif n <= 4:
                rows, cols = 2, 2
            elif n <= 6:
                rows, cols = 2, 3
            elif n <= 9:
                rows, cols = 3, 3
            else:
                import math
                cols = math.ceil(math.sqrt(n))
                rows = math.ceil(n / cols)

            results = [p.get_result() for p in processors]
            frames = [r['frame'] if r else np.zeros((480, 640, 3), dtype=np.uint8) for r in results]

            min_h = min(f.shape[0] for f in frames) if frames else 480
            max_widths = [int(f.shape[1] * min_h / f.shape[0]) for f in frames]
            max_w = max(max_widths) if max_widths else 640
            grid_h, grid_w = rows * min_h, cols * max_w
            combined = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            current_faces_per_cam.clear()
            for i, (processor, result) in enumerate(zip(processors, results)):
                row, col = i // cols, i % cols
                frame = result['frame'] if result else np.zeros((480, 640, 3), dtype=np.uint8)
                h_orig, w_orig = frame.shape[:2]
                scale = min_h / h_orig
                new_w = int(w_orig * scale)
                resized = cv2.resize(frame, (new_w, min_h))
                x_offset = col * max_w + (max_w - new_w) // 2
                y_offset = row * min_h
                combined[y_offset:y_offset + min_h, x_offset:x_offset + new_w] = resized

                scaled_faces = []
                if result:
                    for face in result['faces']:
                        x1, y1, x2, y2 = face['bbox']
                        x1_scaled = int(x1 * scale)
                        y1_scaled = int(y1 * scale)
                        x2_scaled = int(x2 * scale)
                        y2_scaled = int(y2 * scale)
                        scaled_faces.append({
                            'bbox': (x1_scaled, y1_scaled, x2_scaled, y2_scaled),
                            'name': face['name'],
                            'sim': face['sim'],
                            'face_img': face['face_img']
                        })
                current_faces_per_cam[processor.camera_index] = {
                    'faces': scaled_faces,
                    'offset_x': x_offset,
                    'offset_y': y_offset
                }

                reader = next((r for r in camera_readers if r.camera_index == processor.camera_index), None)
                if reader:
                    frame_raw = reader.get_frame(timeout=0.001)
                    if frame_raw is not None:
                        processor.submit_frame(frame_raw)

            unique_ids = set()
            total_faces = 0
            for data in current_faces_per_cam.values():
                for face in data['faces']:
                    unique_ids.add(face['name'])
                    total_faces += 1

            combined = put_text_russian(combined, f'Лица: {total_faces} | Люди: {len(unique_ids)}', (10, 40),
                                        font_path=get_font_path(), font_size=32, color=(0, 0, 255))

            kp_status = "ВКЛ" if show_keypoints else "ВЫКЛ"
            hands_status = "ВКЛ" if show_hands else "ВЫКЛ"
            people_status = "ВКЛ" if show_people else "ВЫКЛ"
            fire_status = "ВКЛ" if show_fire else "ВЫКЛ"
            combined = put_text_russian(combined, f'Точки: {kp_status} (M)  |  Руки: {hands_status} (H)  |  Люди: {people_status} (E)  |  Пожар: {fire_status} (F)',
                                        (10, combined.shape[0] - 30), font_path=get_font_path(), font_size=20, color=(255, 255, 0))

            if status_text and time.time() < status_until:
                combined = put_text_russian(combined, status_text, (10, 110),
                                            font_path=get_font_path(), font_size=28, color=(0, 255, 255))

            cv2.imshow(WINDOW_NAME, combined)
            cv2.resizeWindow(WINDOW_NAME, combined.shape[1], combined.shape[0])

            if not mouse_callback_set:
                try:
                    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
                    mouse_callback_set = True
                    print("✅ Mouse callback установлен")
                except cv2.error as e:
                    print(f"⚠️ Mouse callback не установлен: {e}")
                    print("💡 Click функции недоступны")

            key = cv2.waitKey(1) & 0xFF  # ← ЕДИНСТВЕННЫЙ waitKey
            if key == ord('q'):
                break
            elif key in (ord('m'), ord('M')):
                show_keypoints = not show_keypoints
                print(f"🔑 Ключевые точки: {'ВКЛЮЧЕНЫ' if show_keypoints else 'ВЫКЛЮЧЕНЫ'}")
            elif key in (ord('h'), ord('H')):
                show_hands = not show_hands
                print(f"🖐️ Руки: {'ВКЛЮЧЕНЫ' if show_hands else 'ВЫКЛЮЧЕНЫ'}")
            elif key in (ord('e'), ord('E')):
                show_people = not show_people
                print(f"🚶 Люди (experiments): {'ВКЛЮЧЕНО' if show_people else 'ВЫКЛЮЧЕНО'}")
            elif key in (ord('f'), ord('F')):
                show_fire = not show_fire
                print(f"🔥 Пожар/Дым: {'ВКЛЮЧЕНО' if show_fire else 'ВЫКЛЮЧЕНО'}")

    except KeyboardInterrupt:
        system_logger.info("🛑 Получен сигнал прерывания.")
    finally:
        for processor in processors:
            processor.stop()
        for reader in camera_readers:
            reader.stop()
        cv2.destroyAllWindows()
        system_logger.info("👋 Все потоки остановлены. Выход.")

if __name__ == '__main__':
    main()
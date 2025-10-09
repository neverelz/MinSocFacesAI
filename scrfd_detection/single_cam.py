# single_camera_analyzer.py
import cv2
import time
import numpy as np
import os
import logging
from detection import FaceDetector
from recognizer import FaceRecognizer
from camera import AsyncCameraReader
from hardware_detection import estimate_hardware_level, get_optimal_settings
from hands_detection import MultiHandDetector
from people_detection import HybridPeopleDetector
from fire import EarlyFireDetector
from platform_utils import get_font_candidates, normalize_path, safe_makedirs, get_platform_info
from PIL import Image, ImageDraw

LOG_DIR = safe_makedirs("logs", exist_ok=True)
system_logger = logging.getLogger("single_camera")
system_handler = logging.FileHandler(os.path.join(LOG_DIR, "single_camera.log"), encoding='utf-8')
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

class GlobalFaceTrack:
    def __init__(self, person_id: str):
        self.person_id = person_id
        self.last_seen = time.time()
        self.last_saved_ts = 0.0
        self.last_centers = {}

    def update_center(self, bbox_center):
        self.last_centers[0] = bbox_center
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

    def _find_track_by_proximity(self, bbox_center, max_dist=100) -> GlobalFaceTrack:
        for tr in self.tracks:
            if 0 in tr.last_centers:
                if self._distance(tr.last_centers[0], bbox_center) <= max_dist:
                    return tr
        return None

    def _get_or_create_track(self, bbox_center, person_id_hint: str) -> GlobalFaceTrack:
        if person_id_hint != "Неизвестно":
            tr = self._find_track_by_id(person_id_hint)
            if tr:
                tr.update_center(bbox_center)
                return tr
        tr = self._find_track_by_proximity(bbox_center)
        if tr:
            if tr.person_id == "Неизвестно" and person_id_hint != "Неизвестно":
                tr.person_id = person_id_hint
                system_logger.info(f"Обновлён ID: теперь {person_id_hint}")
            tr.update_center(bbox_center)
            return tr
        tr = GlobalFaceTrack(person_id_hint if person_id_hint != "Неизвестно" else "Неизвестно")
        tr.update_center(bbox_center)
        self.tracks.append(tr)
        system_logger.info(f"Создан трек для {tr.person_id}")
        return tr

    def maybe_save(self, bbox, face_img_bgr, person_id_hint: str, similarity: float):
        if person_id_hint == "Неизвестно" or similarity >= 0.9:
            return None
        now = time.time()
        center = self._center_of_bbox(bbox)
        tr = self._get_or_create_track(center, person_id_hint)
        if now - tr.last_saved_ts < self.save_interval:
            return None
        saved_path = self.recognizer.add_image_to_person(tr.person_id, face_img_bgr)
        tr.last_saved_ts = now
        system_logger.info(f"Сохранено изображение для {tr.person_id}")
        return saved_path

class SingleCameraAnalyzer:
    def __init__(self):
        # Анализ железа и настройки
        estimated_level, details = estimate_hardware_level()
        if estimated_level == "nvidia_gpu":
            print("Обнаружен NVIDIA GPU — будет использоваться GPU")
        else:
            print("NVIDIA GPU не обнаружен — используется CPU")
        self.settings = get_optimal_settings(estimated_level)
        print(f"Настройки: разрешение {self.settings['camera_width']}x{self.settings['camera_height']}, "
              f"обработка раз в {self.settings['process_interval_sec']} сек")

        # Запрос на очистку базы
        auto_clear = input("Очистить базу лиц? (y/N): ").strip().lower() in ("y", "yes", "д", "да")
        system_logger.info(f"Запуск: уровень={estimated_level}, очистка={auto_clear}")

        # Инициализация моделей
        self.detector = FaceDetector(
            model_name='scrfd_10g_kps',
            use_gpu=self.settings['use_gpu'],
            det_size=self.settings['det_size']
        )
        self.recognizer = FaceRecognizer(force_rebuild=auto_clear, use_gpu=self.settings['use_gpu'])
        if auto_clear:
            self.recognizer.clear_database_and_cache()
        self.saver = GlobalFaceSaver(self.recognizer)

        # Модули (ленивая инициализация)
        self._hand_detector = None
        self._people_detector = None
        self._fire_detector = None

        # Флаги
        self.show_keypoints = True
        self.show_hands = False
        self.show_people = False
        self.show_fire = False

        # Состояние GUI
        self.status_text = ""
        self.status_until = 0
        self.current_faces = []

        # FPS
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.processed_count = 0

    def toggle_keypoints(self):
        self.show_keypoints = not self.show_keypoints

    def toggle_hands(self):
        self.show_hands = not self.show_hands

    def toggle_people(self):
        self.show_people = not self.show_people

    def toggle_fire(self):
        self.show_fire = not self.show_fire

    def process_frame(self, frame):
        h_orig, w_orig = frame.shape[:2]
        now = time.time()
        self.processed_count += 1
        if self.processed_count % 5 == 0:
            elapsed = now - self.last_fps_time
            self.fps = 5 / elapsed if elapsed > 0 else 0
            self.last_fps_time = now

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

        # === ИСПРАВЛЕНИЕ: безопасный вызов recognize_batch ===
        if face_patches:
            try:
                names, sims = self.recognizer.recognize_batch(face_patches)
            except (AttributeError, ValueError) as e:
                # Если база пуста или индекс не инициализирован — все "Неизвестно"
                names = ["Неизвестно"] * len(face_patches)
                sims = [0.0] * len(face_patches)
                print(f"⚠️ Распознавание недоступно: {e}. Все лица помечены как 'Неизвестно'.")
        else:
            names, sims = [], []

        self.current_faces = []
        for i, (x1, y1, x2, y2) in enumerate(face_bboxes):
            face_img = face_patches[i]
            name, sim = names[i], sims[i]
            self.current_faces.append({'bbox': (x1, y1, x2, y2), 'name': name, 'sim': sim, 'face_img': face_img.copy()})
            color = (0, 255, 0) if name != "Неизвестно" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({sim:.2f})" + (" — кликните" if name == "Неизвестно" else "")
            frame = put_text_russian(frame, label, (x1, y1 - 30), font_path=get_font_path(), font_size=26, color=color)
            saved_path = self.saver.maybe_save((x1, y1, x2, y2), face_img, name, sim)
            if saved_path:
                frame = put_text_russian(frame, "Сохранено", (x1, y2 + 20), font_path=get_font_path(), font_size=22,
                                         color=(255, 255, 0))

        if self.show_keypoints:
            for face in faces:
                if hasattr(face, 'kps') and face.kps is not None:
                    kps = np.array(face.kps)
                    if kps.shape == (5, 2):
                        for pt in kps.astype(int):
                            if 0 <= pt[0] < w_orig and 0 <= pt[1] < h_orig:
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
                pass

        if self.show_hands:
            if self._hand_detector is None:
                self._hand_detector = MultiHandDetector(max_hands=6)
            try:
                frame, _, _ = self._hand_detector.detect_hands(frame)
            except Exception as e:
                pass

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
                pass

        total_faces = len(self.current_faces)
        unique_ids = set(face['name'] for face in self.current_faces)
        total_people = len(unique_ids)

        frame = put_text_russian(frame, f'Лица: {total_faces} | Люди: {total_people}', (10, 40),
                                 font_path=get_font_path(), font_size=32, color=(0, 0, 255))
        frame = put_text_russian(frame, f"FPS: {self.fps:.1f}", (10, 30), font_path=get_font_path(),
                                 font_size=24, color=(255, 255, 255))
        if self.status_text and time.time() < self.status_until:
            frame = put_text_russian(frame, self.status_text, (10, 70), font_path=get_font_path(),
                                     font_size=28, color=(0, 255, 255))

        kp_status = "ВКЛ" if self.show_keypoints else "ВЫКЛ"
        hands_status = "ВКЛ" if self.show_hands else "ВЫКЛ"
        people_status = "ВКЛ" if self.show_people else "ВЫКЛ"
        fire_status = "ВКЛ" if self.show_fire else "ВЫКЛ"
        frame = put_text_russian(frame,
            f'Точки: {kp_status} (M) | Руки: {hands_status} (H) | Люди: {people_status} (E) | Пожар: {fire_status} (F)',
            (10, frame.shape[0] - 30), font_path=get_font_path(), font_size=20, color=(255, 255, 0))

        return frame

    def on_mouse_click(self, event, x, y, flags, userdata=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            for item in self.current_faces:
                bx1, by1, bx2, by2 = item['bbox']
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    if item['name'] == "Неизвестно":
                        new_id = self.recognizer.get_next_person_id()
                        self.recognizer.add_image_to_person(new_id, item['face_img'])
                        self.status_text = f"Добавлен ID {new_id}"
                        self.status_until = time.time() + 2.0
                        system_logger.info(f"Добавлен {new_id}")
                    else:
                        self.status_text = f"Уже в базе: {item['name']}"
                        self.status_until = time.time() + 1.5
                    return

    def run(self):
        # === ГДЕ БЕРЁТСЯ ПОТОК С КАМЕРЫ ===
        reader = AsyncCameraReader(0, self.settings['camera_width'], self.settings['camera_height'], self.settings['camera_fps'])
        if not reader.start():
            print("Не удалось открыть камеру 0")
            return

        WINDOW_NAME = "Анализатор одной камеры"
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame = put_text_russian(test_frame, "Инициализация...", (200, 240),
                                      font_path=get_font_path(), font_size=28, color=(255, 255, 255))

        # === ГДЕ СОЗДАЁТСЯ ОКНО ВЫВОДА ===
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, test_frame)
        cv2.resizeWindow(WINDOW_NAME, 800, 600)
        for _ in range(5):
            cv2.waitKey(20)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse_click)

        try:
            while True:
                # === ГДЕ ПОТОК ПЕРЕДАЁТСЯ НА ОБРАБОТКУ ===
                frame = reader.get_frame(timeout=0.1)
                if frame is None:
                    continue

                processed_frame = self.process_frame(frame)
                cv2.imshow(WINDOW_NAME, processed_frame)
                cv2.resizeWindow(WINDOW_NAME, processed_frame.shape[1], processed_frame.shape[0])

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key in (ord('m'), ord('M')):
                    self.toggle_keypoints()
                elif key in (ord('h'), ord('H')):
                    self.toggle_hands()
                elif key in (ord('e'), ord('E')):
                    self.toggle_people()
                elif key in (ord('f'), ord('F')):
                    self.toggle_fire()

        except KeyboardInterrupt:
            pass
        finally:
            reader.stop()
            cv2.destroyAllWindows()
            system_logger.info("Анализ завершён")

if __name__ == '__main__':
    analyzer = SingleCameraAnalyzer()
    analyzer.run()
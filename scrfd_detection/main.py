# main.py — финальная версия с логами и FPS

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
from hardware_detection import estimate_hardware_level, get_optimal_settings, select_hardware_level_interactive

# Для русского текста
from PIL import Image, ImageDraw

# === ЛОГИРОВАНИЕ ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Удаляем старые логи
for f in os.listdir(LOG_DIR):
    if f.endswith(".log"):
        os.remove(os.path.join(LOG_DIR, f))

# Общий логгер
system_logger = logging.getLogger("system")
system_handler = logging.FileHandler(os.path.join(LOG_DIR, "system.log"), encoding='utf-8')
system_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
system_handler.setFormatter(system_formatter)
system_logger.addHandler(system_handler)
system_logger.setLevel(logging.INFO)

# Глобальный кэш шрифтов
_FONT_CACHE = {}


def get_font_cached(font_path, font_size):
    key = (font_path, font_size)
    if key not in _FONT_CACHE:
        try:
            from PIL import ImageFont
            _FONT_CACHE[key] = ImageFont.truetype(font_path, font_size)
        except IOError:
            system_logger.warning(f"Шрифт {font_path} не найден. Используется дефолтный.")
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def put_text_russian(img, text, org, font_path="arial.ttf", font_size=24, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_font_cached(font_path, font_size)
    draw.text(org, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_font_path():
    candidates = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans.ttf",
        "times.ttf",
        "C:/Windows/Fonts/times.ttf",
        "verdana.ttf",
        "C:/Windows/Fonts/verdana.ttf"
    ]
    for path in candidates:
        try:
            from PIL import ImageFont
            ImageFont.truetype(path, 10)
            return path
        except:
            continue
    return "arial.ttf"


def align_face_by_kps(frame, kps, output_size=(112, 112)):
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
    transform = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)[0]
    if transform is None:
        return None
    aligned = cv2.warpAffine(frame, transform, output_size, flags=cv2.INTER_LINEAR)
    return aligned


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


class AsyncFaceProcessor:
    def __init__(self, camera_index: int, detector, recognizer, saver, max_queue_size=3, process_interval=4.0):
        self.camera_index = camera_index
        self.detector = detector
        self.recognizer = recognizer
        self.saver = saver
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue(maxsize=1)
        self.process_interval = process_interval

        # Логгер для камеры
        self.logger = logging.getLogger(f"camera_{camera_index}")
        handler = logging.FileHandler(os.path.join(LOG_DIR, f"camera_{camera_index}.log"), encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        height = 480
        width = 640
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        blank = put_text_russian(blank, f"Источник {camera_index + 1} ИНИЦИАЛИЗАЦИЯ...", (50, height // 2),
                                 font_path=get_font_path(), font_size=24, color=(255, 255, 0))

        self.last_result = {
            'frame': blank,
            'faces': [],
            'camera_index': camera_index,
            'original_size': (width, height)
        }

        self.stop_event = threading.Event()
        self.thread = None
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.processed_count = 0

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        print(f"🧠 Поток обработки для камеры {self.camera_index} запущен")

    def run(self):
        self.logger.info(f"▶️ Запущен")
        last_process_time = 0
        frame_count = 0

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get_nowait()
                frame_count += 1
            except Empty:
                time.sleep(0.01)
                continue

            now = time.time()
            if now - last_process_time < self.process_interval:
                continue

            last_process_time = now
            self.processed_count += 1

            if self.processed_count % 5 == 0:
                elapsed = now - self.last_fps_time
                self.fps = 5 / elapsed if elapsed > 0 else 0
                self.logger.info(f"📈 Средний FPS обработки: {self.fps:.2f} (всего обработано: {self.processed_count})")
                self.last_fps_time = now

            self.logger.info(f"🧠 Обрабатываем кадр #{frame_count} (интервал: {self.process_interval} сек)")

            try:
                faces = self.detector.detect(frame)
                face_patches = []
                face_bboxes = []

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

                names = []
                sims = []
                if face_patches:
                    names, sims = self.recognizer.recognize_batch(face_patches)

                current_faces = []
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
                        label += " — кликните"

                    frame = put_text_russian(frame, label, (x1, y1 - 30),
                                             font_path=get_font_path(), font_size=26, color=color)

                    saved_path = self.saver.maybe_save(self.camera_index, (x1, y1, x2, y2), face_img, name, sim)
                    if saved_path:
                        frame = put_text_russian(frame, "Сохранено", (x1, y2 + 20),
                                                 font_path=get_font_path(), font_size=22, color=(255, 255, 0))

                    if hasattr(face, 'kps'):
                        kps = face.kps.astype(int)
                        for pt in kps:
                            cv2.circle(frame, tuple(pt), 3, (255, 0, 0), -1)

                # Отображаем FPS
                frame = put_text_russian(frame, f"FPS: {self.fps:.1f}", (10, 30),
                                         font_path=get_font_path(), font_size=24, color=(255, 255, 255))

                frame = put_text_russian(frame, f"Источник {self.camera_index + 1}", (10, 70),
                                         font_path=get_font_path(), font_size=28, color=(255, 255, 255))

                result = {
                    'frame': frame,
                    'faces': current_faces,
                    'camera_index': self.camera_index,
                    'original_size': (frame.shape[1], frame.shape[0])
                }
                self.last_result = result

                try:
                    if self.result_queue.full():
                        self.result_queue.get_nowait()
                    self.result_queue.put_nowait(result)
                except Full:
                    pass

                self.logger.info(f"✅ Кадр #{frame_count} обработан")

            except Exception as e:
                self.logger.error(f"❌ Ошибка обработки: {e}")
                continue

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


def find_available_cameras(max_tested=20):
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(3):
            ret, _ = cap.read()
            if ret:
                available.append(i)
                break
            time.sleep(0.1)
        cap.release()

    return available


def main():
    estimated_level, score, details = estimate_hardware_level()
    print(f"\n💻 Обнаружено железо: {estimated_level.upper} (score: {score:.1f})")
    print("📊 Детали:", details)

    selected_level = select_hardware_level_interactive(estimated_level)
    settings = get_optimal_settings(selected_level)

    print("\n🚀 Используемые настройки:")
    print(f"   💻 Уровень: {selected_level.upper()}")
    print(f"   ⚙️  Обработка: 1 кадр / {settings['process_interval_sec']} сек")
    print(f"   📷 Разрешение: {settings['camera_width']}x{settings['camera_height']}")
    print(f"   🎞️  FPS камер: {settings['camera_fps']}")
    print(f"   🧩 Выравнивание лиц: Всегда включено ✅")
    print("   📸 Камеры: Все доступные видеоисточники\n")

    auto_clear = False
    try:
        ans = input("Очистить базу лиц и кэш эмбеддингов перед запуском? (y/N): ").strip().lower()
        auto_clear = ans in ("y", "yes", "д", "да")
    except Exception:
        pass

    system_logger.info("=== ЗАПУСК СИСТЕМЫ ===")
    system_logger.info(f"Уровень железа: {selected_level}, Очистка БД: {auto_clear}")

    detector = FaceDetector(model_name='scrfd_10g_kps', device_id=0)
    recognizer = FaceRecognizer(force_rebuild=auto_clear)
    if auto_clear:
        recognizer.clear_database_and_cache()
    saver = GlobalFaceSaver(recognizer, save_interval_sec=2.0)

    print("🔍 Поиск доступных камер...")
    camera_indices = find_available_cameras(max_tested=20)
    print(f"🎥 Найдены камеры: {camera_indices}")

    if not camera_indices:
        print("❌ Не найдено ни одного видеоисточника")
        return

    camera_readers = []
    processors = []

    for idx in camera_indices:
        reader = AsyncCameraReader(
            idx,
            width=settings['camera_width'],
            height=settings['camera_height'],
            fps=settings['camera_fps']
        )
        if reader.start():
            camera_readers.append(reader)
            warmup_time = 0.5
            print(f"⏳ [Камера {idx}] Прогрев на {warmup_time} сек...")
            time.sleep(warmup_time)

            processor = AsyncFaceProcessor(
                idx,
                detector,
                recognizer,
                saver,
                process_interval=settings['process_interval_sec']
            )
            processor.start()
            processors.append(processor)
        else:
            print(f"⚠️  Камера {idx} не запущена")

    if not processors:
        print("❌ Ни один процессор не запущен")
        return

    print("✅ Система запущена. Нажмите 'q' для выхода. Кликните по рамке 'Неизвестно' чтобы добавить в базу.")

    WINDOW_NAME = "Система распознавания лиц"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    status_text = ""
    status_until = 0
    current_faces_per_cam = {}

    def on_mouse(event, x, y, flags, userdata=None):
        nonlocal status_text, status_until
        if event == cv2.EVENT_LBUTTONDOWN:
            for cam_idx, data in current_faces_per_cam.items():
                faces = data['faces']
                offset_x = data['offset_x']
                offset_y = data['offset_y']
                for item in faces:
                    (bx1, by1, bx2, by2) = item['bbox']
                    bx1_global = bx1 + offset_x
                    by1_global = by1 + offset_y
                    bx2_global = bx2 + offset_x
                    by2_global = by2 + offset_y

                    if bx1_global <= x <= bx2_global and by1_global <= y <= by2_global:
                        if item['name'] == "Неизвестно":
                            new_id = recognizer.get_next_person_id()
                            recognizer.add_image_to_person(new_id, item['face_img'])
                            status_text = f"✅ Добавлен ID {new_id} с камеры {cam_idx}"
                            status_until = time.time() + 2.0
                            system_logger.info(f"🎉 УСПЕХ: Добавлен новый человек с ID {new_id} с камеры {cam_idx}")
                        else:
                            status_text = f"ℹ️ Уже в базе: {item['name']}"
                            status_until = time.time() + 1.5
                            system_logger.info(f"📌 Лицо уже известно: {item['name']}")
                        return

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            n = len(processors)
            if n == 0:
                break

            # Определяем сетку один раз
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

            # Получаем кадры
            frames = []
            results = [p.get_result() for p in processors]
            for result in results:
                frame = result['frame'] if result else np.zeros((480, 640, 3), dtype=np.uint8)
                frames.append(frame)

            min_h = min(f.shape[0] for f in frames) if frames else 480
            max_widths = [int(f.shape[1] * min_h / f.shape[0]) for f in frames]
            max_w = max(max_widths) if max_widths else 640

            # Собираем сетку
            grid_h = rows * min_h
            grid_w = cols * max_w
            combined = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            current_faces_per_cam.clear()
            for i, (processor, result) in enumerate(zip(processors, results)):
                row = i // cols
                col = i % cols
                frame = result['frame'] if result else np.zeros((480, 640, 3), dtype=np.uint8)
                h_orig, w_orig = frame.shape[:2]
                scale = min_h / h_orig
                new_w = int(w_orig * scale)
                resized = cv2.resize(frame, (new_w, min_h))

                x_offset = col * max_w + (max_w - new_w) // 2
                y_offset = row * min_h
                combined[y_offset:y_offset + min_h, x_offset:x_offset + new_w] = resized

                # Подготавливаем данные для клика
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

                # Отправляем свежий кадр в обработчик
                reader = next((r for r in camera_readers if r.camera_index == processor.camera_index), None)
                if reader:
                    frame_raw = reader.get_frame(timeout=0.001)
                    if frame_raw is not None:
                        processor.submit_frame(frame_raw)

            # Статистика
            unique_ids = set()
            total_faces = 0
            for data in current_faces_per_cam.values():
                for face in data['faces']:
                    unique_ids.add(face['name'])
                    total_faces += 1

            combined = put_text_russian(combined, f'Лица: {total_faces} | Люди: {len(unique_ids)}', (10, 40),
                                        font_path=get_font_path(), font_size=32, color=(0, 0, 255))

            if status_text and time.time() < status_until:
                combined = put_text_russian(combined, status_text, (10, 110),
                                            font_path=get_font_path(), font_size=28, color=(0, 255, 255))

            cv2.imshow(WINDOW_NAME, combined)
            cv2.resizeWindow(WINDOW_NAME, combined.shape[1], combined.shape[0])

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

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
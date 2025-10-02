# main.py — обновленный код с трекингом

import cv2
import time
import numpy as np
import threading
from queue import Queue
from camera import AsyncCameraReader
from hardware_detection import estimate_hardware_level, get_optimal_settings, select_hardware_level_interactive
import json
import os
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List
plt.ion()

from spatial_mapper import SpatialMapper


def create_camera_grid(frames, min_h):
    n = len(frames)
    if n == 0:
        return np.zeros((min_h, 640, 3), dtype=np.uint8)

    if n == 1:
        rows, cols = 1, 1
    elif n <= 4:
        rows, cols = 2, 2
    else:
        import math
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

    resized_frames = []
    max_widths = []
    for frame in frames:
        scale = min_h / frame.shape[0]
        new_w = int(frame.shape[1] * scale)
        resized = cv2.resize(frame, (new_w, min_h))
        resized_frames.append(resized)
        max_widths.append(new_w)

    grid_h = rows * min_h
    grid_w = cols * max(max_widths) if max_widths else 640
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, frame in enumerate(resized_frames):
        row = i // cols
        col = i % cols
        x_offset = col * max(max_widths) if max_widths else 0
        y_offset = row * min_h
        h, w = frame.shape[:2]
        x_centered = x_offset + (max(max_widths) - w) // 2 if max_widths else x_offset
        grid[y_offset:y_offset+h, x_centered:x_centered+w] = frame

    return grid


def find_available_cameras(max_tested=5):
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_ANY)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def main():
    estimated_level, _, _ = estimate_hardware_level()
    selected_level = select_hardware_level_interactive(estimated_level)
    settings = get_optimal_settings(selected_level)

    print("\n🚀 Настройки:")
    print(f"   Уровень: {selected_level.upper()}")
    print(f"   Разрешение: {settings['camera_width']}x{settings['camera_height']}")
    print(f"   FPS: {settings['camera_fps']}")

    camera_indices = find_available_cameras()
    if not camera_indices:
        print("❌ Не найдено ни одной камеры")
        return
    print(f"🎥 Найдены камеры: {camera_indices}")

    mapper = SpatialMapper()

    for idx in camera_indices:
        cam_cfg = mapper.get_camera_config(idx)
        if not cam_cfg:
            cam_cfg = mapper.add_camera_config(idx)

        print(f"\n📌 Обработка камеры '{cam_cfg['name']}' ({idx})")

        reader_temp = AsyncCameraReader(
            idx,
            width=settings['camera_width'],
            height=settings['camera_height'],
            fps=settings['camera_fps']
        )
        if not reader_temp.start():
            print(f"❌ Не удалось запустить камеру {idx}")
            continue

        frame = None
        for _ in range(10):
            frame = reader_temp.get_frame(timeout=1.0)
            if frame is not None:
                break
            time.sleep(0.5)
        reader_temp.stop()
        time.sleep(0.5)

        if frame is None:
            print(f"❌ Не удалось получить кадр от камеры {idx}")
            continue

        do_calibrate_points = False
        if mapper.is_camera_calibrated(idx):
            ans = input(f"🔧 Перекалибровать точки на изображении для камеры {idx}? (y/N): ").strip().lower()
            do_calibrate_points = ans in ("y", "yes", "д", "да")
        else:
            ans = input(f"🔧 Калибровать точки на изображении для камеры {idx}? (y/N): ").strip().lower()
            do_calibrate_points = ans in ("y", "yes", "д", "да")

        if do_calibrate_points:
            if not mapper.calibrate_camera_points(idx, frame):
                print(f"❌ Калибровка точек отменена для камеры {idx}")
                continue
        else:
            print(f"✅ Используем существующие данные калибровки точек для камеры {idx}")

        do_place = False
        if mapper.is_camera_positioned(idx):
            ans = input(f"🔧 Перенастроить позицию и точки на карте для камеры {idx}? (y/N): ").strip().lower()
            do_place = ans in ("y", "yes", "д", "да")
        else:
            ans = input(f"🔧 Настроить позицию и точки на карте для камеры {idx}? (y/N): ").strip().lower()
            do_place = ans in ("y", "yes", "д", "да")

        if do_place:
            if not mapper.place_camera_and_points(idx, frame):
                print(f"❌ Настройка позиции отменена для камеры {idx}")
                continue
        else:
            print(f"✅ Используем существующие данные позиции для камеры {idx}")

        print(f"✅ Камера {idx} обработана")

    camera_readers = []
    global_positions = {}
    last_map_update = 0
    MAP_UPDATE_INTERVAL = 1.0

    WINDOW_NAME = "Тест: пространственное отслеживание"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    map_fig, map_ax = plt.subplots(1, 1, figsize=(12, 12))
    xlim, ylim = mapper.calculate_limits()
    map_ax.set_xlim(xlim)
    map_ax.set_ylim(ylim)
    plt.show(block=False)

    try:
        while True:
            frames = []
            current_time = time.time()
            for idx in camera_indices:
                reader = next((r for r in camera_readers if r.camera_index == idx), None)
                if not reader:
                    reader = AsyncCameraReader(
                        idx,
                        width=settings['camera_width'],
                        height=settings['camera_height'],
                        fps=settings['camera_fps']
                    )
                    if reader.start():
                        camera_readers.append(reader)
                        print(f"▶️  [Камера {idx}] Поток захвата запущен")
                    else:
                        continue

                raw_frame = reader.get_frame(timeout=0.3)
                if raw_frame is None:
                    continue

                display_frame = raw_frame.copy()

                boxes = mapper.get_people_boxes(raw_frame)

                for box in boxes:
                    cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                foot_points = [mapper.get_bottom_center(box) for box in boxes]
                positions = mapper.project_foot_points(idx, foot_points, current_time)
                if positions:
                    global_positions[idx] = positions

                cv2.putText(display_frame, f"Камера {idx}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                frames.append(display_frame)

            if frames:
                min_h = min(f.shape[0] for f in frames) if frames else 480
                combined = create_camera_grid(frames, min_h)
                cv2.imshow(WINDOW_NAME, combined)

            now = time.time()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('m') or (now - last_map_update >= MAP_UPDATE_INTERVAL):
                print("🌍 Обновляем карту...")
                try:
                    mapper.update_global_map(map_fig, map_ax, global_positions)
                except Exception as e:
                    print(f"❌ Ошибка отображения карты: {e}")
                last_map_update = now

    except KeyboardInterrupt:
        pass
    finally:
        for r in camera_readers:
            r.stop()
        cv2.destroyAllWindows()
        plt.close(map_fig)
        print("👋 Тест завершён.")


if __name__ == '__main__':
    main()
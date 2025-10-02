# spatial_mapper.py — обновленный код с исправлением ошибки и улучшенным трекингом

import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from scipy.spatial import distance
from ultralytics import YOLO


class SpatialMapper:
    def __init__(self, config_file="cameras.json"):
        self.config_file = config_file
        self.config = self.load_config()
        if not self.config:
            raise FileNotFoundError(f"Файл конфигурации {config_file} не найден.")
        self.model = YOLO(self.config["yolo"]["model"])
        # Хранилище для истории позиций людей: {track_id: [(x, y, timestamp)]}
        self.track_history = defaultdict(list)
        self.max_history = 5  # Храним последние 5 позиций
        self.track_id_counter = 0

    def load_config(self):
        if not os.path.exists(self.config_file):
            print(f"[SpatialMapper] ❌ Файл {self.config_file} не найден.")
            return None
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[SpatialMapper] Ошибка чтения: {e}")
            return None

    def save_config(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print("[SpatialMapper] ✅ Конфиг сохранён.")
        except Exception as e:
            print(f"[SpatialMapper] ❌ Ошибка сохранения: {e}")

    def is_camera_calibrated(self, camera_index: int) -> bool:
        for cam in self.config["cameras"]:
            if cam["source"] == camera_index:
                return cam.get("calibrated", False) and cam.get("calibration") is not None
        return False

    def is_camera_positioned(self, camera_index: int) -> bool:
        for cam in self.config["cameras"]:
            if cam["source"] == camera_index:
                return ("global_position" in cam and cam["global_position"] is not None and
                        "rotation_deg" in cam and cam["rotation_deg"] is not None)
        return False

    def get_camera_config(self, camera_index: int) -> dict:
        for cam in self.config["cameras"]:
            if cam["source"] == camera_index:
                return cam
        return None

    def add_camera_config(self, camera_index: int) -> dict:
        new_id = len(self.config["cameras"])
        new_cam = {
            "id": new_id,
            "source": camera_index,
            "name": f"Камера {camera_index}",
            "enabled": True,
            "calibrated": False,
            "settings": {
                "width": 1920,
                "height": 1080,
                "fps": 15,
                "tilt_angle": 15.0,
                "height_m": 2.5
            },
            "calibration": None,
            "global_position": None,
            "rotation_deg": None
        }
        self.config["cameras"].append(new_cam)
        self.save_config()
        print(f"[SpatialMapper] ✅ Добавлена новая камера {camera_index} в конфиг.")
        return new_cam

    def interactive_calibration_matplotlib(self, frame: np.ndarray, window_title: str = "Калибровка") -> Optional[Dict]:
        selected_points_img = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.imshow(frame_rgb)
        ax.set_title(window_title + ". Кликните от 3 до 10 точек. Нажмите 'q' для выхода.")
        ax.axis('off')

        done = [False]

        def onclick(event):
            if done[0] or event.xdata is None or event.ydata is None or len(selected_points_img) >= 10:
                return
            x, y = int(event.xdata), int(event.ydata)
            selected_points_img.append((x, y))
            pt_num = len(selected_points_img)
            print(f"✅ Точка {pt_num}: ({x}, {y})")
            ax.plot(x, y, 'go', markersize=10)
            ax.text(x, y - 10, str(pt_num),
                    color='white', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='green', alpha=0.7))
            fig.canvas.draw()

        def onkey(event):
            if event.key == 'q':
                if len(selected_points_img) >= 3:
                    done[0] = True
                else:
                    print("❌ Нужно минимум 3 точки. Продолжайте.")

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        plt.show(block=False)
        while not done[0]:
            plt.pause(0.1)
        plt.close(fig)

        if len(selected_points_img) < 3:
            print("❌ Калибровка отменена: выбрано менее 3 точек.")
            return None

        return {"image_points": selected_points_img}

    def interactive_map_points(self, num_points: int, cam_name: str) -> Optional[Tuple[List[float], float, List[List[float]]]]:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        self.draw_background(ax)
        xlim, ylim = self.calculate_limits()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f"Настройка '{cam_name}': 1-й клик - позиция камеры, 2-й - точка направления, затем {num_points} глобальных точек. Нажмите 'q' для отмены.")
        ax.set_xlabel("X (м)")
        ax.set_ylabel("Y (м)")

        points = []
        done = [False]

        def onclick(event):
            if done[0] or event.xdata is None or event.ydata is None:
                return
            points.append((event.xdata, event.ydata))
            pt_num = len(points)
            if pt_num == 1:
                ax.plot(event.xdata, event.ydata, 'bo', ms=10)
                ax.text(event.xdata, event.ydata + 0.2, 'Cam', color='blue', fontsize=12, ha='center')
            elif pt_num == 2:
                dx = points[1][0] - points[0][0]
                dy = points[1][1] - points[0][1]
                ax.arrow(points[0][0], points[0][1], dx, dy, head_width=0.5, width=0.2, color='blue')
            else:
                ax.plot(event.xdata, event.ydata, 'ro', ms=8)
                ax.text(event.xdata, event.ydata + 0.2, str(pt_num - 2), color='red', fontsize=10)
            fig.canvas.draw()
            if pt_num >= num_points + 2:
                done[0] = True

        def onkey(event):
            if event.key == 'q':
                done[0] = True

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        plt.show(block=False)
        while not done[0]:
            plt.pause(0.1)
        plt.close(fig)

        if len(points) < num_points + 2:
            print("❌ Настройка отменена: недостаточно точек.")
            return None

        cam_pos = [float(points[0][0]), float(points[0][1])]
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        angle = float(np.degrees(np.arctan2(dy, dx)))
        global_terrain_points = [[float(p[0]), float(p[1])] for p in points[2:]]

        return cam_pos, angle, global_terrain_points

    def compute_relative_points(self, global_points: List[List[float]], cam_pos: List[float], rot_deg: float) -> List[List[float]]:
        theta = np.radians(rot_deg)
        cos, sin = np.cos(theta), np.sin(theta)
        inv_rot_mat = np.array([[cos, -sin], [sin, cos]])
        rel_points = []
        for g in global_points:
            translated = np.array(g) - np.array(cam_pos)
            rel = inv_rot_mat @ translated
            rel_points.append(rel.tolist())
        return rel_points

    def calibrate_camera_points(self, camera_index: int, frame: np.ndarray) -> bool:
        cam_cfg = self.get_camera_config(camera_index)
        if not cam_cfg:
            print(f"[SpatialMapper] ❌ Камера {camera_index} не найдена.")
            return False

        calib_result = self.interactive_calibration_matplotlib(frame, f"Калибровка точек: {cam_cfg['name']}")
        if calib_result is None:
            return False

        cam_cfg["calibration"] = calib_result
        self.save_config()
        print(f"✅ Точки изображения сохранены для {cam_cfg['name']}")
        return True

    def place_camera_and_points(self, camera_index: int, frame: np.ndarray) -> bool:
        cam_cfg = self.get_camera_config(camera_index)
        if not cam_cfg or "calibration" not in cam_cfg or "image_points" not in cam_cfg["calibration"]:
            print(f"[SpatialMapper] ⚠️ Сначала выполните калибровку точек на изображении для камеры {camera_index}.")
            return False

        display_frame = frame.copy()
        for i, (x, y) in enumerate(cam_cfg["calibration"]["image_points"]):
            cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(display_frame, str(i+1), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.namedWindow("Вид камеры с точками", cv2.WINDOW_NORMAL)
        cv2.imshow("Вид камеры с точками", display_frame)

        num_points = len(cam_cfg["calibration"]["image_points"])
        placement_result = self.interactive_map_points(num_points, cam_cfg['name'])
        cv2.destroyWindow("Вид камеры с точками")

        if placement_result is None:
            return False

        cam_pos, angle, global_terrain = placement_result
        rel_terrain = self.compute_relative_points(global_terrain, cam_pos, angle)

        cam_cfg["calibration"]["terrain_points"] = rel_terrain
        cam_cfg["global_position"] = cam_pos
        cam_cfg["rotation_deg"] = angle
        cam_cfg["calibrated"] = True
        self.save_config()
        print(f"✅ Позиция и точки сохранены для {cam_cfg['name']}")
        return True

    def get_relative_position(self, img_x: int, img_y: int, calib: Dict) -> np.ndarray:
        img_pts = np.float32(calib["image_points"])
        terr_pts_rel = np.float32(calib["terrain_points"])
        H, _ = cv2.findHomography(img_pts, terr_pts_rel)
        pt = np.array([[[img_x, img_y]]], dtype=np.float32)
        rel_pt = cv2.perspectiveTransform(pt, H)[0][0]
        return rel_pt

    def get_global_position(self, rel_pt: np.ndarray, cam_pos: List[float], cam_rot_deg: float) -> Tuple[float, float]:
        theta = np.radians(cam_rot_deg)
        cos, sin = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[cos, sin], [-sin, cos]])
        rotated = np.dot(rot_mat, rel_pt)
        global_pt = rotated + np.array(cam_pos)
        return float(global_pt[0]), float(global_pt[1])

    def transform_points(self, points_rel: List[List[float]], pos: List[float], rot_deg: float) -> List[List[float]]:
        theta = np.radians(rot_deg)
        cos, sin = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[cos, sin], [-sin, cos]])
        transformed = []
        for p in points_rel:
            rotated = rot_mat @ np.array(p)
            global_p = rotated + np.array(pos)
            transformed.append(global_p.tolist())
        return transformed

    def get_people_boxes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        results = self.model(frame, verbose=False)
        boxes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == 0 and conf >= self.config["yolo"]["confidence_threshold"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2))
        return boxes

    def get_bottom_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        height = y2 - y1
        if height < 200:  # Порог для "маленького" бокса (сидящий человек)
            cy = y2 + 100  # Смещение вниз для учета ног
        else:
            cy = y2 - (height // 10)  # Стандартный расчет для стоящих
        return cx, cy

    def project_foot_points(self, camera_index: int, foot_points: List[Tuple[int, int]], timestamp: float) -> List[Tuple[int, Tuple[float, float]]]:
        cam_cfg = self.get_camera_config(camera_index)
        if not cam_cfg or not cam_cfg.get("calibrated") or "global_position" not in cam_cfg or cam_cfg["global_position"] is None or "rotation_deg" not in cam_cfg or cam_cfg["rotation_deg"] is None:
            print(f"[SpatialMapper] ⚠️ Камера {camera_index} не полностью настроена. Пропуск проекции.")
            return []
        calib = cam_cfg["calibration"]
        pos = cam_cfg["global_position"]
        rot = cam_cfg["rotation_deg"]
        projected = []
        for fp in foot_points:
            rel = self.get_relative_position(fp[0], fp[1], calib)
            glob = self.get_global_position(rel, pos, rot)
            projected.append(glob)
        # Простой трекинг: связываем новые позиции с предыдущими по евклидову расстоянию
        tracked_positions = []
        for pos in projected:
            min_dist = float('inf')
            closest_id = None
            for track_id, history in self.track_history.items():
                if history:
                    last_pos = history[-1][:2]
                    dist = distance.euclidean(pos, last_pos)
                    if dist < min_dist and dist < 2.0:  # Увеличенный порог для стабильности
                        min_dist = dist
                        closest_id = track_id
            if closest_id is None:
                closest_id = self.track_id_counter
                self.track_id_counter += 1
            self.track_history[closest_id].append((pos[0], pos[1], timestamp))
            if len(self.track_history[closest_id]) > self.max_history:
                self.track_history[closest_id].pop(0)
            tracked_positions.append((closest_id, pos))
        # Очистка старых треков (старше 3 секунд для меньше шума)
        current_time = timestamp
        for track_id in list(self.track_history.keys()):
            if self.track_history[track_id] and current_time - self.track_history[track_id][-1][2] > 3.0:
                del self.track_history[track_id]
        return tracked_positions

    def draw_background(self, ax):
        floor_plans = self.config.get("floor_plans", [])
        if floor_plans:
            for plan in floor_plans:
                image_path = plan.get("image")
                if image_path and os.path.exists(image_path):
                    img = plt.imread(image_path)
                    ext = plan["extent"]
                    alpha = plan.get("alpha", 0.5)
                    ax.imshow(img, extent=ext, alpha=alpha, zorder=0)
        else:
            room_map = self.config.get("room_map", {})
            if room_map:
                origin_x, origin_y = room_map["origin"]
                room_w = room_map["width"]
                room_h = room_map["height"]
                rect = plt.Rectangle((origin_x, origin_y), room_w, room_h,
                                     linewidth=2, edgecolor='black', facecolor='none', zorder=0)
                ax.add_patch(rect)
        ax.grid(True, linestyle='--', alpha=0.5)

    def calculate_limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        updated = False

        for cam in self.config["cameras"]:
            if (cam.get("enabled") and cam.get("calibrated") and
                "global_position" in cam and cam["global_position"] is not None and
                "rotation_deg" in cam and cam["rotation_deg"] is not None and
                "calibration" in cam and "terrain_points" in cam["calibration"]):
                terr_rel = cam["calibration"]["terrain_points"]
                terr_global = self.transform_points(terr_rel, cam["global_position"], cam["rotation_deg"])
                xs = [p[0] for p in terr_global]
                ys = [p[1] for p in terr_global]
                min_x = min(min_x, min(xs))
                max_x = max(max_x, max(xs))
                min_y = min(min_y, min(ys))
                max_y = max(max_y, max(ys))
                updated = True

        floor_plans = self.config.get("floor_plans", [])
        for plan in floor_plans:
            ext = plan.get("extent", [])
            if ext:
                min_x = min(min_x, ext[0])
                max_x = max(max_x, ext[1])
                min_y = min(min_y, ext[2])
                max_y = max(max_y, ext[3])
                updated = True

        if not updated:
            room_map = self.config.get("room_map", {})
            if room_map:
                min_x = room_map["origin"][0]
                max_x = room_map["origin"][0] + room_map["width"]
                min_y = room_map["origin"][1]
                max_y = room_map["origin"][1] + room_map["height"]
                updated = True

        if not updated:
            # Резервные значения, если конфигурация пуста
            min_x, max_x = 0.0, 10.0
            min_y, max_y = 0.0, 10.0

        margin = max((max_x - min_x) * 0.1, 1.0)
        return (min_x - margin, max_x + margin), (min_y - margin, max_y + margin)

    def update_global_map(self, fig, ax, all_positions: Dict[int, List[Tuple[int, Tuple[float, float]]]]):
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        if current_xlim == (0.0, 1.0):  # Default Matplotlib lim
            current_xlim, current_ylim = self.calculate_limits()
        ax.clear()
        self.draw_background(ax)
        colors = ['blue', 'green', 'orange', 'purple', 'red']

        xlim_default, ylim_default = self.calculate_limits()
        map_width = xlim_default[1] - xlim_default[0]
        map_height = ylim_default[1] - ylim_default[0]
        scale_factor = self.config.get("map_scale_factor", 1.0)
        base_font = 10 * scale_factor
        base_marker = 100 * scale_factor

        for i, cam in enumerate(self.config["cameras"]):
            if not cam.get("enabled") or not cam.get("calibrated") or "global_position" not in cam or cam["global_position"] is None or "rotation_deg" not in cam or cam["rotation_deg"] is None:
                continue
            color = colors[i % len(colors)]
            terr_rel = cam["calibration"].get("terrain_points", [])
            if terr_rel:
                terr_global = self.transform_points(terr_rel, cam["global_position"], cam["rotation_deg"])
                poly = plt.Polygon(terr_global, closed=True, edgecolor=color, facecolor=color, alpha=0.2)
                ax.add_patch(poly)
                center = np.mean(terr_global, axis=0)
                ax.text(center[0], center[1], cam["name"], fontsize=base_font, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7))
            ax.plot(cam["global_position"][0], cam["global_position"][1], 'bo', ms=base_marker / 10)

        # Отрисовка людей и их траекторий
        all_people = []
        for pos_list in all_positions.values():
            all_people.extend(pos_list)
        if all_people:
            for track_id, pos in all_people:
                # Отрисовка текущей позиции
                ax.scatter([pos[0]], [pos[1]], c='red', s=base_marker, alpha=1.0, zorder=5)
                ax.text(pos[0], pos[1] + 0.5 * scale_factor, f"P{track_id}", fontsize=base_font * 0.9, ha='center')
                # Отрисовка следа
                history = self.track_history[track_id]
                if len(history) > 1:
                    for i in range(len(history) - 1):
                        x1, y1, _ = history[i]
                        x2, y2, _ = history[i + 1]
                        alpha = 0.5 * (i + 1) / len(history)  # Увеличенная прозрачность для видимости
                        ax.plot([x1, x2], [y1, y2], c='red', alpha=alpha, linewidth=2, zorder=4)

        ax.set_aspect('equal')
        ax.set_title("Карта помещения: положение людей", fontsize=16 * scale_factor)
        ax.set_xlabel("X (м)")
        ax.set_ylabel("Y (м)")

        ax.set_xlim(current_xlim)
        ax.set_ylim(current_ylim)
        fig.canvas.draw()
        plt.pause(0.01)
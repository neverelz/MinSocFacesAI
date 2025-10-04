import cv2
import numpy as np
import time


class EarlyFireDetector:
    def __init__(self,
                 history: int = 120,
                 var_threshold: int = 16,
                 min_region_area_ratio: float = 0.0018,
                 temporal_window: int = 8,
                 enable_optical_flow: bool = True):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.min_region_area_ratio = min_region_area_ratio
        self.temporal_window = temporal_window
        self.prev_fire_masks = []
        self.prev_smoke_masks = []
        self.enable_optical_flow = enable_optical_flow
        self.prev_gray_small = None

    @staticmethod
    def _preprocess(bgr):
        # Контраст + легкая нормализация баланса белого
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        bgr_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return bgr_eq

    @staticmethod
    def _fire_color_mask(bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Более чувствительный, но всё ещё разумный огонь
        saturated = s > 70
        not_dark = v > 60

        # Расширенный, но контролируемый диапазон оттенков
        red_orange = (h <= 22) | (h >= 165)
        orange_yellow = (h > 22) & (h <= 48)
        fire_hue = red_orange | orange_yellow

        mask = fire_hue & saturated & not_dark
        return mask.astype(np.uint8) * 255

    @staticmethod
    def _skin_mask(bgr):
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        return cv2.inRange(ycrcb, lower, upper)

    @staticmethod
    def _smoke_mask(bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Дым: низкая насыщенность, средняя яркость, почти серый
        low_sat = s < 45
        mid_brightness = (v >= 95) & (v <= 220)
        b, g, r = cv2.split(bgr)
        near_gray = (cv2.absdiff(r, g) < 25) & (cv2.absdiff(r, b) < 25) & (cv2.absdiff(g, b) < 25)
        mask = low_sat & mid_brightness & near_gray
        return mask.astype(np.uint8) * 255

    @staticmethod
    def _morph(mask, k=5, iters=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
        return mask

    @staticmethod
    def _boxes_from_mask(mask, min_area):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Умеренный фильтр по соотношению (огонь/дым редко бывают "линиями")
            aspect = w / (h + 1e-6)
            if not (0.25 <= aspect <= 4.0):
                continue
            # Фильтр по плотности (солидности)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            if solidity < 0.60:
                continue
            boxes.append((x, y, x + w, y + h))
        return boxes

    @staticmethod
    def _temporal_score(mask_list, current_mask):
        if len(mask_list) < 2:
            return 0.0
        curr = current_mask.astype(np.float32) / 255.0
        diffs = []
        for prev in mask_list[-min(3, len(mask_list)):]:
            prev_norm = prev.astype(np.float32) / 255.0
            diffs.append(np.mean(cv2.absdiff(curr, prev_norm)))
        return min(1.0, max(0.0, np.mean(diffs)))

    def detect(self, frame):
        h, w = frame.shape[:2]
        min_area = max(300, int(self.min_region_area_ratio * h * w))

        # Препроцесс и движение
        frame_p = self._preprocess(frame)
        fg = self.bg.apply(frame_p)
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        fg = self._morph(fg, 5, 1)

        # Оптический поток (внизскейленный) для проверки направления движения дыма
        flow_field = None
        if self.enable_optical_flow:
            gray = cv2.cvtColor(frame_p, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            if self.prev_gray_small is not None:
                flow_field = cv2.calcOpticalFlowFarneback(
                    self.prev_gray_small, small, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
            self.prev_gray_small = small

        boxes, scores, labels = [], [], []

        # === ОГОНЬ ===
        fire_mask = self._fire_color_mask(frame_p)
        skin = self._skin_mask(frame_p)
        fire_mask = cv2.bitwise_and(fire_mask, cv2.bitwise_not(skin))
        fire_mask = cv2.bitwise_and(fire_mask, fg)
        fire_mask = self._morph(fire_mask, 5, 2)

        fire_temporal = self._temporal_score(self.prev_fire_masks, fire_mask)
        self.prev_fire_masks.append(fire_mask.copy())
        if len(self.prev_fire_masks) > self.temporal_window:
            self.prev_fire_masks.pop(0)

        # Требуем достаточно тёплый и насыщенный цвет + некоторое мерцание
        for (x1, y1, x2, y2) in self._boxes_from_mask(fire_mask, min_area):
            roi = frame_p[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            b_roi, g_roi, r_roi = cv2.split(roi)
            r_dom = np.mean(r_roi) / (np.mean(g_roi) + np.mean(b_roi) + 1e-6)
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            s_mean = np.mean(hsv_roi[:, :, 1]) / 255.0
            v_mean = np.mean(hsv_roi[:, :, 2]) / 255.0
            # Эмпирическая формула уверенности для огня
            score = 0.45 * s_mean + 0.20 * v_mean + 0.35 * fire_temporal
            if (r_dom > 0.8) and (s_mean > 0.45) and (v_mean > 0.45) and (fire_temporal > 0.06 or r_dom > 1.05) and score > 0.50:
                boxes.append((x1, y1, x2, y2))
                scores.append(float(score))
                labels.append("FIRE")

        # === ДЫМ (без фильтрации по позиции!) ===
        smoke_mask = self._smoke_mask(frame_p)
        smoke_mask = cv2.bitwise_and(smoke_mask, fg)
        smoke_mask = self._morph(smoke_mask, 5, 1)

        smoke_temporal = self._temporal_score(self.prev_smoke_masks, smoke_mask)
        self.prev_smoke_masks.append(smoke_mask.copy())
        if len(self.prev_smoke_masks) > self.temporal_window:
            self.prev_smoke_masks.pop(0)

        # Дым должен быть мягким, слаботекстурным и подниматься вверх
        for (x1, y1, x2, y2) in self._boxes_from_mask(smoke_mask, min_area):
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var > 140:
                continue
            texture_score = 1.0 - min(1.0, lap_var / 200.0)
            upward_ok = True
            if flow_field is not None:
                # Оценим средний вертикальный поток внутри бокса (в масштабе 0.5)
                sx1, sy1, sx2, sy2 = int(x1 * 0.5), int(y1 * 0.5), int(x2 * 0.5), int(y2 * 0.5)
                sx1 = max(0, min(flow_field.shape[1] - 1, sx1))
                sx2 = max(0, min(flow_field.shape[1], sx2))
                sy1 = max(0, min(flow_field.shape[0] - 1, sy1))
                sy2 = max(0, min(flow_field.shape[0], sy2))
                if sx2 > sx1 and sy2 > sy1:
                    region = flow_field[sy1:sy2, sx1:sx2]
                    if region.size > 0:
                        vy = -np.mean(region[..., 1])  # отрицательное вверх → инвертируем знак
                        upward_ok = vy > 0.06
            score = 0.60 * texture_score + 0.40 * smoke_temporal
            if upward_ok and smoke_temporal > 0.05 and score > 0.62:
                boxes.append((x1, y1, x2, y2))
                scores.append(float(score))
                labels.append("SMOKE")

        return np.array(boxes, dtype=np.int32), np.array(scores, dtype=np.float32), labels


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    detector = EarlyFireDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores, labels = detector.detect(frame)

        for (x1, y1, x2, y2), s, lab in zip(boxes, scores, labels):
            color = (0, 0, 255) if lab == "FIRE" else (180, 180, 180)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{lab} {s:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Fire & Smoke Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
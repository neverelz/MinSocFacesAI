import cv2
import numpy as np

# --- НАСТРОЙКИ ---
CAMERA_INDEX = 0
WINDOW_NAME = "Camera View"
BIRD_VIEW_NAME = "Room Plan"
PLAN_SIZE = (800, 700)

# --- КАЛИБРОВКА: 4 точки (пиксели) ↔ метры ---
img_points = np.array([
    [180, 120],
    [520, 130],
    [540, 400],
    [160, 410]
], dtype=np.float32)

real_points = np.array([
    [0.0, 0.0],
    [5.0, 0.0],
    [5.0, 4.0],
    [0.0, 4.0]
], dtype=np.float32)

H = cv2.findHomography(img_points, real_points)[0]

# --- ФУНКЦИЯ ДЕТЕКЦИИ (заглушка — замени на свою) ---
def detect_face_and_body(frame):
    """
    Возвращает:
        faces: список [x1, y1, x2, y2, score]
        bodies: список [x1, y1, x2, y2, score]
    """
    # Пример: вызов SCRFD
    # bboxes, _ = detector.detect(frame, thresh=0.5)
    # Раздели на лица и тела по размеру
    faces = []
    bodies = []

    # Замени на реальный вызов
    # for box in bboxes:
    #     x1, y1, x2, y2, score = box[:5]
    #     area = (x2 - x1) * (y2 - y1)
    #     if area < 4000:
    #         faces.append(box)
    #     else:
    #         bodies.append(box)

    return faces, bodies

# --- ОСНОВНОЙ ЦИКЛ ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Ошибка: камера не подключена.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces, bodies = detect_face_and_body(frame)
    used_faces = set()  # чтобы одно лицо не привязывалось к двум телам
    people_positions = []

    # --- Обработка каждого бокса тела ---
    for j, body in enumerate(bodies):
        bx1, by1, bx2, by2, bscore = body[:5]
        bcx = (bx1 + bx2) / 2  # центр тела
        bcy = (by1 + by2) / 2

        # Ищем ближайшее лицо
        best_face = None
        min_dist = float('inf')
        best_i = -1
        for i, face in enumerate(faces):
            if i in used_faces:
                continue
            fx1, fy1, fx2, fy2, fscore = face[:5]
            fcx = (fx1 + fx2) / 2
            fcy = (fy1 + fy2) / 2
            dist = np.sqrt((fcx - bcx)**2 + (fcy - bcy)**2)
            if dist < min_dist and dist < 150:  # порог расстояния
                min_dist = dist
                best_face = face
                best_i = i

        # --- Определяем точку для перевода в метры ---
        if best_face is not None:
            fx1, fy1, fx2, fy2, _ = best_face
            fcx = (fx1 + fx2) / 2
            fcy = (fy1 + fy2) / 2
            # Усреднённая точка между лицом и телом
            avg_cx = int((fcx + bcx) / 2)
            avg_cy = int((fcy + bcy) / 2)
            used_faces.add(best_i)  # отмечаем лицо как использованное
        else:
            # Если лица нет — используем центр тела
            avg_cx = int(bcx)
            avg_cy = int(bcy)

        # Рисуем бокс тела
        cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 2)

        # --- Переводим в реальные координаты ---
        point_pixel = np.array([[avg_cx, avg_cy]], dtype=np.float32)
        point_pixel = np.array([point_pixel])
        point_real = cv2.perspectiveTransform(point_pixel, H)
        x_real, y_real = point_real[0][0]
        people_positions.append((x_real, y_real))

        # --- Подпись координат к боксу тела ---
        label = f"({x_real:.1f}, {y_real:.1f}m)"
        # Позиция подписи: над боксом тела
        cv2.putText(frame, label, (int(bx1), int(by1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Доп. визуализация: точка усреднённого положения
        cv2.circle(frame, (avg_cx, avg_cy), 5, (0, 0, 255), -1)

    # --- Отрисовка лиц (для проверки) ---
    for face in faces:
        x1, y1, x2, y2, score = face[:5]
        if score > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

    # --- Отображение плана комнаты ---
    bird_view = np.ones((PLAN_SIZE[1], PLAN_SIZE[0], 3), dtype=np.uint8) * 240

    min_x, max_x = real_points[:, 0].min(), real_points[:, 0].max()
    min_y, max_y = real_points[:, 1].min(), real_points[:, 1].max()
    scale = min(PLAN_SIZE[0] / (max_x - min_x), PLAN_SIZE[1] / (max_y - min_y))

    def to_plan(x, y):
        px = int((x - min_x) * scale)
        py = int((y - min_y) * scale)
        return (px, PLAN_SIZE[1] - py)

    # Оси
    cv2.putText(bird_view, "Room Plan", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    for x_tick in np.arange(min_x, max_x + 1, 1):
        px, py = to_plan(x_tick, min_y)
        cv2.line(bird_view, (px, PLAN_SIZE[1] - py - 5), (px, PLAN_SIZE[1] - py + 5), (150, 150, 150), 1)
        cv2.putText(bird_view, f"{x_tick}m", (px - 10, PLAN_SIZE[1] - py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    for y_tick in np.arange(min_y, max_y + 1, 1):
        px, py = to_plan(min_x, y_tick)
        cv2.line(bird_view, (px - 5, PLAN_SIZE[1] - py), (px + 5, PLAN_SIZE[1] - py), (150, 150, 150), 1)
        cv2.putText(bird_view, f"{y_tick}m", (px + 10, PLAN_SIZE[1] - py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    # Люди на плане
    for i, (x, y) in enumerate(people_positions):
        px, py = to_plan(x, y)
        cv2.circle(bird_view, (px, py), 10, (255, 0, 0), -1)
        cv2.putText(bird_view, f"P{i+1}", (px - 10, py - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # --- Показываем ---
    cv2.imshow(WINDOW_NAME, frame)
    cv2.imshow(BIRD_VIEW_NAME, bird_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
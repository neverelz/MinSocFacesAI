import cv2

def get_ivcam_stream(source="0"):
    """
    Подключение к IVcam:
    - source="0" или "1" -> индекс виртуальной камеры
    - source="http://..." или "rtsp://..." -> сетевой поток
    """
    cap = None

    # Если source число -> открываем как устройство камеры
    if str(source).isdigit():
        cam_index = int(source)
        print(f"[INFO] Trying to open camera with index {cam_index}...")
        cap = cv2.VideoCapture(cam_index)
    else:
        # Иначе считаем, что это URL
        print(f"[INFO] Trying to open video stream at {source}...")
        cap = cv2.VideoCapture(source)

    if not cap or not cap.isOpened():
        raise RuntimeError(f"Cannot open IVcam source: {source}")

    print("[INFO] Camera stream successfully opened.")
    return cap


if __name__ == "__main__":
    # Тест: пробуем открыть как локальную камеру
    try:
        cap = get_ivcam_stream("0")
        ret, frame = cap.read()
        if ret:
            print("[TEST] Successfully grabbed a frame from camera 0.")
        cap.release()
    except Exception as e:
        print("[ERROR]", e)

import time
import cv2


def _try_open_device_with_backends(device_index):
    backends = [
        cv2.CAP_DSHOW,  # DirectShow (часто лучше всего работает с виртуальными камерами в Windows)
        cv2.CAP_MSMF,   # Media Foundation
        cv2.CAP_VFW,    # Video for Windows (устаревший, но иногда помогает)
        cv2.CAP_ANY,
    ]
    last_cap = None
    for backend in backends:
        try:
            print(f"[INFO] Trying device {device_index} with backend {backend}...")
            cap = cv2.VideoCapture(device_index, backend)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
            last_cap = cap
        except Exception:
            # Продолжаем к следующему бэкенду
            pass
    return last_cap


def _try_open_url_with_backends(source_url: str):
    backends = [
        cv2.CAP_FFMPEG,  # лучший выбор для rtsp/http потоков
        cv2.CAP_ANY,
    ]
    last_cap = None
    for backend in backends:
        try:
            print(f"[INFO] Trying URL {source_url} with backend {backend}...")
            cap = cv2.VideoCapture(source_url, backend)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
            last_cap = cap
        except Exception:
            pass
    return last_cap


def get_ivcam_stream(source="1", retries: int = 3, retry_delay_sec: float = 1.0):
    """
    Подключение к IVCam или другой камере.

    - source: "0"/"1" или int -> индекс устройства камеры (IVCam обычно регистрируется как веб-камера)
    - source: "http://...", "rtsp://...", "tcp://..." -> сетевой поток
    - retries: количество попыток подключения
    - retry_delay_sec: пауза между попытками
    """
    cap = None

    def open_once():
        # Если source число -> открываем как устройство камеры с перебором бэкендов
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            cam_index = int(source)
            print(f"[INFO] Trying to open camera device index {cam_index}...")
            return _try_open_device_with_backends(cam_index)
        # Иначе считаем, что это URL (rtsp/http/tcp)
        src = str(source)
        print(f"[INFO] Trying to open video stream at {src}...")
        return _try_open_url_with_backends(src)

    for attempt in range(1, retries + 1):
        cap = open_once()
        if cap is not None and cap.isOpened():
            print("[INFO] Camera stream successfully opened.")
            return cap
        print(f"[WARN] Failed to open source (attempt {attempt}/{retries}).")
        time.sleep(retry_delay_sec)

    raise RuntimeError(f"Cannot open IVcam source: {source}")


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

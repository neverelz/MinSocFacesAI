import cv2
import numpy as np
import os
from datetime import datetime
import threading
import time

class IVCamCapture:
    def __init__(self, screenshot_dir="screenshots"):
        """
        Класс для работы с камерой через iVCam
        """
        self.cap = None
        self.screenshot_dir = screenshot_dir
        self.is_running = False
        self.frame = None
        self.lock = threading.Lock()
        
        # Создаем директорию для скриншотов
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
    
    def find_ivcam_camera(self):
        """
        Поиск камеры iVCam среди доступных устройств с расширенной диагностикой
        """
        print("=" * 60)
        print("ПОИСК КАМЕРЫ iVCam")
        print("=" * 60)
        
        available_cameras = []
        
        # Проверяем камеры от 1 до 15 (начинаем с камеры 1, пропускаем 0)
        for i in range(1, 16):
            print(f"\nПроверка камеры {i}...")
            cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                # Получаем информацию о камере
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                backend = cap.getBackendName()
                
                print(f"  ✓ Камера {i} найдена:")
                print(f"    - Разрешение: {width}x{height}")
                print(f"    - FPS: {fps}")
                print(f"    - Backend: {backend}")
                
                # Пробуем получить кадр для проверки
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"  ✓ Камера {i} работает корректно")
                    available_cameras.append({
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'backend': backend,
                        'working': True
                    })
                else:
                    print(f"  ✗ Камера {i} не может получить кадры")
                    available_cameras.append({
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'backend': backend,
                        'working': False
                    })
            else:
                print(f"  ✗ Камера {i} недоступна")
            
            cap.release()
        
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ПОИСКА:")
        print("=" * 60)
        
        if not available_cameras:
            print("❌ Никаких камер не найдено!")
            print("\nВозможные причины:")
            print("1. iVCam не запущен на телефоне")
            print("2. iVCam Client не запущен на компьютере")
            print("3. Устройства не в одной сети")
            print("4. Проблемы с драйверами")
            return None
        
        print(f"Найдено камер: {len(available_cameras)}")
        for cam in available_cameras:
            status = "✓ РАБОТАЕТ" if cam['working'] else "✗ НЕ РАБОТАЕТ"
            print(f"  Камера {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']} FPS - {status}")
        
        # Ищем рабочую камеру
        working_cameras = [cam for cam in available_cameras if cam['working']]
        
        if not working_cameras:
            print("\n❌ Ни одна камера не работает корректно!")
            print("Попытка использовать камеру 0 как fallback...")
            fallback_camera = self._check_camera_0_fallback()
            if fallback_camera is not None:
                print(f"✅ Используется камера 0 как fallback")
                return fallback_camera
            return None
        
        # Приоритет для iVCam (обычно имеет специфические характеристики)
        # iVCam часто имеет разрешение 1280x720 или 1920x1080
        ivcam_candidates = []
        for cam in working_cameras:
            # Проверяем типичные разрешения iVCam
            if (cam['width'] == 1280 and cam['height'] == 720) or \
               (cam['width'] == 1920 and cam['height'] == 1080) or \
               (cam['width'] == 640 and cam['height'] == 480):
                ivcam_candidates.append(cam)
        
        if ivcam_candidates:
            selected = ivcam_candidates[0]
            print(f"\n🎯 Выбрана камера iVCam: {selected['index']}")
            return selected['index']
        else:
            # Если не нашли по характеристикам, берем первую рабочую
            selected = working_cameras[0]
            print(f"\n⚠️  iVCam не определен точно, выбрана камера: {selected['index']}")
            print("Если это не iVCam, попробуйте:")
            print("1. Перезапустить iVCam на телефоне")
            print("2. Перезапустить iVCam Client на компьютере")
            print("3. Проверить подключение к сети")
            return selected['index']
    
    def _check_camera_0_fallback(self):
        """
        Проверка камеры 0 как fallback (если камера 1 недоступна)
        """
        print("\nПроверка камеры 0 как fallback...")
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()
            
            print(f"  ✓ Камера 0 найдена: {width}x{height} @ {fps} FPS ({backend})")
            
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  ✓ Камера 0 работает корректно")
                cap.release()
                return 0
            else:
                print(f"  ✗ Камера 0 не может получить кадры")
        
        cap.release()
        return None
    
    def start_camera(self, camera_index=None):
        """
        Запуск камеры
        """
        if camera_index is None:
            camera_index = self.find_ivcam_camera()
        
        if camera_index is None:
            print("Не удалось найти камеру")
            return False
        
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print(f"Не удалось открыть камеру {camera_index}")
            return False
        
        # Настройка параметров камеры
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Камера {camera_index} успешно запущена")
        print("Разрешение:", self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("FPS:", self.cap.get(cv2.CAP_PROP_FPS))
        
        self.is_running = True
        return True
    
    def get_frame(self):
        """
        Получение текущего кадра
        """
        if not self.is_running or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            with self.lock:
                self.frame = frame.copy()
            return frame
        return None
    
    def take_screenshot(self):
        """
        Создание скриншота текущего кадра
        """
        if self.frame is None:
            print("Нет доступного кадра для скриншота")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        filepath = os.path.join(self.screenshot_dir, filename)
        
        with self.lock:
            if self.frame is not None:
                cv2.imwrite(filepath, self.frame)
                print(f"Скриншот сохранен: {filepath}")
                return True
        
        return False
    
    def stop_camera(self):
        """
        Остановка камеры
        """
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("Камера остановлена")
    
    def __del__(self):
        """
        Деструктор для корректного закрытия камеры
        """
        self.stop_camera()

def main():
    """
    Основная функция для тестирования
    """
    camera = IVCamCapture()
    
    if not camera.start_camera():
        return
    
    print("\nУправление:")
    print("- Нажмите 'q' для выхода")
    print("- Нажмите 's' для скриншота")
    print("- Нажмите 'SPACE' для скриншота")
    
    try:
        while camera.is_running:
            frame = camera.get_frame()
            if frame is not None:
                # Добавляем информацию на кадр
                cv2.putText(frame, "iVCam Live Feed", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit, 's' or SPACE for screenshot", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('iVCam Live Feed', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') or key == ord(' '):
                    camera.take_screenshot()
    
    except KeyboardInterrupt:
        print("\nПрерывание пользователем")
    
    finally:
        camera.stop_camera()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

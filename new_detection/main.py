#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Основной скрипт для работы с камерой iVCam
Запуск: python main.py
"""

import sys
import os
import cv2
import time
from camera import IVCamCapture
from scrfd_detector import SCRFDDetector


def print_instructions():
    """
    Вывод инструкций по управлению
    """
    print("=" * 50)
    print("iVCam Live Feed + SCRFD Face Detection")
    print("=" * 50)
    print("Управление:")
    print("  'q' или 'ESC' - выход из приложения")
    print("  's' или 'SPACE' - сделать скриншот")
    print("  '+' - увеличить порог confidence")
    print("  '-' - уменьшить порог confidence")
    print("  'h' - показать эту справку")
    print("=" * 50)

def main():
    """
    Основная функция приложения
    """
    print_instructions()
    
    # Создаем экземпляр камеры
    camera = IVCamCapture(screenshot_dir="screenshots")
    
    # Создаем детектор лиц (SCRFD или простой fallback)
    print("Инициализация детектора лиц...")
    detector = None
    detector_type = "None"
    
    # Пробуем SCRFD
    print("Попытка загрузить SCRFD детектор...")
    try:
        scrfd_detector = SCRFDDetector(model_path="models")
        if scrfd_detector.net is not None:
            detector = scrfd_detector
            detector_type = "SCRFD"
            print("✅ SCRFD детектор готов!")
            print(f"Порог confidence: {detector.confidence_threshold}")
            print(f"Порог NMS: {detector.nms_threshold}")
            # Понижаем порог для тестирования
            detector.set_confidence_threshold(0.3)
            print(f"Установлен тестовый порог confidence: {detector.confidence_threshold}")
        else:
            print("❌ SCRFD не загрузился")
    except Exception as e:
        print(f"❌ Ошибка при загрузке SCRFD: {e}")

    if detector is None:
        print("❌ Не удалось загрузить ни один детектор лиц")
        print("Продолжаем без детекции лиц...")
    else:
        print(f"✅ Детектор лиц готов: {detector_type}")
    
    # Запускаем камеру
    print("Запуск камеры...")
    if not camera.start_camera():
        print("Ошибка: Не удалось запустить камеру!")
        print("\nВозможные решения:")
        print("1. Убедитесь, что iVCam запущен на телефоне")
        print("2. Убедитесь, что iVCam Client запущен на компьютере")
        print("3. Проверьте подключение к сети")
        print("4. Перезапустите iVCam приложения")
        print("5. Запустите диагностику: python camera_test.py")
        return 1
    
    print("Камера успешно запущена!")
    print("Нажмите 'h' для справки по управлению")
    
    screenshot_count = 0
    detection_count = 0
    fps_counter = 0
    start_time = time.time()
    
    try:
        while camera.is_running:
            # Получаем кадр
            frame = camera.get_frame()
            if frame is None:
                print("Ошибка: Не удается получить кадр с камеры")
                break
            
            # Детекция лиц (если детектор доступен)
            detections = []
            if detector is not None:
                detections = detector.detect_faces(frame)
                if len(detections) > 0:
                    detection_count += len(detections)
                    print(f"Найдено лиц: {len(detections)}")  # Отладка
                    # Отрисовка детекций
                    frame = detector.draw_detections(frame, detections)
            
            # Добавляем информацию на кадр
            height, width = frame.shape[:2]
            
            # Заголовок
            if detector:
                title = f"iVCam Live Feed + {detector_type} Face Detection"
            else:
                title = "iVCam Live Feed"
            cv2.putText(frame, title, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Информация о кадре
            frame_info = f"Resolution: {width}x{height}"
            cv2.putText(frame, frame_info, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Информация о детекциях
            if detector is not None:
                faces_info = f"Faces: {len(detections)} | Total: {detection_count}"
                cv2.putText(frame, faces_info, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Порог confidence (только для SCRFD)
                if detector_type == "SCRFD":
                    conf_info = f"Confidence: {detector.confidence_threshold:.2f}"
                    cv2.putText(frame, conf_info, (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Счетчик скриншотов
            if screenshot_count > 0:
                screenshot_info = f"Screenshots: {screenshot_count}"
                y_pos = 150 if detector and detector_type == "SCRFD" else 120 if detector else 90
                cv2.putText(frame, screenshot_info, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - start_time
                fps = fps_counter / elapsed
                fps_text = f"FPS: {fps:.1f}"
                y_pos = 180 if detector and detector_type == "SCRFD" else 150 if detector else 120
                cv2.putText(frame, fps_text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Инструкции
            if detector and detector_type == "SCRFD":
                instructions = "q:quit s:screenshot +/-:confidence h:help"
            else:
                instructions = "q:quit s:screenshot h:help"
            cv2.putText(frame, instructions, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Показываем кадр
            if detector:
                window_title = f'iVCam + {detector_type} Face Detection'
            else:
                window_title = 'iVCam Live Feed'
            cv2.imshow(window_title, frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' или ESC
                print("Выход из приложения...")
                break
            elif key == ord('s') or key == ord(' '):  # 's' или пробел
                if camera.take_screenshot():
                    screenshot_count += 1
                    print(f"Скриншот #{screenshot_count} сохранен")
                else:
                    print("Ошибка при создании скриншота")
            elif key == ord('h'):  # 'h' - справка
                print_instructions()
            elif key == ord('+') or key == ord('='):  # '+' - увеличить confidence
                if detector is not None and detector_type == "SCRFD":
                    new_threshold = min(1.0, detector.confidence_threshold + 0.05)
                    detector.set_confidence_threshold(new_threshold)
                    print(f"Порог confidence: {new_threshold:.2f}")
            elif key == ord('-'):  # '-' - уменьшить confidence
                if detector is not None and detector_type == "SCRFD":
                    new_threshold = max(0.1, detector.confidence_threshold - 0.05)
                    detector.set_confidence_threshold(new_threshold)
                    print(f"Порог confidence: {new_threshold:.2f}")
    
    except KeyboardInterrupt:
        print("\nПрерывание пользователем (Ctrl+C)")
    
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1
    
    finally:
        # Корректное завершение
        print("Завершение работы...")
        camera.stop_camera()
        cv2.destroyAllWindows()
        print("Приложение завершено")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)

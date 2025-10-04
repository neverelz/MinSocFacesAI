#!/usr/bin/env python3
"""
gui_test.py — расширенная диагностика GUI проблем на Linux

Этот скрипт тестирует конкретные операции OpenCV GUI, которые могут вызывать проблемы.
"""

import cv2
import numpy as np
import time
import os
import sys
from platform_utils import get_platform_info

def test_basic_window_creation():
    """Тест базового создания окна"""
    print("🪟 Тест создания окна:")
    
    try:
        # Тест 1: Простое создание окна
        cv2.namedWindow('test_window', cv2.WINDOW_NORMAL)
        print("✅ Окно создано")
        
        # Тест 2: Создание изображения
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(img, "Test Image", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        print("✅ Изображение создано")
        
        # Тест 3: Отображение изображения
        cv2.imshow('test_window', img)
        print("✅ Изображение отображено")
        
        # Тест 4: Ожидание событий
        print("⏳ Ожидание вывода изображения (нажмите любую клавишу)...")
        key = cv2.waitKey(1)
        time.sleep(0.1)
        print("✅ waitKey работает")
        
        # Тест 5: Изменение размера окна
        cv2.resizeWindow('test_window', 600, 400)
        cv2.waitKey(1)
        print("✅ Изменение размера работает")
        
        # Тест 6: Mouse callback
        mouse_callback_called = False
        def test_mouse_callback(event, x, y, flags, param):
            nonlocal mouse_callback_called
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_callback_called = True
                print(f"✅ Mouse click detected at ({x}, {y})")
        
        try:
            cv2.setMouseCallback('test_window', test_mouse_callback)
            print("✅ Mouse callback установлен")
            
            # Даем время пользователю протестировать щелчок
            print("⏳ ОЩЕЛКНИТЕ ПО ИЗОБРАЖЕНИЮ для тестирования (или нажмите 'q' для пропуска)...")
            for i in range(50):  # 5 секунд ожидания
                cv2.imshow('test_window', img)
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q') or mouse_callback_called:
                    break
                    
            if mouse_callback_called:
                print("✅ Mouse callback работает!")
            else:
                print("⚠️ Mouse callback не сработал")
                
        except Exception as e:
            print(f"❌ Mouse callback не работает: {e}")
        
        cv2.destroyWindow('test_window')
        print("✅ Окно закрыто")
        
        return True
        
    except Exception as e:
        print(f"❌ Базовая проверка не прошла: {e}")
        return False

def test_multiple_windows():
    """Тест создания нескольких окон"""
    print("\n🪟🪟 Тест множественных окон:")
    
    try:
        # Создаем несколько окон
        window_names = ['window1', 'window2', 'window3']
        for i, name in enumerate(window_names):
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            img = np.full((200, 300, 3), i * 80, dtype=np.uint8)
            cv2.putText(img, f"Window {i+1}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(name, img)
            
        cv2.waitKey(1)
        print("✅ Множественные окна созданы")
        
        # Закрываем все
        for name in window_names:
            cv2.destroyWindow(name)
        print("✅ Множественные окна закрыты")
        
        return True
        
    except Exception as e:
        print(f"❌ Тест множественных окон не прошел: {e}")
        return False

def test_resize_window():
    """Тест изменений размера"""
    print("\n📏 Тест изменения размера окна:")
    
    try:
        cv2.namedWindow('resize_test', cv2.WINDOW_NORMAL)
        
        sizes = [(400, 300), (800, 600), (1200, 900), (400, 300)]
        
        for w, h in sizes:
            cv2.resizeWindow('resize_test', w, h)
            cv2.waitKey(50)  # Даем время на обработку
            
        cv2.destroyWindow('resize_test')
        print("✅ Изменение размера работает")
        return True
        
    except Exception as e:
        print(f"❌ Тест изменения размера не прошел: {e}")
        return False

def test_image_operations():
    """Тест операций с изображением"""
    print("\n🖼️ Тест операций с изображением:")
    
    try:
        # Создаем сложное изображение
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Сложные графические операции
        cv2.circle(img, (320, 240), 100, (0, 255, 0), 3)
        cv2.rectangle(img, (200, 150), (450, 330), (255, 0, 0), 2)
        cv2.line(img, (100, 100), (500, 400), (0, 0, 255), 2)
        
        for i in range(10):
            cv2.putText(img, f"Line {i}", (10, 30 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.namedWindow('image_test', cv2.WINDOW_NORMAL)
        cv2.imshow('image_test', img)
        cv2.waitKey(1)
        
        print("✅ Сложные операции с изображением работают")
        
        cv2.destroyWindow('image_test')
        return True
        
    except Exception as e:
        print(f"❌ Тест операций с изображением не прошел: {e}")
        return False

def test_linux_specific():
    """Специфичные для Linux тесты"""
    print("\n🐧 Linux-специфичные тесты:")
    
    platform_info = get_platform_info()
    
    if not platform_info['is_linux']:
        print("ℹ️ Эти тесты предназначены для Linux")
        return True
    
    print(f"Platform: {platform_info}")
    
    # Проверяем переменные окружения
    display = os.environ.get('DISPLAY')
    print(f"DISPLAY: {display}")
    
    xdg_session = os.environ.get('XDG_SESSION_TYPE')
    print(f"XDG_SESSION_TYPE: {xdg_session}")
    
    wayland_display = os.environ.get('WAYLAND_DISPLAY')
    print(f"WAYLAND_DISPLAY: {wayland_display}")
    
    # Рекомендации
    if not display:
        print("❌ DISPLAY не установлен")
        print("💡 Выполните: export DISPLAY=:0.0")
        return False
    
    if xdg_session == 'wayland':
        print("⚠️ Обнаружен Wayland. Некоторые функции OpenCV могут не работать.")
        print("💡 Попробуйте запустить из X11 сессии")
    
    return True

def interactive_test():
    """Интерактивный тест"""
    print("\n👆 Интерактивный тест:")
    print("Будет открыто окно для тестирования.")
    print("Нажмите любые клавиши для тестирования.")
    print("Нажмите 'q' для выхода.")
    
    cv2.namedWindow('Interactive Test', cv2.WINDOW_NORMAL)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Создаем анимированное изображение
            img = np.zeros((400, 600, 3), dtype=np.uint8)
            
            # Анимация
            t = time.time() - start_time
            center_x = int(300 + 100 + 50 * np.sin(t * 2))
            center_y = int(200 + 50 * np.cos(t * 2))
            
            cv2.circle(img, (center_x, center_y), 30, (0, 255, 0), -1)
            cv2.putText(img, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            fps = frame_count / (time.time() - start_time + 0.001)
            cv2.putText(img, f"FPS: {fps:.1f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(img, "Press 'q' to quit", (10, 370), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Interactive Test', img)
            frame_count += 1
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("✅ Интерактивный тест завершен пользователем")
                break
            elif key != 255:  # Любая клавиша нажата
                print(f"Нажата клавиша: {chr(key) if key else key}")
    
    finally:
        cv2.destroyWindow('Interactive Test')

def main():
    print("🔬 Расширенная диагностика GUI на Linux")
    print("=" * 50)
    
    # Запускаем все тесты
    tests = [
        ("Базовая проверка", test_basic_window_creation),
        ("Множественные окна", test_multiple_windows),
        ("Изменение размера", test_resize_window),
        ("Операции с изображением", test_image_operations),
        ("Linux-специфичные", test_linux_specific),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: ПРОЙДЕН")
            else:
                print(f"❌ {test_name}: НЕ ПРОЙДЕН")
        except Exception as e:
            print(f"❌ {test_name}: ОШИБКА - {e}")
            results.append((test_name, False))
    
    # Сводка
    print("\n" + "="*60)
    print("📊 СВОДКА РЕЗУЛЬТАТОВ:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ ПРОЙДЕН" if result else "❌ НЕ ПРОЙДЕН"
        print(f"  {test_name}: {status}")
    
    print(f"\nОбщий результат: {passed}/{total}")
    
    if passed == total:
        print("🎉 Все тесты пройдены! GUI должен работать корректно.")
        
        # Предлагаем интерактивный тест
        try:
            interactive_test()
        except KeyboardInterrupt:
            print("\nИнтерактивный тест прерван")
        except Exception as e:
            print(f"Ошибка в интерактивном тесте: {e}")
    else:
        print("⚠️ Некоторые тесты не прошли. Возможны проблемы с GUI.")
        print("\n💡 Рекомендации:")
        print("• Установите переменную DISPLAY: export DISPLAY=:0.0")
        print("• Установите необходимые библиотеки для Linux")
        print("• Проверьте права доступа к дисплею")
        print("• Попробуйте запустить из X11 сессии (не Wayland)")
    
    cv2.destroyAllWindows()
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())

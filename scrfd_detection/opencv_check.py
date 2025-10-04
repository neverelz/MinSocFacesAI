#!/usr/bin/env python3
"""
opencv_check.py — диагностика проблем OpenCV в Linux

Этот скрипт помогает диагностировать проблемы с OpenCV и GUI на Linux системах.
"""

import os
import sys
import subprocess
from platform_utils import get_platform_info

def check_display():
    """Проверка настроек дисплея"""
    print("🖥️ Проверка настроек дисплея:")
    
    display_var = os.environ.get('DISPLAY')
    if display_var:
        print(f"✅ DISPLAY установлен: {display_var}")
    else:
        print("❌ DISPLAY не установлен")
        print("💡 Попробуйте: export DISPLAY=:0.0")
        return False
    
    # Проверяем доступность X11
    try:
        result = subprocess.run(['xset', 'q'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ X11 сервер доступен")
        else:
            print("⚠️ X11 сервер может быть недоступен")
    except FileNotFoundError:
        print("⚠️ xset не найден - возможно X11 не установлен")
    except subprocess.TimeoutExpired:
        print("⚠️ Тайм-аут при проверке X11")
    except Exception as e:
        print(f"⚠️ Ошибка при проверке X11: {e}")
    
    return True

def check_opencv_gui():
    """Проверка GUI поддержки OpenCV"""
    print("\n📱 Проверка OpenCV GUI:")
    
    try:
        import cv2
        print(f"✅ OpenCV версия: {cv2.__version__}")
        
        # Проверяем доступные backend'ы
        backends = []
        for attr in dir(cv2):
            if attr.startswith('CAP_'):
                backends.append((attr, getattr(cv2, attr)))
        
        print("🎥 Доступные видео backend'ы:")
        for name, value in sorted(backends, key=lambda x: x[1]):
            print(f"  {name}: {value}")
        
        # Проверяем GUI поддержку
        try:
            test_img = cv2.imread('/dev/null')  # Это всегда будет None, но не ошибка
            if test_img is None:
                print("✅ OpenCV GUI функции доступны")
            else:
                print("✅ OpenCV GUI функции доступны")
        except Exception as e:
            print(f"⚠️ Проблемы с GUI функциями: {e}")
        
        # Пробуем создать тестовое окно
        try:
            cv2.namedWindow('test', cv2.WINDOW_NORMAL)
            cv2.waitKey(1)
            cv2.destroyWindow('test')
            print("✅ Создание окон работает")
        except Exception as e:
            print(f"❌ Создание окон не работает: {e}")
            return False
            
    except ImportError:
        print("❌ OpenCV не установлен")
        return False
    
    return True

def check_linux_packages():
    """Проверка необходимых Linux пакетов"""
    print("\n📦 Проверка системных пакетов:")
    
    packages_to_check = [
        'libgl1-mesa-glx',
        'libglib2.0-0', 
        'libgtk-3-0',
        'python3-opencv',
        'x11-apps'
    ]
    
    for package in packages_to_check:
        try:
            result = subprocess.run(['dpkg', '-l', package], 
                                 capture_exitcode=True, text=True)
            if result.returncode == 0:
                print(f"✅ {package} установлен")
            else:
                print(f"⚠️ {package} не найден")
        except FileNotFoundError:
            print(f"⚠️ dpkg не найден (возможно не Ubuntu/Debian)")
            break
        except Exception as e:
            print(f"⚠️ Ошибка при проверке {package}: {e}")

def suggest_solutions():
    """Предложить решения проблем"""
    print("\n🔧 Рекомендации:")
    
    platform_info = get_platform_info()
    
    if platform_info['is_linux']:
        print("🐧 Linux обнаружен:")
        print("1. Установите необходимые пакеты:")
        print("   sudo apt update")
        print("   sudo apt install python3-opencv libgl1-mesa-glx libglib2.0-0 libgtk-3-0 x11-apps")
        print("\n2. Проверьте переменную DISPLAY:")
        print("   echo $DISPLAY")
        print("   export DISPLAY=:0.0  # если не установлена")
        print("\n3. Для headless режима используйте:")
        print("   ssh -X username@hostname python main.py  # с X11 forwarding")
        print("\n4. Альтернативные решения:")
        print("   • Используйте Docker с X11 forwarding")
        print("   • Попробуйте запуск без GUI (headless mode)")
        
def main():
    print("🔍 Диагностика проблем OpenCV на Linux\\n")
    
    platform_info = get_platform_info()
    if not platform_info['is_linux']:
        print("ℹ️ Этот ск ript предназначен для Linux систем")
        if platform_info['is_windows']:
            print("🪟 Для Windows возможно проще запустить напрямую")
        elif platform_info['is_macos']:
            print("🍎 Для macOS убедитесь что установлен XQuartz")
        return
    
    success = True
    
    # Проверки
    success &= check_display()
    success &= check_opencv_gui()
    check_linux_packages()
    
    suggest_solutions()
    
    if success:
        print("\n🎉 Основные компоненты работают корректно!")
        print("💡 Если проблемы продолжаются, попробуйте:")
        print("   • Перезапустить X11 сервер")
        print("   • Проверить права пользователя")
        print("   • Использовать виртуальное окружение")
    else:
        print("\n❌ Обнаружены проблемы с GUI компонентами")
        print("💡 Рекомендуется устранить проблемы перед запуском основного приложения")

if __name__ == "__main__":
    main()

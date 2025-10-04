#!/usr/bin/env python3
"""
system_check.py — проверка совместимости системы

Запустите этот скрипт перед использованием основного приложения для проверки всех зависимостей.
"""

import sys
import os
from platform_utils import get_platform_info, get_font_candidates, is_gpu_supported, get_optimal_threading_settings

def check_python_version():
    """Проверка версии Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Требуется Python 3.8 или выше")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Проверка основных зависимостей"""
    dependencies = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('psutil', 'PSUtil'),
        ('tensorflow', 'TensorFlow'),
        ('sklearn', 'Scikit Learn'),
        ('insightface', 'InsightFace')
    ]
    
    missing = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} не установлен")
            missing.append(name)
    
    return len(missing) == 0

def check_model_files():
    """Проверка наличия файлов модели"""
    model_path = 'checkpoints/GN_W1.3_S1_ArcFace_epoch46.h5'
    if os.path.exists(model_path):
        print(f"✅ Модель найдена: {model_path}")
        return True
    else:
        print(f"❌ Модель не найдена: {model_path}")
        
        # Попробуем найти в других местах
        alternatives = [
            os.path.join('scrfd_detection', 'checkpoints', 'GN_W1.3_S1_ArcFace_epoch46.h5'),
            '../checkpoints/GN_W1.3_S1_ArcFace_epoch46.h5',
            './checkpoints/GN_W1.3_S1_ArcFace_epoch46.h5'
        ]
        
        for alt in alternatives:
            if os.path.exists(alt):
                print(f"✅ Альтернативная модель найдена: {alt}")
                return True
        
        return False

def check_camera_access():
    """Проверка доступа к камере"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✅ Камера доступна")
                cap.release()
                return True
            else:
                print("⚠️ Камера подключена, но не может захватывать кадры")
        else:
            print("❌ Камера недоступна")
        cap.release()
    except Exception as e:
        print(f"❌ Ошибка при проверке камеры: {e}")
    
    return False

def check_fonts():
    """Проверка доступных шрифтов"""
    candidates = get_font_candidates()
    available_fonts = []
    
    for font_path in candidates:
        try:
            from PIL import ImageFont
            if font_path is None:
                # Проверяем системный шрифт по умолчанию
                font = ImageFont.load_default()
                available_fonts.append("Default system font")
                break
            else:
                font = ImageFont.truetype(font_path, 10)
                available_fonts.append(os.path.basename(font_path))
        except:
            continue
    
    if available_fonts:
        print(f"✅ Доступные шрифты: {', '.join(available_fonts[:3])}")
        return True
    else:
        print("❌ Шрифты недоступны, будет использован системный шрифт")
        return True  # Не критично

def check_gpu():
    """Проверка поддержки GPU"""
    gpu_available, gpu_info = is_gpu_supported()
    if gpu_available:
        print(f"✅ GPU доступен: {gpu_info}")
        return True
    else:
        print(f"⚠️ GPU недоступен: {gpu_info} (будет использоваться CPU)")
        return True  # Не критично

def check_threading():
    """Проверка настроек потоков"""
    threading_settings = get_optimal_threading_settings()
    print(f"✅ Рекомендуемые настройки потоков: {threading_settings}")
    return True

def main():
    print("🔍 Проверка совместимости системы...\\n")
    
    platform_info = get_platform_info()
    print(f"📍 Платформа: {platform_info['platform']}\\n")
    
    checks = [
        check_python_version,
        check_dependencies,
        check_model_files,
        check_camera_access,
        check_fonts,
        check_gpu,
        check_threading
    ]
    
    results = []
    for check in checks:
        results.append(check())
        print()
    
    critical_checks = results[:3]  # Python, зависимости, модель
    
    if all(critical_checks):
        print("🎉 Проверка пройдена! Система готова к работе.")
        
        if not all(results):
            print("\\n⚠️ Некоторые компоненты недоступны, но система может работать:")
            if not results[3]:  # Камера
                print("• Камера недоступна - проверьте подключение и разрешения")
            if not results[5]:  # GPU
                print("• GPU недоступен - будет использован CPU режим")
    else:
        print("❌ Критические ошибки обнаружены. Установите недостающие компоненты.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

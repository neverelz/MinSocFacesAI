# platform_utils.py — кросс-платформенные утилиты

import os
import sys
import platform
from pathlib import Path


def get_platform_info():
    """Возвращает информацию о текущей платформе"""
    return {
        'system': platform.system().lower(),  # 'windows', 'linux', 'darwin'
        'platform': platform.platform(),
        'is_windows': platform.system().lower() == 'windows',
        'is_linux': platform.system().lower() == 'linux',
        'is_macos': platform.system().lower() == 'darwin',
        'architecture': platform.machine()
    }


def get_font_candidates():
    """Возвращает список кандидатов для шрифтов в зависимости от платформы"""
    platform_info = get_platform_info()
    
    if platform_info['is_windows']:
        return [
            "arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "DejaVuSans.ttf"
        ]
    elif platform_info['is_linux']:
        return [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/opentype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Arial.ttf",  # Иногда в Linux может быть
            "arial.ttf",
            "DejaVuSans.ttf"
        ]
    elif platform_info['is_macos']:
        return [
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttf",
            "/System/Library/Fonts/SanFrancisco.ttf",
            "arial.ttf",
            "DejaVuSans.ttf"
        ]
    else:
        # Fallback для неизвестных платформ
        return [
            "arial.ttf",
            "DejaVuSans.ttf"
        ]


def get_optimal_camera_backends():
    """Возвращает приоритетные backend'ы для camera в зависимости от платформы"""
    import cv2
    platform_info = get_platform_info()
    
    if platform_info['is_windows']:
        # Windows: DSHOW обычно работает лучше всего
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif platform_info['is_linux']:
        # Linux: V4L2 обычно работает лучше всего
        return [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
    elif platform_info['is_macos']:
        # macOS: AVFoundation обычно работает лучше всего
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        return [cv2.CAP_ANY]


def safe_makedirs(path, exist_ok=True):
    """Создает директории с обработкой ошибок для разных платформ"""
    path = Path(path).resolve()
    try:
        path.mkdir(parents=True, exist_ok=exist_ok)
        return str(path)
    except PermissionError:
        # Попробуем создать в домашней директории пользователя
        home_path = Path.home() / Path(path).name
        try:
            home_path.mkdir(parents=True, exist_ok=exist_ok)
            return str(home_path)
        except Exception:
            return os.getcwd()  # Fallback к текущей директории
    except Exception as e:
        print(f"Warning: Could not create directory {path}: {e}")
        return os.path.curdir()


def normalize_path(path):
    """Нормализует путь для текущей платформы"""
    return str(Path(path).resolve())


def get_tmp_dir():
    """Возвращает временную директорию в зависимости от платформы"""
    import tempfile
    return tempfile.gettempdir()


def is_gpu_supported():
    """Проверяет поддержку GPU для текущей платформы"""
    platform_info = get_platform_info()
    
    if platform_info['is_windows'] or platform_info['is_linux']:
        # На Windows и Linux поддерживается CUDA
        try:
            import torch
            cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
            return cuda_available, torch.cuda.get_device_name(0) if cuda_available else "No CUDA"
        except ImportError:
            return False, "torch not installed"
        except Exception as e:
            return False, f"Error checking CUDA: {e}"
    
    elif platform_info['is_macos']:
        # macOS поддерживает только MPS (Metal Performance Shaders)
        try:
            import torch
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            return mps_available, "MPS" if mps_available else "No Metal GPU"
        except ImportError:
            return False, "torch not installed"
        except Exception as e:
            return False, f"Error checking Metal: {e}"
    
    else:
        return False, "Unsupported platform"


def get_optimal_threading_settings():
    """Возвращает оптимальные настройки потоков для текущей платформы"""
    import psutil
    
    platform_info = get_platform_info()
    cpu_count = psutil.cpu_count(logical=False)
    
    if platform_info['is_windows']:
        # Windows может быть капризен с большим количеством потоков
        return {
            'max_workers': min(4, cpu_count),
            'thread_pool_size': min(2, cpu_count)
        }
    elif platform_info['is_linux']:
        # Linux лучше справляется с многопоточностью
        return {
            'max_workers': min(8, cpu_count * 2),
            'thread_pool_size': min(4, cpu_count)
        }
    elif platform_info['is_macos']:
        # macOS также хорошо справляется с многопоточностью
        return {
            'max_workers': min(6, cpu_count * 2),
            'thread_pool_size': min(3, cpu_count)
        }
    else:
        return {
            'max_workers': 2,
            'thread_pool_size': 1
        }


if __name__ == "__main__":
    # Тестирование утилит
    info = get_platform_info()
    print(f"Platform: {info}")
    print(f"Font candidates: {get_font_candidates()[:3]}...")
    print(f"GPU support: {is_gpu_supported()}")
    print(f"Threading: {get_optimal_threading_settings()}")

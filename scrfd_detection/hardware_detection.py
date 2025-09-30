# hardware_detection.py

import psutil
import torch

class HardwareLevel:
    CPU_ONLY = "cpu_only"      # AMD / Intel без CUDA
    NVIDIA_GPU = "nvidia_gpu"  # Есть CUDA

def get_cpu_info():
    cores = psutil.cpu_count(logical=False) or 2
    logical_cores = psutil.cpu_count(logical=True) or 4
    freq = psutil.cpu_freq()
    max_freq = freq.max if freq else 2000
    return {
        'physical_cores': cores,
        'logical_cores': logical_cores,
        'max_freq_mhz': max_freq
    }

def get_memory_info():
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    return total_gb

def is_nvidia_gpu_available():
    """Проверяет, доступен ли NVIDIA GPU через CUDA"""
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return True, torch.cuda.get_device_name(0)
    except Exception:
        pass
    return False, "No NVIDIA GPU"

def estimate_hardware_level():
    gpu_available, gpu_name = is_nvidia_gpu_available()

    if gpu_available:
        level = HardwareLevel.NVIDIA_GPU
    else:
        level = HardwareLevel.CPU_ONLY

    details = {
        'cpu_cores_physical': get_cpu_info()['physical_cores'],
        'ram_gb': round(get_memory_info(), 1),
        'gpu_available': gpu_available,
        'gpu_name': gpu_name
    }

    return level, details

def get_optimal_settings(hardware_level):
    if hardware_level == HardwareLevel.NVIDIA_GPU:
        return {
            'process_interval_sec': 0.05,
            'camera_width': 1280,
            'camera_height': 720,
            'camera_fps': 30,
            'use_gpu': True,
            'det_size': (640, 640),
            'recog_batch_size': 8
        }
    else:  # CPU_ONLY
        return {
            'process_interval_sec': 0.35,
            'camera_width': 640,
            'camera_height': 480,
            'camera_fps': 10,
            'use_gpu': False,
            'det_size': (320, 320),      # Меньше — быстрее на CPU
            'recog_batch_size': 1        # CPU не любит большие батчи
        }

def select_hardware_level_interactive(estimated_level):
    print("\n⚙️  Автоматически определён уровень:")
    if estimated_level == HardwareLevel.NVIDIA_GPU:
        print("   ✅ NVIDIA GPU обнаружен — используется GPU")
    else:
        print("   ⚠️  NVIDIA GPU не обнаружен — используется CPU")

    print("   Enter — оставить автоматически определённый уровень\n")
    input("👉 Нажмите Enter для продолжения...")
    return estimated_level
# hardware_detection.py — финальная версия

import psutil
import torch

class HardwareLevel:
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"

def get_cpu_info():
    cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
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

def is_gpu_available():
    if torch.cuda.is_available():
        return True, f"CUDA: {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return True, "Apple MPS"
    else:
        return False, "No GPU"

def estimate_hardware_level():
    cpu = get_cpu_info()
    ram_gb = get_memory_info()
    gpu_available, gpu_name = is_gpu_available()

    score = 0

    if cpu['physical_cores'] >= 8 and cpu['max_freq_mhz'] >= 3500:
        score += 3
    elif cpu['physical_cores'] >= 4 and cpu['max_freq_mhz'] >= 3000:
        score += 2
    else:
        score += 1

    if ram_gb >= 16:
        score += 2
    elif ram_gb >= 8:
        score += 1

    if gpu_available:
        score += 3

    if score >= 7:
        level = HardwareLevel.STRONG
    elif score >= 4:
        level = HardwareLevel.MEDIUM
    else:
        level = HardwareLevel.WEAK

    details = {
        'cpu_cores_physical': cpu['physical_cores'],
        'cpu_max_freq_ghz': round(cpu['max_freq_mhz'] / 1000, 1),
        'ram_gb': round(ram_gb, 1),
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'score': score
    }

    return level, score, details

def get_optimal_settings(hardware_level):
    """Возвращает настройки под уровень железа. Выравнивание лиц ВСЕГДА включено."""
    if hardware_level == HardwareLevel.STRONG:
        return {
            'process_interval_sec': 0.1,
            'camera_width': 1280,
            'camera_height': 720,
            'camera_fps': 30
        }
    elif hardware_level == HardwareLevel.MEDIUM:
        return {
            'process_interval_sec': 1.0,
            'camera_width': 960,
            'camera_height': 540,
            'camera_fps': 15
        }
    else:  # WEAK
        return {
            'process_interval_sec': 4.0,
            'camera_width': 640,
            'camera_height': 480,
            'camera_fps': 10
        }

def select_hardware_level_interactive(estimated_level):
    """Позволяет пользователю выбрать уровень вручную."""
    print("\n⚙️  Автоматически определён уровень железа:", estimated_level.upper())
    print("   Вы можете оставить его или выбрать другой вручную.\n")

    options = {
        "1": HardwareLevel.WEAK,
        "2": HardwareLevel.MEDIUM,
        "3": HardwareLevel.STRONG
    }

    print("   1 — Слабое железо (экономия ресурсов)")
    print("   2 — Среднее железо (баланс)")
    print("   3 — Сильное железо (максимальное качество)")
    print("   Enter — оставить автоматически определённый уровень\n")

    while True:
        choice = input("👉 Выберите уровень (1/2/3) или нажмите Enter: ").strip()
        if choice == "":
            print(f"✅ Используем автоматически определённый уровень: {estimated_level.upper()}")
            return estimated_level
        elif choice in options:
            selected = options[choice]
            print(f"✅ Выбран уровень вручную: {selected.upper()}")
            return selected
        else:
            print("⚠️  Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    level, score, details = estimate_hardware_level()
    print("📊 Оценка производительности системы:")
    print(f"   Уровень: {level.upper()}")
    print(f"   Счёт: {score}/8")
    print(f"   CPU: {details['cpu_cores_physical']} ядер, {details['cpu_max_freq_ghz']} GHz")
    print(f"   RAM: {details['ram_gb']} GB")
    print(f"   GPU: {'✅ ' + details['gpu_name'] if details['gpu_available'] else '❌ Отсутствует'}")
    print()

    settings = get_optimal_settings(level)
    print("⚙️  Рекомендуемые настройки (по умолчанию):")
    for key, value in settings.items():
        print(f"   {key}: {value}")
# Инструкции по установке для разных платформ

## Общие требования
- Python 3.8 или выше
- Git

## Windows

### Подготовка среды
1. Установите Visual Studio Build Tools:
   - Скачайте с сайта Microsoft
   - Или установите через conda: `conda install vc

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Особенности для Windows
- Используется DirectShow для камер
- Поддерживается CUDA для GPU-ускорения
- Шрифты доступны в `C:/Windows/Fonts/`

### Устранение проблем
- Если возникают ошибки с компиляцией C++, установите Visual Studio Build Tools
- Для CUDA убедитесь, что установлены драйвера NVIDIA CUDA Toolkit

## Linux (Ubuntu/Debian)

### Предварительно установите системные библиотеки:
```bash
sudo apt update
sudo apt install python3-dev python3-pip libgl1-mesa-glx libglib2.0-0 libgtk-3-0 libgstreamer1.0-0 libgstream-plugins-base1.0-0
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

### Установка зависимостей:
```bash
pip install -r requirements.txt
```

### Особенности для Linux
- Используется V4L2 для камер
- Поддерживается CUDA для GPU-ускорения
- Шрифты доступны в `/usr/share/fonts/`

### Устранение проблем
- Если не работает камера: `sudo apt install v4l-utils`
- Для графических интерфейсов: убедитесь что установлены библиотеки OpenGL

## Linux (CentOS/RHEL/Fedora)

### Предварительно установите системные библиотеки:
```bash
# CentOS/RHEL
sudo yum install python3-devel gcc gcc-c++ mesa-libGL glib2 gtk3 gstreamer1 gstreamer1-plugins-base

# Fedora
sudo dnf install python3-devel gcc gcc-c++ mesa-libGL glib2 gtk3 gstreamer1 gstreamer1-plugins-base
```

### Установка зависимостей:
```bash
pip install -r requirements.txt
```

## macOS

### Предварительно установите системные библиотеки:
```bash
brew install python3 opencv
```

### Установка зависимостей:
```bash
pip install -r requirements.txt
```

### Особенности для macOS
- Используется AVFoundation для камер
- Поддерживается Metal Performance Shaders (MPS) вместо CUDA
- Шрифты доступны в `/System/Library/Fonts/`
- Используется специальная версия TensorFlow для Apple Silicon

### Устранение проблем
- Для M1/M2 Mac: убедитесь что установлены правильные версии TensorFlow
- Если не работает камера: проверьте разрешения в System Preferences

## Проверка установки

Запустите тест для проверки совместимости:
```bash
python platform_utils.py
```

Это выведет информацию о:
- Текущей платформе
- Доступных шрифтах
- Поддержке GPU
- Рекомендуемых настройках потоков


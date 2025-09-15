- После клонирования репозитория в Windows powershell, в корне проекта необходимо выполнить команду:
### .\download_models.ps1
- После выполнения создадутся папки models, assets, outputs
- Скачаются все модели (.pth, .onnx, .xml, .bin) и тестовое изображение dog.jpg;
- Сразу же установятся зависимости из requirements.txt
- Запустится inference_onnx.py, результат сохранится в outputs/.
- Также сразу обновится .gitignore, в него войдут папки с загруженными моделями
- В каждом проекте заново разворачивать .\download_models.ps1

- Если Windows блокирует скрипты, включи разрешение выполнения в PowerShell (один раз от админа):
### Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

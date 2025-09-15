## MinSocFacesAI — Object Detection Only

### Обзор
- **Назначение**: Локальная детекция предметов с использованием OpenVINO (модели OMZ подходят для коммерческого использования). Все данные хранятся на диске, без отправки в сеть.
- **Запуск**: CLI `core/detector/runner.py` — источник (IVcam/видео/индекс), модель OpenVINO, опциональные метки; вывод: JSONL и (опц.) видео с аннотациями.
- **Данные**:
  - JSONL: `outputs/detections.jsonl` — записи детекций по кадрам.
  - Видео: `outputs/output.mp4` — при включённой опции `--save_video`.

---

## Модули

### core/detector/ov_detector.py
- **Назначение файла**: Обёртка над моделью детекции OpenVINO.
- **Класс** `OpenVINOObjectDetector`:
  - `__init__(...)`: загружает/компилирует модель, настраивает вход/выход, хранит метки и пороги.
  - `preprocess(image_bgr) -> (blob, meta)`: подгоняет кадр под вход модели (letterbox, нормализация).
  - `postprocess_default(raw, meta) -> List[Dict]`: переводит стандартный формат вывода в список `{bbox, score, class_id, class_name}`.
  - `infer(image_bgr) -> List[Dict]`: выполняет полный цикл инференса.

### core/detector/processing.py
- **Назначение файла**: Отрисовка детекций и простая оценка физических размеров.
- **Dataclass** `SizeEstimatorConfig`:
  - `pixels_per_meter: float` — глобальный PPM для перевода пикселей в метры.
- **Класс** `SizeEstimator`:
  - `estimate(bbox, ...)` — оценивает ширину/высоту объекта в метрах по PPM.
- **Функции**:
  - `draw_detections(image, detections, sizes=None) -> image` — рисует прямоугольники и подписи (класс, вероятность, размеры).
  - `append_json_lines(path, records)` — дозаписывает записи в локальный JSONL-файл.

### core/detector/runner.py
- **Назначение файла**: CLI для запуска детекции из источника (включая IVcam).
- **Функции**:
  - `build_estimator(ppm)` — создаёт оценщик размеров с глобальным PPM.
  - `process_stream(source, model, labels, device, out_dir, ppm, display, save_video)` — открывает источник, выполняет детекцию на каждом кадре, сохраняет JSONL и (опц.) видео, показывает окно при `--display`.
  - `main()` — точка входа CLI с аргументами командной строки.

### core/utils/logger.py
- **Назначение файла**: Логирование ошибок и событий на диск.
- **Функции**:
  - `log_error(message)` — пишет ошибки в `logs/errors/YYYY-MM-DD.log`.
  - `log_event(name)` — добавляет строки `[timestamp, name]` в `logs/events/YYYY-MM-DD.csv`.

---

## Удалённые компоненты
- Веб-интерфейс Flask, распознавание лиц, Telegram-уведомления, планировщик отчётов и утилиты, не относящиеся к детекции предметов, удалены для упрощения и локальности.

---

## Требования
- `opencv-python`, `numpy`, `openvino`

Установите зависимости:
```bash
pip install -r requirements.txt
```

---

## Запуск (пример с IVcam)
- Автозапуск: заполните `config.json` (ключи `SOURCE`, `MODEL_PATH`, `LABELS_PATH`, `DEVICE`, `OUT_DIR`, `PPM`, `DISPLAY`, `SAVE_VIDEO`) и запустите:
```bash
python core/detector/runner.py
```
- Переопределить можно через аргументы CLI (они имеют приоритет над конфигом).

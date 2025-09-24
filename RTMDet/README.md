# Реалтайм детекция (RTMDet ONNX) — пример

## Что делает
Запускает инференс ONNX-модели (RTMDet/YOLO-like) через onnxruntime. Исключает людей и животных по COCO-именам.

## Установка
1. Создайте виртуальное окружение и установите зависимости:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

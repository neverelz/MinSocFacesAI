# Реалтайм детекция (RTMDet ONNX) — пример

## Что делает
Запускает инференс ONNX-модели (RTMDet/YOLO-like) через onnxruntime. Исключает людей и животных по COCO-именам.

## Установка
1. Создайте виртуальное окружение и установите зависимости:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt



python mmdeploy/tools/deploy.py `
  mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py `
  mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py `
  checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth `
  mmdetection/demo/demo.jpg `
  --work-dir mmdeploy_model/rtmdet `
  --device cpu `
  --dump-info



python MinSocFacesAI/RTMDet/main.py --model mmdeploy_model/rtmdet/end2end.onnx
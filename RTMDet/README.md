Полная установка
### Виртуальное окружение + библиотеки
   python -m venv venv
   source venv/bin/activate
   pip install -U openmim
   pip install -r requirements.txt

### git clone -b main https://github.com/open-mmlab/mmdeploy.git
### git clone -b 3.x https://github.com/open-mmlab/mmdetection.git

# Развертка модели m 
python mmdeploy/tools/deploy.py `
  mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py `
  mmdetection/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py `
  checkpoints/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth `
  mmdetection/demo/demo.jpg `
  --work-dir mmdeploy_model/rtmdet `
  --device cpu `
  --dump-info

#Это для оптимизации, с ней я пока не разобралась
python mmdeploy/tools/deploy.py `
  mmdeploy/configs/mmdet/detection/detection_openvino_static.py `
  mmdetection/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py `
  checkpoints/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth `
  mmdetection/demo/demo.jpg `
  mmdetection/demo/demo.jpg `
  --work-dir mmdeploy_model/rtmdet_openvino `
  --device cpu `
  --dump-info

#Запуск. Если камера внешняя, то --camera 1 --model...
python MinSocFacesAI/RTMDet/main.py --model mmdeploy_model/rtmdet/end2end.onnx
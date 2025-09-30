import numpy as np
import onnxruntime as ort
from utils import letterbox, scale_coords, nms
import cv2
from classes_coco import COCO_CLASSES
from config import INPUT_SIZE, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD, MAX_DETECTIONS


class ONNXModel:
    def __init__(self, model_path, use_gpu=False, input_size=640):
        """
        model_path : путь к end2endm.onnx
        use_gpu    : попытка использовать CUDA/ TensorRT (для Intel CPU оставляем False)
        input_size : размер, на который приводим входное изображение
        """
        self.model_path = model_path
        self.input_size = input_size
        self.use_gpu = use_gpu
        self.session = self._load_model()

    def _load_model(self):
        # Пытаемся выбрать максимально быстрый провайдер
        if self.use_gpu:
            providers = ['TensorrtExecutionProvider',
                         'CUDAExecutionProvider',
                         'CPUExecutionProvider']
        else:
            # Для Intel CPU можно добавить OpenVINO, если установлен пакет
            try:
                providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
            except Exception:
                providers = ['CPUExecutionProvider']

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
        print("ONNX Runtime providers:", session.get_providers())
        return session

    def preprocess(self, image):
        """Resize + letterbox + нормализация"""
        img0 = image.copy()
        img, ratio, (pad_x, pad_y) = letterbox(img0, new_shape=(self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]  # [1,3,h,w]
        return img, img0.shape[:2], (ratio, (pad_x, pad_y))

    def infer(self, image):
        """Запуск инференса и пост-обработка"""
        img, orig_shape, ratio_pad = self.preprocess(image)
        inp_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {inp_name: img})

        # ======== Обработка вывода ========
        # У тебя модель выдаёт два массива:
        # bboxes: [1,300,5] -> [x1,y1,x2,y2,score]
        # labels: [1,300]
        bboxes, labels = outputs
        bboxes = bboxes[0]    # (300,5)
        labels = labels[0]    # (300,)

        # Фильтруем по confidence
        mask = bboxes[:, 4] >= CONFIDENCE_THRESHOLD
        bboxes = bboxes[mask]
        labels = labels[mask]

        if bboxes.shape[0] == 0:
            return []

        # Масштабируем координаты обратно в размер исходного кадра
        # scale_coords принимает: исходный размер кадра, bbox в формате xyxy,
        # размер инференса, ratio/pad
        boxes_scaled = scale_coords(
            orig_shape, bboxes[:, :4],
            (self.input_size, self.input_size),
            ratio_pad
        )

        scores = bboxes[:, 4]
        cls_ids = labels.astype(int)

        # NMS по каждому классу
        final = []
        for cls in np.unique(cls_ids):
            cls_mask = cls_ids == cls
            boxes_cls = boxes_scaled[cls_mask]
            scores_cls = scores[cls_mask]
            keep_idx = nms(boxes_cls, scores_cls, NMS_IOU_THRESHOLD)
            kept = np.hstack([
                boxes_cls[keep_idx],
                scores_cls[keep_idx][:, None],
                np.full((len(keep_idx), 1), cls)
            ])
            final.append(kept)

        if len(final) == 0:
            return []

        final = np.vstack(final)
        # сортировка по score
        order = final[:, 4].argsort()[::-1]
        final = final[order][:MAX_DETECTIONS]

        # Приводим к удобному формату
        results = []
        for x1, y1, x2, y2, score, cls in final:
            results.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score),
                "class_id": int(cls),
                "class_name": COCO_CLASSES[int(cls)]
            })
        return results

    def draw_detections(self, frame, detections):
        """Отрисовка боксов на кадре"""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            score = det["score"]
            cls_name = det["class_name"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {score:.2f}",
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)
        return frame

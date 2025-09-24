import cv2
import numpy as np

# COCO классы (YOLOX-s)
COCO_CLASSES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
    "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush"
]

def preprocess(image, input_shape=(640, 640)):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = min(input_shape[0]/h, input_shape[1]/w)
    nh, nw = int(h*scale), int(w*scale)
    image_resized = cv2.resize(img, (nw, nh))
    new_img = np.full((input_shape[0], input_shape[1], 3), 114, dtype=np.uint8)
    new_img[:nh, :nw, :] = image_resized
    blob = new_img.transpose(2,0,1)[None, :, :, :].astype(np.float32)/255.0
    return blob, scale

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _generate_grids_and_strides(input_h, input_w, strides=(8, 16, 32)):
    """Генерация сетки для YOLOX. Порядок аргументов: (height, width)."""
    grids = []
    expanded_strides = []
    for stride in strides:
        grid_h = input_h // stride
        grid_w = input_w // stride
        yv, xv = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((grid.size // 2, 1), stride))
    grids = np.concatenate(grids, axis=0)
    expanded_strides = np.concatenate(expanded_strides, axis=0)
    return grids.astype(np.float32), expanded_strides.astype(np.float32)


def _decode_yolox_outputs(pred, input_size=(640, 640)):
    # pred: (N, 85) до сигмоиды, box в формате дельт
    # input_size: (h, w)
    grids, strides = _generate_grids_and_strides(input_size[0], input_size[1])
    # убеждаемся в совместимости размеров
    n = pred.shape[0]
    g = grids[:n]
    s = strides[:n]
    decoded = pred.copy()
    decoded[:, 0:2] = (decoded[:, 0:2] + g) * s
    decoded[:, 2:4] = np.exp(decoded[:, 2:4]) * s
    # применяем сигмоиду к obj и классам
    decoded[:, 4:] = _sigmoid(decoded[:, 4:])
    return decoded


def postprocess(outputs, scale, conf_thres=0.3, iou_thres=0.45, input_size=(640, 640), mode: str = "auto"):
    pred = outputs[0]
    # Ожидаемый формат YOLOX ONNX: (1, N, 85) или (N, 85)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    elif pred.ndim != 2:
        return []

    if pred.size == 0:
        return []

    # Выбор режима декодирования
    if mode not in ("auto", "raw", "decoded"):
        mode = "auto"

    if mode == "raw":
        pred = _decode_yolox_outputs(pred, input_size=input_size)
    elif mode == "decoded":
        # вероятности могут быть логитами — приведём к [0,1] при необходимости
        if pred.shape[1] > 5 and (np.max(pred[:, 4]) > 1.0 or np.max(pred[:, 5:]) > 1.0):
            pred[:, 4:] = _sigmoid(pred[:, 4:])
    else:
        # Авто: пытаемся угадать
        n = pred.shape[0]
        max_cls_logits = float(np.max(pred[:, 5:])) if pred.shape[1] > 5 else 0.0
        max_xy = float(np.max(pred[:, :2])) if pred.shape[1] >= 2 else 0.0
        likely_raw = (n in (8400, 25200)) and (max_cls_logits > 1.5 or max_xy < 10.0)
        if likely_raw:
            pred = _decode_yolox_outputs(pred, input_size=input_size)
        else:
            if max_cls_logits > 1.0 or np.max(pred[:, 4]) > 1.0:
                pred[:, 4:] = _sigmoid(pred[:, 4:])

    boxes_cxcywh = pred[:, :4]
    objectness = pred[:, 4:5]
    class_scores = pred[:, 5:]
    scores = objectness * class_scores

    # Преобразуем в XYXY и масштабируем обратно к исходному изображению
    boxes_xyxy = np.zeros_like(boxes_cxcywh)
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2.0
    boxes_xyxy /= max(scale, 1e-6)

    results = []
    num_classes = scores.shape[1]
    for class_id in range(num_classes):
        cls_scores = scores[:, class_id]
        mask = cls_scores > conf_thres
        if not np.any(mask):
            continue
        cls_boxes_xyxy = boxes_xyxy[mask]
        cls_scores_f = cls_scores[mask].astype(float)

        # cv2.dnn.NMSBoxes ожидает [x, y, w, h]
        cls_boxes_xywh = np.empty_like(cls_boxes_xyxy)
        cls_boxes_xywh[:, 0] = cls_boxes_xyxy[:, 0]
        cls_boxes_xywh[:, 1] = cls_boxes_xyxy[:, 1]
        cls_boxes_xywh[:, 2] = cls_boxes_xyxy[:, 2] - cls_boxes_xyxy[:, 0]
        cls_boxes_xywh[:, 3] = cls_boxes_xyxy[:, 3] - cls_boxes_xyxy[:, 1]

        valid_wh = (cls_boxes_xywh[:, 2] > 1) & (cls_boxes_xywh[:, 3] > 1)
        if not np.any(valid_wh):
            continue
        cls_boxes_xywh = cls_boxes_xywh[valid_wh]
        cls_boxes_xyxy = cls_boxes_xyxy[valid_wh]
        cls_scores_f = cls_scores_f[valid_wh]

        indices = cv2.dnn.NMSBoxes(
            bboxes=cls_boxes_xywh.tolist(),
            scores=cls_scores_f.tolist(),
            score_threshold=conf_thres,
            nms_threshold=iou_thres,
        )
        if indices is None or len(indices) == 0:
            continue
        for i in np.array(indices).flatten():
            # Ограничим координаты границами изображения после обратного масштабирования
            x1, y1, x2, y2 = cls_boxes_xyxy[i]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)
            results.append((np.array([x1, y1, x2, y2], dtype=np.float32), float(cls_scores_f[i]), class_id))
    return results

def draw_boxes(image, results):
    for (box, score, class_id) in results:
        x1, y1, x2, y2 = map(int, box)
        label = f"{COCO_CLASSES[class_id]} {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return image

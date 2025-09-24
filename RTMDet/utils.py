# utils.py
import numpy as np
import cv2

def letterbox(image, new_shape=(640, 640), color=(114,114,114)):
    """Resize and pad image while meeting stride-multiple constraints (common YOLO letterbox)."""
    shape = image.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    # resize
    img = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x_center, y_center, w, h] to [x1,y1,x2,y2]
    y = np.copy(x)
    y[:,0] = x[:,0] - x[:,2] / 2
    y[:,1] = x[:,1] - x[:,3] / 2
    y[:,2] = x[:,0] + x[:,2] / 2
    y[:,3] = x[:,1] + x[:,3] / 2
    return y

def scale_coords(img_shape, coords, input_shape, ratio_pad=None):
    # Rescale coords (x1,y1,x2,y2) from model to original image shape
    if ratio_pad is None:
        gain = min(input_shape[0] / img_shape[0], input_shape[1] / img_shape[1])
        pad = ((input_shape[1] - img_shape[1]*gain)/2, (input_shape[0] - img_shape[0]*gain)/2)
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]
    coords[:, [0,2]] -= pad[0]
    coords[:, [1,3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].round()
    return coords

def nms(boxes, scores, iou_threshold=0.45):
    # boxes: (N,4) x1,y1,x2,y2
    # scores: (N,)
    if len(boxes) == 0:
        return []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def postprocess_onx_output(outputs, img_shape, input_size, conf_thres=0.25):
    """
    Постобработка для популярных экспортов:
    - Если выход (1, N, 85) или (N,85) => YOLOv5/YOLOX-like: x,y,w,h,objectness,cls_scores...
    - Если выход (1, N, 6) => assumed format [x1,y1,x2,y2,score,class]
    Возвращает list of detections: [x1,y1,x2,y2,score,class_id]
    """
    out = outputs
    if isinstance(outputs, (list, tuple)):
        out = outputs[0]
    arr = np.asarray(out)
    if arr.ndim == 3:
        arr = arr[0]
    # Case A: (N,85) or (N,5+num_classes)
    if arr.shape[1] >= 6:
        # assume first 4 = x,y,w,h, 5 = obj_conf, rest class scores
        if arr.shape[1] > 6:
            xywh = arr[:, :4].copy()
            obj_conf = arr[:, 4:5]
            cls_scores = arr[:, 5:]
            cls_ids = cls_scores.argmax(axis=1)
            cls_max = cls_scores.max(axis=1, keepdims=True)
            confs = (obj_conf * cls_max).reshape(-1)
            mask = confs >= conf_thres
            if mask.sum() == 0:
                return np.zeros((0,6))
            xyxy = xywh2xyxy(xywh)
            boxes = xyxy[mask]
            scores = confs[mask]
            classes = cls_ids[mask]
            dets = np.concatenate([boxes, scores[:,None], classes[:,None]], axis=1)
            return dets
        else:
            # shape (N,6) assume x1,y1,x2,y2,score,class
            boxes = arr[:, :4]
            scores = arr[:, 4]
            classes = arr[:, 5].astype(np.int32)
            mask = scores >= conf_thres
            if mask.sum() == 0:
                return np.zeros((0,6))
            dets = np.concatenate([boxes[mask], scores[mask,None], classes[mask,None]], axis=1)
            return dets
    else:
        # Unknown format -> return empty
        return np.zeros((0,6))

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

def postprocess(outputs, scale, conf_thres=0.3, iou_thres=0.45):
    boxes = outputs[0][:, :4]
    scores = outputs[0][:, 4:5] * outputs[0][:,5:]
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:,0] = boxes[:,0] - boxes[:,2]/2
    boxes_xyxy[:,1] = boxes[:,1] - boxes[:,3]/2
    boxes_xyxy[:,2] = boxes[:,0] + boxes[:,2]/2
    boxes_xyxy[:,3] = boxes[:,1] + boxes[:,3]/2
    boxes_xyxy /= scale
    results = []
    for class_id in range(scores.shape[1]):
        cls_scores = scores[:, class_id]
        mask = cls_scores > conf_thres
        if not np.any(mask):
            continue
        cls_boxes = boxes_xyxy[mask]
        cls_scores = cls_scores[mask]
        indices = cv2.dnn.NMSBoxes(cls_boxes.tolist(), cls_scores.tolist(), conf_thres, iou_thres)
        if len(indices) > 0:
            for i in indices.flatten():
                results.append((cls_boxes[i], cls_scores[i], class_id))
    return results

def draw_boxes(image, results):
    for (box, score, class_id) in results:
        x1, y1, x2, y2 = map(int, box)
        label = f"{COCO_CLASSES[class_id]} {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return image

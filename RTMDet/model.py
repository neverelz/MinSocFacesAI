# model.py
import numpy as np
import onnxruntime as ort
from utils import letterbox, postprocess_onx_output, scale_coords
import cv2
from config import INPUT_SIZE, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD, MAX_DETECTIONS

class ONNXModel:
    def __init__(self, model_path, use_gpu=True, input_size=640):
        self.model_path = model_path
        self.input_size = input_size
        self.use_gpu = use_gpu
        self.session = self._load_model()

    def _load_model(self):
        providers = []
        # try TensorRT, then CUDA, then CPU
        if self.use_gpu:
            try:
                providers = ['TensorrtExecutionProvider','CUDAExecutionProvider','CPUExecutionProvider']
            except Exception:
                providers = ['CUDAExecutionProvider','CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
        print("ONNX Runtime providers:", session.get_providers())
        return session

    def preprocess(self, image):
        img0 = image.copy()
        img, ratio, (pad_x, pad_y) = letterbox(img0, new_shape=(self.input_size, self.input_size))
        img = img[:, :, ::-1]  # BGR to RGB
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        img = np.expand_dims(img, 0)
        return img, img0.shape[:2], (ratio, (pad_x, pad_y))

    def infer(self, image):
        img, orig_shape, ratio_pad = self.preprocess(image)
        inp_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {inp_name: img})
        print([o.shape for o in outputs])
        dets = postprocess_onx_output(outputs, orig_shape, (self.input_size, self.input_size),
                                      conf_thres=CONFIDENCE_THRESHOLD)
        if dets.shape[0] == 0:
            return []
        # dets = [x1,y1,x2,y2,score,class]
        # scale back to original image
        dets[:,:4] = scale_coords(orig_shape, dets[:,:4], (self.input_size, self.input_size), ratio_pad)
        # apply NMS per class
        final = []
        for cls in np.unique(dets[:,5]):
            cls_mask = dets[:,5] == cls
            boxes = dets[cls_mask][:,:4]
            scores = dets[cls_mask][:,4]

            keep = []
            if boxes.shape[0] > 0:
                keep_idx = np.array( self._nms(boxes, scores, NMS_IOU_THRESHOLD) , dtype=int)
                kept = dets[cls_mask][keep_idx]
                final.append(kept)
        if len(final) == 0:
            return []
        final = np.vstack(final)
        # sort by score desc
        order = final[:,4].argsort()[::-1]
        final = final[order][:MAX_DETECTIONS]
        # return list of dicts
        results = []
        for r in final:
            x1,y1,x2,y2,score,cls = r
            results.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score),
                "class_id": int(cls)
            })
        return results

    def _nms(self, boxes, scores, iou_thr):
        # boxes numpy (N,4)
        from utils import nms
        keep = nms(boxes, scores, iou_thr)
        return keep

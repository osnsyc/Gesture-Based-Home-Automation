import cv2
import numpy as np
from yolov8post import Yolov8_post
from rknnlite.api import RKNNLite
import onnxruntime

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)
    # get maximum value locations
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)
    # get maximum value across x and y axis
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1
    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals

def decode(simcc_x: np.ndarray, simcc_y: np.ndarray,
           simcc_split_ratio) -> tuple[np.ndarray, np.ndarray]:

    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio

    return keypoints, scores

def gold_postprocess(
        frame_shape,
        boxes,
        ratio,
        dw_dh,
        xy,
    ) -> tuple[np.ndarray, np.ndarray]:

    result_boxes = []
    result_scores = []
    if len(boxes) > 0:
        scores = boxes[:, 6:7]
        keep_idxs = scores[:, 0] > 0.5
        scores_keep = scores[keep_idxs, :]
        boxes_keep = boxes[keep_idxs, :]

        if len(boxes_keep) > 0:
            for box, score in zip(boxes_keep, scores_keep):
                class_id = int(box[1])
                x_min = int(max((box[2]-dw_dh[0])/ratio + xy[0],0))
                y_min = int(max((box[3]-dw_dh[1])/ratio + xy[1],0))
                x_max = int(min((box[4]-dw_dh[0])/ratio + xy[0],frame_shape[1]))
                y_max = int(min((box[5]-dw_dh[1])/ratio + xy[1],frame_shape[0]))

                result_boxes.append(
                    [x_min, y_min, x_max, y_max, class_id]
                )
                result_scores.append(
                    score
                )

    return result_boxes, result_scores

def find_most_relevant_head(boxes, head, frame_shape):
    if boxes == []:
        return boxes
    if len(boxes) == 1:
        return boxes[0]

    best_dist = float('inf')
    best_box = []
    for box_id, box in enumerate(boxes):
        dist = (head[0] - box[0])**2 + (head[1] - box[1])**2
        if dist < best_dist:
            best_box = box
            best_dist = dist
    return best_box

def find_most_relevant_hand(boxes, hands, frame_shape):
    if boxes == []:
        return boxes

    idx = [[-1,float('inf')],[-1,float('inf')]]
    for hand_id, hand in enumerate(hands):
        for box_id, box in enumerate(boxes):
            dist = (hand[0] - box[0])**2 + (hand[1] - box[1])**2
            if dist < idx[hand_id][1]:
                idx[hand_id][0] = box_id
                idx[hand_id][1] = dist

    bbox_xxyy_0 = (int(max(boxes[idx[0][0]][0]-boxes[idx[0][0]][2],0)),
                int(max(boxes[idx[0][0]][1]-boxes[idx[0][0]][2],0)),
                int(min(boxes[idx[0][0]][0]+boxes[idx[0][0]][2],frame_shape[1])),
                int(min(boxes[idx[0][0]][1]+boxes[idx[0][0]][2],frame_shape[0])),
                0
            )
    bbox_xxyy_1 = (int(max(boxes[idx[1][0]][0]-boxes[idx[1][0]][2],0)),
                int(max(boxes[idx[1][0]][1]-boxes[idx[1][0]][2],0)),
                int(min(boxes[idx[1][0]][0]+boxes[idx[1][0]][2],frame_shape[1])),
                int(min(boxes[idx[1][0]][1]+boxes[idx[1][0]][2],frame_shape[0])),
                1
            )

    if idx[0][0] != idx[1][0]:
        return [bbox_xxyy_0, bbox_xxyy_1]
    elif idx[0][1] < idx[1][1]:
        return [bbox_xxyy_0]
    else:
        return [bbox_xxyy_1]

class Person_det():
    def __init__(self, model_det_person, model_det_hands, model_post_ort):

        self.rknn_det_person = RKNNLite()
        self.rknn_det_hands = RKNNLite()
        ret1 = self.rknn_det_person.load_rknn(model_det_person)
        ret2 = self.rknn_det_hands.load_rknn(model_det_hands)
        ret1 = self.rknn_det_person.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        ret2 = self.rknn_det_hands.init_runtime(core_mask=RKNNLite.NPU_CORE_1)

        onnxruntime.set_default_logger_severity(3)
        self.post_ort = onnxruntime.InferenceSession(model_post_ort,providers=['CPUExecutionProvider'])
        self.sess_input_name = []
        for out in self.post_ort.get_inputs():
                self.sess_input_name.append(out.name)
        self.sess_output_name = []
        for out in self.post_ort.get_outputs():
            self.sess_output_name.append(out.name)

        self.yolov8_post = Yolov8_post()
    
    def release(self):
        self.rknn_det_person.release()
        self.rknn_det_hands.release()
        del self.post_ort

    def inference_det(self, frame):
        """
        Human detection and body pose estimation
        """
        model_size_det = (640,640)
        model_size_kpt = (512,512)
        frame_shape = frame.shape[:2] #[height, width]

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor1, input1_ratio, (input1_dw, input1_dh) = letterbox(img, model_size_det)
        input_tensor1 = np.expand_dims(input_tensor1, axis=0)

        outputs = self.rknn_det_person.inference(inputs=[input_tensor1])

        person_boxes = self.yolov8_post.postprocess(outputs, input1_ratio, frame_shape, (input1_dw, input1_dh))

        crop_human = None
        crop_xy = (0,0)
        boxes = None
        for i in range(len(person_boxes)):
            pose = person_boxes[i].pose
            if pose[5*3+2] > pose[9*3+2] or pose[6*3+2] > pose[10*3+2]: #right-shoulder,r-hand,l-shoulder,l-hand
                cx = (person_boxes[i].xmin + person_boxes[i].xmax) / 2
                cy = (person_boxes[i].ymin + person_boxes[i].ymax) / 2
                sdx = (person_boxes[i].xmax - person_boxes[i].xmin) / 2 * 1.1 # 1.1 zoom the prebox
                sdy = (person_boxes[i].ymax - person_boxes[i].ymin) / 2 * 1.1 # 1.1 zoom the prebox
                crop_human = frame[int(max(cy-sdy,0)):int(min(cy+sdy,frame_shape[0])),int(max(cx-sdx,0)):int(min(cx+sdx,frame_shape[1]))]
                crop_xy = (person_boxes[i].xmin,person_boxes[i].ymin)

                head_xy = [pose[0*3+1], pose[0*3+2]]
                hand_xy_left = [pose[9*3+1], pose[9*3+2]]
                hand_xy_right = [pose[10*3+1], pose[10*3+2]]
                break
        """
        Hand detection
        """
        if crop_human is not None:
            input_tensor2, input2_ratio, (input2_dw, input2_dh) = letterbox(crop_human, model_size_kpt)
            input_tensor2 = cv2.cvtColor(input_tensor2, cv2.COLOR_BGR2RGB)
            input_tensor2 = np.expand_dims(input_tensor2, axis=0)

            output_detect = self.rknn_det_hands.inference(inputs=[input_tensor2])

            sess_input = {
                    self.sess_input_name[0]: output_detect[0], #x1y1x2y2
                    self.sess_input_name[1]: output_detect[1], #main01_y1x1y2x2
                    self.sess_input_name[2]: output_detect[2], #main01_scores
                    }

            output_detect_post = self.post_ort.run(self.sess_output_name, sess_input)

            boxes, scores = gold_postprocess(frame_shape=frame_shape,
                                    boxes=output_detect_post[0],
                                    ratio=input2_ratio,
                                    dw_dh=(input2_dw, input2_dh),
                                    xy=crop_xy)
            
        hand_boxes = []
        head_boxes = []
        if boxes is not None:
            for idx, box in enumerate(boxes):
                min_x,min_y,max_x,max_y,class_id = box
                
                if class_id == 0: # 0:head, 1:hand
                    sdx = max_x - min_x
                    head_boxes.append([cx,cy,sdx])
                elif class_id == 1:
                    cx = (min_x + max_x) / 2
                    cy = (min_y + max_y) / 2
                    sd = max((max_x - min_x),(max_y - min_y))
                    hand_boxes.append([cx,cy,sd])

            head_boxes = find_most_relevant_head(head_boxes, head_xy, frame_shape)
            hand_boxes = find_most_relevant_hand(hand_boxes, [hand_xy_left, hand_xy_right], frame_shape)

        return person_boxes, hand_boxes, head_boxes

# def inference_pose(rknn_lite, frame, box):
def inference_pose(rknn_lite, frame, box):
    if box is None:
        return None, None, None
    model_size_kpt = (256,256)
    frame_shape = frame.shape[:2] #[height, width]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    crop_hand = img[box.y1:box.y2,box.x1:box.x2]
    #TODO ZeroDivisionError
    input_tensor2, input2_ratio, (input2_dw, input2_dh) = letterbox(crop_hand, model_size_kpt)
    input_tensor2 = np.expand_dims(input_tensor2, axis=0)

    output_hand = rknn_lite.inference(inputs=[input_tensor2])

    # hand post
    simcc_x, simcc_y = output_hand
    landmarks, scores = decode(simcc_x, simcc_y, 2.0)
    landmarks_rel = (landmarks - np.array([input2_dw, input2_dh])) / np.array([input2_ratio, input2_ratio]) + np.array([box.x1, box.y1])

    return landmarks_rel[0], scores[0], box.handness

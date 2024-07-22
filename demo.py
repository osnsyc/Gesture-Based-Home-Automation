import cv2
import time
import os
import argparse

from rknnpool import PosePoolExecutor
from func import Person_det, inference_pose
from handtracker import HandTracker
from renderer import Renderer

current_directory = os.getcwd()
print("Current Directory:", current_directory)

base_dir = os.path.dirname(os.path.abspath(__file__))
print("base_dir:", base_dir)

parser = argparse.ArgumentParser(description='Process some integers.')
# # basic params
parser.add_argument('--model1_path', type=str, default='model/yolov8n-pose.rknn', help='model path, could be .pt or .rknn file')
parser.add_argument('--model2_path', type=str, default='model/gold_yolo_n_head_hand_0190_0.4332_1x3x512x512.rknn', help='model path, could be .pt or .rknn file')
parser.add_argument('--onnxmodel_path', type=str, default='model/gold_yolo_n_head_hand_0190_0.4332_1x3x512x512_post.onnx', help='model path, could be .pt or .rknn file')
parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
parser.add_argument('--video_path', type=str, default='video.mp4', help='video path for inference')
    
args = parser.parse_args()
print(vars(args))

video_path = os.path.join(base_dir, args.video_path)

# cap = cv2.VideoCapture(video_path)
# start_time_ms = 0*1000  
# cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_EXPOSURE, 500)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 10)

model1_path = os.path.join(base_dir, args.model1_path)
model2_path = os.path.join(base_dir, args.model2_path)
onnxmodel_path = os.path.join(base_dir, args.onnxmodel_path)

# Number of threads, increasing this can improve the frame rate
TPEs = 3
pool_pose = PosePoolExecutor(
    rknnModel='model/rtmpose_hand_1x3x256x256.rknn',
    TPEs=TPEs,
    func=inference_pose)

frames, loopTime, initTime= 0, time.time(), time.time()

tracker = HandTracker((720,1280),16)
render  = Renderer(True, True, './out.mp4')

person_det = Person_det(model1_path, model2_path, onnxmodel_path)

# Initialize the frames needed for asynchronous processing
if (cap.isOpened()):
    for i in range(TPEs + 0):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool_pose
            exit(-1)
        pool_pose.put(frame, None)
        if i == TPEs - 1:
            frame_pre = frame

while (cap.isOpened()):
    frames += 1
    ret, frame = cap.read()
    if not ret:
        break

    box_num = tracker.get_tracked_num()
    if box_num:
        results = []
        for idx in range(box_num):
            res, flag = pool_pose.get()
            if flag and res[0] is not None:
                results.append(res)
        
        hands = tracker.update_tracker(results)
        for hand in hands:
            if hand.useful:
                pool_pose.put(frame, hand)

        render.draw(frame_pre, hands, None, tracker.point_history)
    else:
        
        persons, hands, head = person_det.inference_det(frame)
        
        if hands:
            tracker.reset_tracker(hands, head)
        else:
            render.draw(frame_pre, None, persons, tracker.point_history)
    
    tracker.update_point_history()
    
    # fit the window
    frame_pre = cv2.resize(frame_pre, (1280,720), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Renderer', frame_pre)
    window_name = 'Renderer'
    cv2.waitKey(1)
    rect = cv2.getWindowImageRect(window_name)
    window_width = rect[2]
    window_height = rect[3]
    screen_width = cv2.getWindowImageRect('Renderer')[2]
    screen_height = cv2.getWindowImageRect('Renderer')[3]
    x = int((screen_width - window_width) / 2)
    y = int((screen_height - window_height) / 2)
    cv2.moveWindow(window_name, x, y)

    frame_pre = frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frames % 30 == 0:
        print("Average frame rate for 30 frames:\t", 30 / (time.time() - loopTime), "frames")
        loopTime = time.time()

print("Overall average frame rate\t", frames / (time.time() - initTime))

person_det.release()
cap.release()
pool_pose.release()
cv2.destroyAllWindows()


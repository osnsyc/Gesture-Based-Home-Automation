import sys
import datetime
import time
from time import monotonic
import threading
from flask import Flask, request, jsonify
import cv2

from rknnpool import PosePoolExecutor
from func import Person_det, inference_pose
from handtracker import HandTracker


ALL_POSES = ["ONE", "TWO", "THREE", "OPEN", "FIST", "Unknown"]
ALL_LINES = ["Up", "Down", "Left", "Right", "Unknown"]

# Default values for config parameters
# Each one of these parameters can be superseded by a new value if specified in client code
DEFAULT_CONFIG = {
    'pose_params': 
    {
        "callback": "_DEFAULT_",
        "hand": "left",
        "trigger": "enter", 
        "first_trigger_delay": 0.2, 
        "next_trigger_delay": 0.2, 
        "max_missing_frames": 5,
    },

    'tracker': 
    { 
        'args': 
        {
            "device_id": 0,
            "frame_height": 720,
            "frame_width": 1280,
            "model_det_person": "model/yolov8n-pose.rknn", 
            "model_det_hands": "model/gold_yolo_n_head_hand_0190_0.4332_1x3x512x512.rknn", 
            "model_post_ort": "model/gold_yolo_n_head_hand_0190_0.4332_1x3x512x512_post.onnx",
            "model_pose_hand": "model/rtmpose_hand_1x3x256x256.rknn",            
            "target": "rk3588",
            "point_history_len": 16,
        },
    },

    'renderer':
    {   
        'enable': True,
        'args':
        {
            'draw_hand': True,
            'draw_body': True,
            'output': None,
        }
    },

    'debug':
    {   
        'enable': True,
    }
}

class Event:
    def __init__(self, category, hand, pose_action, trigger, frame_nb):
        self.category = category
        self.hand = hand
        self.frame_nb = frame_nb
        if isinstance(self.hand, list):
            self.handedness = 'all'
            self.pose = self.hand[0].gesture + ',' + self.hand[1].gesture
            self.line = self.hand[0].point_history_gesture + ',' + self.hand[1].point_history_gesture
            self.xyz_real = str(self.hand[0].xyz_real) + ',' + str(self.hand[1].xyz_real)
        elif self.hand:
            self.handedness = self.hand.handness
            self.pose = self.hand.gesture
            self.line = self.hand.point_history_gesture
            self.xyz_real  = self.hand.xyz_real
        else:
            self.handedness = None
            self.pose = None
            self.line = None
        self.name = pose_action["name"]
        self.callback = pose_action["callback"]
        self.trigger = trigger
        self.time = datetime.datetime.now()
    def print(self):
        attrs = vars(self)
        print("--- EVENT :")
        print('\n'.join("\t%s: %s" % item for item in attrs.items()))
    def print_line(self):
        print(f"{self.time.strftime('%H:%M:%S.%f')[:-3]} : Frame{self.frame_nb} {self.category} {self.name} [{self.pose}] - hand: {self.handedness} - line: {self.line} - xyz: {self.xyz_real} - trigger: {self.trigger} - callback: {self.callback}")

class PoseEvent(Event):
    def __init__(self, hand, pose_action, trigger, frame_nb):
        super().__init__("Pose",
                    hand,
                    pose_action,
                    trigger = trigger,
                    frame_nb = frame_nb)

class EventHist:
    def __init__(self, triggered=False, first_triggered=False, time=0, frame_nb=0):
        self.triggered = triggered
        self.first_triggered = first_triggered
        self.time = time
        self.frame_nb = frame_nb
        
def default_callback(event):
    event.print_line()

def merge_dicts(d1, d2):
    """
    Merge 2 dictionaries. The 2nd dictionary's values overwrites those from the first
    """
    return {**d1, **d2}

def merge_config(c1, c2):
    """
    Merge 2 configs c1 and c2 (where c1 is the default config and c2 the user defined config).
    A config is a python dictionary. The result config takes key:value from c1 if key
    is not present in c2, and key:value from c2 otherwise (key is either present both in 
    in c1 and c2, or only in c2). 
    Note that merge_config is recursive : value can itself be a dictionary.
    """
    res = {}
    for k1,v1 in c1.items():
        if k1 in c2:
            if isinstance(v1, dict):
                assert isinstance(c2[k1], dict), f"{c2[k1]} should be a dictionary"
                res[k1] = merge_config(v1, c2[k1])
            else:
                res[k1] = c2[k1]
        else:
            res[k1] = v1
    for k2,v2 in c2.items():
        if k2 not in c1:
            res[k2] = v2
    return res

def check_mandatory_keys(dic, mandatory_keys):
    """
    Check that mandatory keys are present in a dic
    """
    for k in mandatory_keys:
        assert k in dic.keys(), f"Mandatory key '{k}' not present in {dic}"

def is_point_in_space(point, bounds):
    """
    Determine if a point is within a given 3D space.
    """
    x, y, z = point
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    return (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max)

class HandController:
    def __init__(self, config={}):
        self.config = merge_config(DEFAULT_CONFIG, config)
        # HandController will run callback functions defined in the calling app
        # self.caller_globals contains the globals from the calling app (including callbacks)
        self.caller_globals = sys._getframe(1).f_globals # Or vars(sys.modules['__main__'])
        # Parse pose config
        # Pose list is stored in self.poses
        self.parse_poses()
        # Keep records of previous pose status 
        self.poses_hist = [EventHist() for i in range(len(self.pose_actions))]
        # Init tracker
        self.init_tracker(**self.config['tracker']['args'])

        # Renderer
        self.use_renderer = self.config['renderer']['enable']
        if self.use_renderer:
            from renderer import Renderer
            self.renderer = Renderer(**self.config['renderer']['args'])

        self.debug = self.config['debug']['enable']
        if not self.debug:
            self.init_http_server()

        self.frame_nb = 0

        self.should_pause = False

    def init_tracker(
            self,
            device_id=0,
            frame_height=720,
            frame_width=1280,
            model_det_person='model/yolov8n-pose.rknn', 
            model_det_hands='model/gold_yolo_n_head_hand_0190_0.4332_1x3x512x512.rknn', 
            model_post_ort='model/gold_yolo_n_head_hand_0190_0.4332_1x3x512x512_post.onnx',
            model_pose_hand='model/rtmpose_hand_1x3x256x256.rknn',
            target='rk3588',  #target RKNPU platform
            point_history_len=16,
            vector_x=[-1,0,0],
            vector_y=[0,-1,0],
            vector_o=[0,0,0],
        ):

        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 500)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 10)

        self.person_det = Person_det(model_det_person, model_det_hands, model_post_ort)

        # Number of threads, increasing this can improve the frame rate
        TPEs = 2
        # Initialize the rknn pool
        self.pool = PosePoolExecutor(
            rknnModel=model_pose_hand,
            TPEs=TPEs,
            func=inference_pose
            )

        # Initialize the frames needed for asynchronous processing
        if (self.cap.isOpened()):
            for i in range(TPEs + 0):
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.release()
                    del self.pool
                    exit(-1)
                self.pool.put(frame, None)
                if i == TPEs-1:
                    self.frame_pre = frame

        self.tracker = HandTracker((frame_height,frame_width),point_history_len,vector_x,vector_y,vector_o)

    def init_http_server(self):
        app = Flask(__name__)

        @app.route('/update_status', methods=['POST'])
        def update_status():
            data = request.json
            if 'hasperson' in data:
                self.should_pause = not data['hasperson']
            return jsonify({"status": "success"})

        def run_server():
            app.run(port=5000, debug=False)

        threading.Thread(target=run_server).start()

    def parse_poses(self):
        """
        The part of the config related to poses looks like: 
        'pose_params': {"trigger": "enter", 
                        "first_trigger_delay":0.6, 
                        "next_trigger_delay":0.6, 
                        "max_missing_frames":3},
    
        'pose_actions' : [
            {'name': 'LIGHT', 'pose':'ONE', 'hand':'left', 'callback': 'set_context'},
            {'name': 'TV', 'pose':'TWO', 'hand':'left', 'callback': 'set_context'},
            ]
        
        In the 'pose_actions' list, one element is a dict which have :
            - 2 mandatory key: 
                - name: arbitrary name chosen by the user,
                - pose : one or a list of poses (from the predefined poses listed in ALL_POSES)
                            or keyword 'ALL' to specify any pose from ALL_POSES 
            - optional keys which are the keys of DEFAULT_CONFIG['pose_params']:
                - hand: specify the handedness = hand used to make the pose.
                        Values: 'left', 'right', 'all' 
        """
        mandatory_keys = ['name', 'hand', 'pose', 'line', 'region']
        optional_keys = self.config['pose_params'].keys()
        self.pose_actions = []
        if 'pose_actions' in self.config:
            for pa in self.config['pose_actions']:
                check_mandatory_keys(pa, mandatory_keys)
                
                hand = pa['hand']
                pose = pa['pose']
                line = pa['line']
                region = pa['region']

                # Check value
                assert hand in ['left', 'right', 'all'], f"Incorrect hand! Optional keys: 'left', 'right', 'all'."

                assert isinstance(pose, list), f"Incorrect pose! Pose should be a list."
                for x in pose:
                    assert x in ALL_POSES or x == 'all', f"Incorrect pose {x} in {pa} !"
                
                assert isinstance(line, list), f"Incorrect line! Line should be a list."
                for x in line:
                    assert x in ALL_LINES or x == 'all', f"Incorrect line {x} in {pa} !"

                assert region == 'all' or (isinstance(region, list) and len(region) == 6 and all(isinstance(x, (int, float)) for x in region)), f"Incorrect region! Region should be 'all' or a list of 6 integers or floats."

                if hand == 'all':
                    assert len(pose) == 2, f"Incorrect length of pose!"
                    assert len(line) == 2, f"Incorrect length of line!"
                
                optional_args = {k:pa.get(k, self.config['pose_params'][k]) for k in optional_keys}
                mandatory_args = { k:pa[k] for k in mandatory_keys}
                all_args = merge_dicts(mandatory_args, optional_args)

                self.pose_actions.append(all_args)

    def generate_events(self, hands):
        handness_mapping = {
                -1: 'unknown',
                0: 'left',
                1: 'right'
            }
        if hands[0].handness == 1 or (hands[0].handness == -1 and hands[0].handness == 0):
            hands[0], hands[1] = hands[1], hands[0]

        events = []
        #TODO optimize
        for i, pa in enumerate(self.pose_actions):

            hist = self.poses_hist[i]
            trigger = pa['trigger']

            if pa['hand'] == 'all' and hands[0].useful and hands[1].useful:
                if (pa['pose'][0] == 'all' or hands[0].gesture in pa['pose'][0]) and \
                   (pa['pose'][1] == 'all' or hands[1].gesture in pa['pose'][1]) and \
                   (pa['line'][0] == 'all' or hands[0].point_history_gesture in pa['line'][0]) and \
                   (pa['line'][1] == 'all' or hands[1].point_history_gesture in pa['line'][1]) and \
                   (pa['region'] == 'all' or (is_point_in_space(hands[0].xyz_real, pa['region']) and is_point_in_space(hands[1].xyz_real, pa['region']))):
                    if trigger == "continuous":
                        events.append(PoseEvent(hands, pa, "continuous", frame_nb=self.frame_nb))

                    else: # trigger in ["enter", "enter_leave", "periodic"]:
                        if not hist.triggered:
                            if hist.time != 0 and (self.frame_nb - hist.frame_nb <= pa['max_missing_frames']):
                                if  hist.time and \
                                    ((hist.first_triggered and self.now - hist.time > pa['next_trigger_delay']) or \
                                        (not hist.first_triggered and self.now - hist.time > pa['first_trigger_delay'])):
                                    if trigger == "enter" or trigger == "enter_leave":
                                        hist.triggered = True
                                        events.append(PoseEvent(hands, pa, "enter", frame_nb=self.frame_nb))

                                    else: # "periodic"
                                        hist.time = self.now
                                        hist.first_triggered = True
                                        events.append(PoseEvent(hands, pa, "periodic", frame_nb=self.frame_nb))

                            else:
                                hist.time = self.now
                                hist.first_triggered = False
                        else:
                            if self.frame_nb - hist.frame_nb > pa['max_missing_frames']:
                                hist.time = self.now
                                hist.triggered = False
                                hist.first_triggered = False
                                if trigger == "enter_leave":
                                    events.append(PoseEvent(hands, pa, "leave", frame_nb=self.frame_nb)) 

                hist.frame_nb = self.frame_nb
                continue

            elif pa['hand'] != 'all':
                hand = hands[0 if pa['hand']=='left' else 1]
                if hand.gesture != "Unknown" and \
                    hand.gesture in pa['pose'] and \
                    (pa['region'] == 'all' or is_point_in_space(hand.xyz_real, pa['region'])) and \
                    ('all' in pa['line'] or hand.point_history_gesture in pa['line']) :
                    
                    if trigger == "continuous":
                        events.append(PoseEvent(hand, pa, "continuous", frame_nb=self.frame_nb))

                    else: # trigger in ["enter", "enter_leave", "periodic"]:
                        if not hist.triggered:
                            if hist.time != 0 and (self.frame_nb - hist.frame_nb <= pa['max_missing_frames']):
                                if hist.time and \
                                    ((hist.first_triggered and self.now - hist.time > pa['next_trigger_delay']) or \
                                        (not hist.first_triggered and self.now - hist.time > pa['first_trigger_delay'])):
                                    
                                    if trigger == "enter" or trigger == "enter_leave":
                                        hist.triggered = True
                                        events.append(PoseEvent(hand, pa, "enter", frame_nb=self.frame_nb))

                                    else: # "periodic"
                                        hist.time = self.now
                                        hist.first_triggered = True
                                        events.append(PoseEvent(hand, pa, "periodic", frame_nb=self.frame_nb))

                            else:
                                hist.time = self.now
                                hist.first_triggered = False
                        else:
                            if self.frame_nb - hist.frame_nb > pa['max_missing_frames']:
                                hist.time = self.now
                                hist.triggered = False
                                hist.first_triggered = False
                                if trigger == "enter_leave":
                                    events.append(PoseEvent(hand, pa, "leave", frame_nb=self.frame_nb))

                    hist.frame_nb = self.frame_nb
                    continue

            if hist.triggered and self.frame_nb - hist.frame_nb > pa['max_missing_frames']:
                hist.time = self.now
                hist.triggered = False
                hist.first_triggered = False 
                if trigger == "enter_leave":
                    if pa['hand'] == 'all':
                        events.append(PoseEvent(hands, pa, "leave", frame_nb=self.frame_nb)) 
                    else:
                        events.append(PoseEvent(hands[0 if pa['hand']=='left' else 1], pa, "leave", frame_nb=self.frame_nb)) 
        return events

    def process_events(self, events):
        for e in events:
            if e.callback == "_DEFAULT_":
                default_callback(e)
            else:
                self.caller_globals[e.callback](e)

    def stop_renderer_and_tracker(self):
        self.person_det.release()
        self.cap.release()
        self.pool.release()
        if self.use_renderer:
            self.renderer.output.release()
        self.frame_nb = 0
        
    def restart_renderer_and_tracker(self):
        self.init_tracker(**self.config['tracker']['args'])
        # Renderer
        if self.use_renderer:
            from renderer import TrackerRenderer
            self.renderer = TrackerRenderer(**self.config['renderer']['args'])

    def loop(self):
        try:
            while True:
                if not self.debug:
                    if self.should_pause:
                        self.stop_renderer_and_tracker()
                        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Tracker Pause!")
                        while self.should_pause:
                            time.sleep(1)
                        self.restart_renderer_and_tracker()
                        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Tracker Restart!")

                self.now = monotonic()

                ret, frame = self.cap.read()
                if not ret:
                    break

                box_num = self.tracker.get_tracked_num()
                if box_num:
                    results = []
                    for idx in range(box_num):
                        res, flag = self.pool.get()
                        if flag and res[0] is not None:
                            results.append(res)
                    
                    hands = self.tracker.update_tracker(results)
                    for hand in hands:
                        if hand.useful:
                            self.pool.put(frame, hand)
                    if self.use_renderer:
                        self.renderer.draw(self.frame_pre, hands, None, self.tracker.point_history)
                else:
                    persons, hands, head = self.person_det.inference_det(frame)
                    if hands:
                        self.tracker.reset_tracker(hands, head)
                    else:
                        if self.use_renderer:
                            self.renderer.draw(self.frame_pre, None, persons, self.tracker.point_history)

                self.tracker.update_point_history()

                if self.tracker.get_tracked_num():
                    events = self.generate_events(self.tracker.trackers)
                    self.process_events(events)

                self.frame_nb += 1

                if self.use_renderer:
                    self.frame_pre = cv2.resize(self.frame_pre, (1280,720), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow('Renderer', self.frame_pre)
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
                    cv2.waitKey(1)

                    self.frame_pre = frame

            self.stop_renderer_and_tracker()
        except KeyboardInterrupt: 
            if not self.should_pause:
                self.stop_renderer_and_tracker()

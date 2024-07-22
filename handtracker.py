from collections import deque
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from math import sin, cos, tan, atan2, pi, degrees, floor

@dataclass(frozen=False)
class Hand:
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    handness: int = -1 # -1: Unknown, 0: Left, 1: Right
    score: float = -1
    xyz: tuple = (0,0,0)
    xyz_real: tuple = (0,0,0)
    rotation: float = -4 # -4: Unknown
    gesture: str = "Unknown"
    point_history_gesture: str = "Unknown"
    lm: np.ndarray = field(default_factory=lambda: np.array([]))
    useful: bool = False

class HandTracker:
    def __init__(
            self, 
            frame_shape=(720,1280), 
            point_history_len=16,
            vector_x=[-1,0,0],
            vector_y=[0,-1,0],
            vector_o=[0,0,0],
            ):

        self.frame_shape = frame_shape
        self.point_history_len = point_history_len
        self.vector_x = vector_x / np.linalg.norm(vector_x)
        self.vector_y = vector_y / np.linalg.norm(vector_y)
        self.vector_z = np.cross(self.vector_x, self.vector_y )
        self.vector_o = np.array(vector_o)

        self.trackers = []
        for i in range(2):
            hand = Hand(
                    x1=0,
                    y1=0,
                    x2=0,
                    y2=0,
                    cx=0,
                    cy=0,
                )
            self.trackers.append(hand)

        self.point_history = [deque([(-1,-1)]*self.point_history_len,maxlen=self.point_history_len) for _ in range(2)]

    def reset_tracker(self, hands, head):
        if head:
            head_cx, head_cy, head_sd = head
            """Depth estimation
            165:The pixels at a distance of 1 meter from the camera to the head
            z in centimeter
            """
            z = int(165*100 / head_sd)
        else:
            z = 0
            
        for hand in hands:
            x1, y1, x2, y2, handness = hand
            self.trackers[handness] = Hand(
                                        x1=x1,
                                        y1=y1,
                                        x2=x2,
                                        y2=y2,
                                        cx=0,
                                        cy=0,
                                        handness=handness,
                                        xyz=(0,0,z),
                                        useful=True
                                        )

    def update_tracker(self, hands):
        
        for hand in hands:
            lm, scores, handness = hand

            hand_obj = self._predict_box(lm, scores, handness)
            if hand_obj is not None:
                xyz = self.trackers[handness].xyz
                self.trackers[handness] = hand_obj
                # 0.239*pi: half of the camera's field of view, which is approximately 86 degrees here.
                self.trackers[handness].xyz = (
                        int(xyz[2] * tan(0.239*pi) * 2 * (self.trackers[handness].cx - self.frame_shape[1]/2) / self.frame_shape[1]),
                        int(xyz[2] * tan(0.239*pi) * 2 * (self.trackers[handness].cy - self.frame_shape[0]/2) / self.frame_shape[0]),
                        int(xyz[2])
                    )
                self.trackers[handness].xyz_real = self._get_xyz_real(self.trackers[handness].xyz)
                rotation, gesture = self._recognize_gesture(lm, scores)
                self.trackers[handness].rotation = rotation
                self.trackers[handness].gesture  = gesture
                self.trackers[handness].lm = lm
            else:
                self.trackers[handness].useful = False

        return self.trackers

    def get_tracked_num(self):
        num = 0
        for tracker in self.trackers:
            if tracker.useful:
                num += 1
        return num

    def _get_xyz_real(self, xyz):

        xyz_array = np.array(xyz)
        new_xyz = xyz_array[0] * self.vector_x + xyz_array[1] * self.vector_y + xyz_array[2] * self.vector_z + self.vector_o
        
        return tuple(map(int, new_xyz))


    def _predict_box(self, lm, scores, handness):
        """
        TODO box predictor
        """
        if np.any(lm[:, 0] > self.frame_shape[1]) or \
            np.any(lm[:, 0] < 0) or \
            np.any(lm[:, 1] > self.frame_shape[0]) or \
            np.any(lm[:, 1] < 0):
            #np.any(scores < 0.1) or 
            return None

        min_x = np.min(lm[:, 0])
        max_x = np.max(lm[:, 0])
        min_y = np.min(lm[:, 1])
        max_y = np.max(lm[:, 1])
        if min_x == max_x or min_y == max_y:
            return None

        cx = (min_x + max_x) /2
        cy = (min_y + max_y) /2
        sd = max((max_x - min_x),(max_y - min_y))
        
        hand_obj = Hand(
                x1=int(max(cx-sd,0)),
                y1=int(max(cy-sd,0)),
                x2=int(min(cx+sd,self.frame_shape[1])),
                y2=int(min(cy+sd,self.frame_shape[0])),
                cx=int(cx),
                cy=int(cy),
                handness=handness,
                useful=True,
            )

        return hand_obj

    def _angle(self, a, b, c):
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return np.nan
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def _recognize_gesture(self, landmarks, scores):

        rotation = pi - atan2(((landmarks[5][0] + landmarks[9][0] + landmarks[13][0]) / 3 - landmarks[0][0]), 
                        ((landmarks[5][1] + landmarks[9][1] + landmarks[13][1]) / 3 - landmarks[0][1]))

        rotation = rotation - 2 * pi * floor((rotation + pi) / (2 * pi))
        landmarks_formulated = np.array([
                (x * cos(rotation) + y * sin(rotation), -x * sin(rotation) + y * cos(rotation))
                for x, y in landmarks
            ])
        landmarks_formulated = landmarks_formulated - landmarks_formulated[0]
        # Finger states
        # state: -1=unknown, 0=close, 1=open
        thumb_state, index_state, middle_state, ring_state, little_state = -1, -1, -1, -1, -1
        """
        The thumb is often occluded, so caution should be exercised when using it as a criterion.
        """
        """
        d_3_5 = np.linalg.norm(landmarks_formulated[3]-landmarks_formulated[5])
        d_2_3 = np.linalg.norm(landmarks_formulated[2]-landmarks_formulated[3])
        angle0 = _angle(landmarks_formulated[0], landmarks_formulated[1], landmarks_formulated[2])
        angle1 = _angle(landmarks_formulated[1], landmarks_formulated[2], landmarks_formulated[3])
        angle2 = _angle(landmarks_formulated[2], landmarks_formulated[3], landmarks_formulated[4])

        # for some reason, landmarks overlap would cause angles to be NaN
        if np.isnan(angle0) or np.isnan(angle1) or np.isnan(angle2):
            return rotation, "Unknown"

        thumb_angle = angle0+angle1+angle2
        if thumb_angle > 460 and d_3_5 / d_2_3 > 1.2: 
            thumb_state = 1
        else:
            thumb_state = 0
        """

        if landmarks_formulated[8][1] < landmarks_formulated[7][1] < landmarks_formulated[6][1]:
            index_state = 1
        elif landmarks_formulated[6][1] < landmarks_formulated[8][1]:
            index_state = 0
        else:
            index_state = -1

        if landmarks_formulated[12][1] < landmarks_formulated[11][1] < landmarks_formulated[10][1]:
            middle_state = 1
        elif landmarks_formulated[10][1] < landmarks_formulated[12][1]:
            middle_state = 0
        else:
            middle_state = -1

        if landmarks_formulated[16][1] < landmarks_formulated[15][1] < landmarks_formulated[14][1]:
            ring_state = 1
        elif landmarks_formulated[14][1] < landmarks_formulated[16][1]:
            ring_state = 0
        else:
            ring_state = -1

        if landmarks_formulated[20][1] < landmarks_formulated[19][1] < landmarks_formulated[18][1]:
            little_state = 1
        elif landmarks_formulated[18][1] < landmarks_formulated[20][1]:
            little_state = 0
        else:
            little_state = -1
        # Gesture
        if index_state == 1 and middle_state == 1 and ring_state == 1 and little_state == 1:
            gesture = "OPEN"
        elif index_state == 0 and middle_state == 0 and ring_state == 0 and little_state == 0:
            gesture = "FIST" 
        elif index_state == 1 and middle_state == 0 and ring_state == 0 and little_state == 0:
            gesture = "ONE" 
        elif index_state == 1 and middle_state == 1 and ring_state == 0 and little_state == 0:
            gesture = "TWO" 
        elif index_state == 0 and middle_state == 1 and ring_state == 1 and little_state == 1:
            gesture = "THREE"
        else:
            gesture = "Unknown"

        return rotation, gesture

    def update_point_history(self):
        for hand in self.trackers:
            if hand.useful and hand.lm.any():
                self.point_history[hand.handness].append(tuple(hand.lm[8]))
            else:
                self.point_history[hand.handness].append((-1,-1))

        self._get_point_history_gesture()

    def _get_point_history_gesture(self):
        for hand in self.trackers:

            points = np.array(self.point_history[hand.handness])
            points = points[np.all(points != [-1, -1], axis=1)]
            gesture = 'Unknown'
            if points.shape[0] >= self.point_history_len / 2:
                dist_matrix = squareform(pdist(points))

                mean_distances = np.mean(dist_matrix, axis=1)
                # The Z-score method to detect outliers
                z_scores = (mean_distances - np.mean(mean_distances)) / np.std(mean_distances)
                threshold = 2
                outliers = np.where(z_scores > threshold)[0]
                # Removing outliers
                cleaned_points = np.delete(points, outliers, axis=0)

                min_x, min_y = np.min(cleaned_points, axis=0)
                max_x, max_y = np.max(cleaned_points, axis=0)

                width = max_x - min_x
                height = max_y - min_y
                aspect_ratio = width / height if height != 0 else 0

                x_cleaned = cleaned_points[:, 0].reshape(-1, 1)
                y_cleaned = cleaned_points[:, 1].reshape(-1, 1)

                x_trend_model = LinearRegression().fit(np.arange(len(x_cleaned)).reshape(-1, 1), x_cleaned)
                y_trend_model = LinearRegression().fit(np.arange(len(y_cleaned)).reshape(-1, 1), y_cleaned)
                
                if aspect_ratio < 0.2:
                    if y_trend_model.coef_[0] > 3.0:
                        gesture = 'Down'
                    elif y_trend_model.coef_[0] < -3.0:
                        gesture = 'Up'
                if aspect_ratio > 5.0:
                    if x_trend_model.coef_[0] > 3.0:
                        gesture = 'Left'
                    elif x_trend_model.coef_[0] < -3.0:
                        gesture = 'Right'

                hand.point_history_gesture = gesture

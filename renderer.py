import cv2

skeleton_body = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], 
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], 
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], 
    [1, 3], [2, 4], [3, 5], [4, 6]
]

link_color_body = [
    1,1,2,2,3,
    1,2,3,1,2,
    1,2,3,3,3,
    3,3,1,2
]

palette_body = [
    [0, 0, 0],      # Shadow
    [72, 255, 72],  # Line-Left
    [255, 0, 128],  # Line-Right
    [255, 0, 0],    # Line-Mid
    [79, 91, 221],  # Point
    [72, 218, 240],    # Rect
]

skeleton_hands = [
    (0,1), (1,2), (2,3), (3,4), 
    (5,6), (6,7), (7,8),
    (9,10), (10,11), (11,12),
    (13,14), (14,15), (15,16), 
    (17,18), (18,19), (19,20),
    (0,5), (0,17), (5,9), (9,13), (13,17)
]

link_color_hands = [
    1,1,1,1,
    1,1,1,
    1,1,1,
    1,1,1,
    1,1,1,
    1,1,1,1,1,
]

palette_hand = [
    [0, 0, 0],      # Shadow
    [72, 255, 72],  # Line-Left
    [255, 0, 128],  # Line-Right
    [79, 91, 221],  # Point
    [72, 218, 240], # Text
]

# LINE_WIDTH = 1
# POINT_WIDTH = 2
# POINT_HISTORY_WIDTH = 1
# TEXT_SCALE = 2
# TEXT_WEIGHT = 2
# SHADOW = 1

LINE_WIDTH = 2
POINT_WIDTH = 3
POINT_HISTORY_WIDTH = 1
TEXT_SCALE = 2
TEXT_WEIGHT = 3
SHADOW = 2

# LINE_WIDTH = 3
# POINT_WIDTH = 5
# TEXT_SCALE = 3
# TEXT_WEIGHT = 5
# SHADOW = 3

class Renderer:
    def __init__(self, draw_hand=True, draw_body=False, output=None):

        self.frame = None
        self.draw_hand = draw_hand
        self.draw_body = draw_body

        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.output = cv2.VideoWriter(output,fourcc,24,(1280, 720)) 

    def draw(self, frame, hands, bodys, point_history):
        self.frame = frame

        if hands is not None:
            self._draw_hand(hands)
            self._draw_point_history(point_history)

        if bodys is not None:
            self._draw_body(bodys)

        if self.output:
            self.output.write(self.frame)

        return self.frame

    def _draw_hand(self, hands):
        for hand in hands:
            if hand.useful:
                cv2.rectangle(
                    self.frame,
                    (int(2*hand.x1),int(2*hand.y1)),
                    (int(2*hand.x2),int(2*hand.y2)),
                    palette_hand[1+hand.handness],
                    LINE_WIDTH,
                    cv2.LINE_AA,
                    1
                )
                cv2.putText(self.frame, f"{'RIGHT' if hand.handness else 'LEFT'}", 
                    (int(hand.x1),int(hand.y1)), 
                    cv2.FONT_HERSHEY_PLAIN, TEXT_SCALE, palette_hand[1+hand.handness], TEXT_WEIGHT)

                if hand.lm.any():
                    for (u, v), color in zip(skeleton_hands, link_color_hands):
                        cv2.line(
                            self.frame,
                            tuple(map(int, hand.lm[u][0:2])),
                            tuple(map(int, hand.lm[v][0:2])),
                            palette_hand[color+hand.handness],
                            LINE_WIDTH,
                            cv2.LINE_AA,
                        )
                    for (x, y) in hand.lm:
                        cv2.circle(self.frame, (int(x), int(y)), POINT_WIDTH+SHADOW, palette_hand[0], -1)
                        cv2.circle(self.frame, (int(x), int(y)), POINT_WIDTH, palette_hand[3], -1)
                    
                    cv2.putText(self.frame, f"{hand.rotation:.2f} {hand.gesture}", 
                        (int(hand.lm[0][0]- 80 + +SHADOW), int(hand.lm[0][1] + 30 + +SHADOW)), 
                        cv2.FONT_HERSHEY_PLAIN, TEXT_SCALE, palette_hand[0], TEXT_WEIGHT)
                    cv2.putText(self.frame, f"{hand.rotation:.2f} {hand.gesture}", 
                        (int(hand.lm[0][0]-80), int(hand.lm[0][1] + 30)), 
                        cv2.FONT_HERSHEY_PLAIN, TEXT_SCALE, palette_hand[4], TEXT_WEIGHT)

    def _draw_point_history(self, point_history):
        for i in range(2):
            for j, point in enumerate(point_history[i]):
                if point[0] > 0 and point[1] > 0:
                    cv2.circle(self.frame, (int(point[0]), int(point[1])), POINT_HISTORY_WIDTH + int(j / 2), palette_hand[1+i], -1)

    def _draw_body(self, bodys):
        for body in bodys:
            cv2.rectangle(
                self.frame,
                (int(2*body.xmin+SHADOW),int(2*body.ymin+SHADOW)),
                (int(2*body.xmax+SHADOW),int(2*body.ymax+SHADOW)),
                palette_body[0],
                LINE_WIDTH,
                cv2.LINE_AA,
                1
            )
            cv2.rectangle(
                self.frame,
                (int(2*body.xmin),int(2*body.ymin)),
                (int(2*body.xmax),int(2*body.ymax)),
                palette_body[5],
                LINE_WIDTH,
                cv2.LINE_AA,
                1
            )

            for (u, v), color in zip(skeleton_body, link_color_body):
                x1, y1 = int(body.pose[u*3+1]), int(body.pose[u*3+2])
                x2, y2 = int(body.pose[v*3+1]), int(body.pose[v*3+2])
                cv2.line(
                    self.frame,
                    (x1,y1),
                    (x2,y2),
                    palette_body[color],
                    LINE_WIDTH,
                    cv2.LINE_AA,
                )

            for i in range(17):
                x, y = int(body.pose[i*3+1]), int(body.pose[i*3+2])
                cv2.circle(self.frame, (x, y), POINT_WIDTH+SHADOW, palette_body[0], -1)
                cv2.circle(self.frame, (x, y), POINT_WIDTH, palette_body[4], -1)






    
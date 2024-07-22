import sys
import numpy as np
import cv2
from math import exp

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, pose):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.pose = pose

def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total

def NMS(detectResult, nmsThresh):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs

def sigmoid(x):
    return 1 / (1 + exp(-x))

class Yolov8_post():
    def __init__(self):
        self.CLASSES = ['person']

        self.class_num = len(self.CLASSES)
        self.headNum = 3
        self.keypoint_num = 17

        self.strides = [8, 16, 32]
        self.mapSize = [[80, 80], [40, 40], [20, 20]]
        self.nmsThresh = 0.55
        self.objectThresh = 0.5

        self.meshgrid = []
        self.GenerateMeshgrid()


    def GenerateMeshgrid(self):
        for index in range(self.headNum):
            for i in range(self.mapSize[index][0]):
                for j in range(self.mapSize[index][1]):
                    self.meshgrid.append(j + 0.5)
                    self.meshgrid.append(i + 0.5)


    def postprocess(self, out, ratio, frame_shape, dw_dh):
        img_h, img_w = frame_shape
        dw, dh = dw_dh
        detectResult = []

        output = []
        for i in range(len(out)):
            output.append(out[i].reshape((-1)))

        gridIndex = -2
        
        for index in range(self.headNum):
            reg = output[index * 2 + 0]
            cls = output[index * 2 + 1]
            pose = output[self.headNum * 2 + index]

            for h in range(self.mapSize[index][0]):
                for w in range(self.mapSize[index][1]):
                    gridIndex += 2

                    for cl in range(self.class_num):
                        cls_val = sigmoid(cls[cl * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w])

                        if cls_val > self.objectThresh:
                            x1 = (self.meshgrid[gridIndex + 0] - reg[0 * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w]) * self.strides[index]
                            y1 = (self.meshgrid[gridIndex + 1] - reg[1 * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w]) * self.strides[index]
                            x2 = (self.meshgrid[gridIndex + 0] + reg[2 * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w]) * self.strides[index]
                            y2 = (self.meshgrid[gridIndex + 1] + reg[3 * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w]) * self.strides[index]

                            xmin = (x1 - dw) / ratio
                            ymin = (y1 - dh) / ratio
                            xmax = (x2 - dw) / ratio
                            ymax = (y2 - dh) / ratio

                            xmin = xmin if xmin > 0 else 0
                            ymin = ymin if ymin > 0 else 0
                            xmax = xmax if xmax < img_w else img_w
                            ymax = ymax if ymax < img_h else img_h

                            poseResult = []
                            for kc in range(self.keypoint_num):
                                px = pose[(kc * 3 + 0) * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w]
                                py = pose[(kc * 3 + 1) * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w]
                                vs = sigmoid(pose[(kc * 3 + 2) * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w])

                                x = ((px * 2.0 + (self.meshgrid[gridIndex + 0] - 0.5)) * self.strides[index] - dw) / ratio
                                y = ((py * 2.0 + (self.meshgrid[gridIndex + 1] - 0.5)) * self.strides[index] - dh) / ratio

                                poseResult.append(vs)
                                poseResult.append(x)
                                poseResult.append(y)

                            box = DetectBox(cl, cls_val, xmin, ymin, xmax, ymax, poseResult)
                            detectResult.append(box)

        predBox = NMS(detectResult, self.nmsThresh)
        
        return predBox
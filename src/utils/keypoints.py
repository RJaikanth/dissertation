"""
dissertation.keypoints.py
Author: Raghhuveer Jaikanth
Date  : 25/07/2023

# Enter Description Here
"""


def xyxy2xywhn(bbox, img_size):
    x = int((bbox[0] + bbox[2]) / 2) / img_size[0]
    y = int((bbox[1] + bbox[3]) / 2) / img_size[1]
    w = int(bbox[2] - bbox[0]) / img_size[0]
    h = int(bbox[3] - bbox[1]) / img_size[1]
    return [x, y, w, h]


def xywhn2xyxy(bbox, img_size):
    x1 = int(img_size[1] * (bbox[0] - bbox[2] / 2))
    y1 = int(img_size[0] * (bbox[1] - bbox[3] / 2))
    x2 = int(img_size[1] * (bbox[0] + bbox[2] / 2))
    y2 = int(img_size[0] * (bbox[1] + bbox[3] / 2))
    return [x1, y1, x2, y2]

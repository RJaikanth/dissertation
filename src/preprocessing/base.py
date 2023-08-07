"""
dissertation.base.py
Author: Raghhuveer Jaikanth
Date  : 05/08/2023

# Enter Description Here
"""

from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
from torch.utils.data import Dataset
from src.utils.keypoints import xyxy2xywhn


class PreprocessingBaseClass(Dataset, metaclass=ABCMeta):
    def __init__(self, root: str):
        super().__init__()

        self._root = root

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    def _normalize_kp(self, kps, img_size):
        kps = np.asarray(kps, dtype=float)
        kps = np.divide(kps, img_size).reshape(34, )
        return " ".join(map(lambda x: str(x), kps))

    def _normalize_bbox(self, bbox, img_size):
        bbox = xyxy2xywhn(bbox, img_size)
        return " ".join(map(lambda x: str(x), bbox))

    def _create_annotation(self, kps, bbox, img_size):
        img_size = img_size[:-1]
        return f"{self._normalize_bbox(bbox, img_size)} {self._normalize_kp(kps, img_size)}"

    def _add_offset(self, bbox, img_size, offset=100):
        offset_kp = bbox.copy()
        offset_kp[0] = max(0, offset_kp[0] - offset)
        offset_kp[1] = max(0, offset_kp[1] - offset)
        offset_kp[2] = min(img_size[1], offset_kp[2] + offset)
        offset_kp[3] = min(img_size[0], offset_kp[3] + offset)
        return offset_kp

    def _extract_freq(self, cap, desired_count = 30):
        return round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / desired_count)

"""
dissertation.asp_new.py
Author: Raghhuveer Jaikanth
Date  : 05/08/2023

# ASP Preprocessing Set
"""
import os.path

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.base import PreprocessingBaseClass
from src.utils.c3d import extract_2d_from_c3d
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import DataLoader


class ASPDataPreProcessor(PreprocessingBaseClass):
    def __init__(self, root: str):
        super().__init__(root)

        # Input dirs
        self._BROOT = os.path.join(root, "boxes")
        self._CROOT = os.path.join(root, "cameras")
        self._JROOT = os.path.join(root, "joints_3d")
        self._VROOT = os.path.join(root, "videos")

        # Splits
        col_names = ["sub_id", "clip_id", "trainval", "cam_id"]
        dtypes = {col: str for col in col_names}
        self._splits = pd.read_csv(
            f"{os.path.dirname(root)}/splits.csv",
            names = col_names, dtype = dtypes
        )
        self._splits = self._splits[self._splits['trainval'] != "test"]

        # Output dirs
        self._OIROOT = "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/combined-ds/images"
        self._OLROOT = "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/combined-ds/labels"
        os.makedirs(self._OLROOT, exist_ok = True)
        os.makedirs(self._OIROOT, exist_ok = True)

    def __len__(self):
        return len(self._splits)

    def __getitem__(self, item):
        # Get clip metadata
        sub_id, clip_id, is_train, cam_ids = self._splits.iloc[item, :].values
        cam_ids = ["left", "right", "mid"] if cam_ids == "all" else [cam_ids]

        # Joint File
        joint_file = os.path.join(self._JROOT, sub_id, f"{sub_id}-{clip_id}.c3d")

        # Loop over all cameras
        for cam_id in cam_ids:
            # Input files
            box_file = os.path.join(self._BROOT, sub_id, f"{sub_id}-{clip_id}-{cam_id}.csv")
            cam_file = os.path.join(self._CROOT, sub_id, f"{sub_id}-{cam_id}.json")
            vid_file = os.path.join(self._VROOT, sub_id, f"{sub_id}-{clip_id}-{cam_id}.mkv")

            # Extract joints and bbox
            joints = extract_2d_from_c3d(joint_file, cam_file)
            bboxes = pd.read_csv(box_file).values

            # Read video
            cap = cv2.VideoCapture(vid_file)
            extract_freq = self._extract_freq(cap, 30)
            frame_counter = 0
            save_counter = 0

            while True:
                ret, frame = cap.read()

                if save_counter > 30 or not ret:
                    break

                if (frame_counter % extract_freq == 0):
                    # Get bbox and crop for frame
                    bbox = list(map(lambda x: int(x), bboxes[frame_counter]))
                    crop = self._add_offset(bbox, frame.shape, 200)

                    # Crop and transform image
                    bbox.append("0")
                    transforms = [
                        A.Crop(crop[0], crop[1], crop[2], crop[3], always_apply = True, p = 1.0),
                        A.LongestMaxSize(max_size = 640, interpolation = 1),
                        A.PadIfNeeded(min_height = 640, min_width = 640, border_mode = 4)
                    ]
                    transform = A.Compose(transforms, keypoint_params = A.KeypointParams("xy"), bbox_params = A.BboxParams("pascal_voc"))
                    transformed = transform(image=frame, bboxes=[bbox], keypoints=joints[frame_counter])

                    # Get new image
                    crop_frame = transformed['image']

                    # Pose keypoints
                    kps = np.asarray(transformed['keypoints'])

                    # Offset bbox
                    bbox = np.asarray(transformed["bboxes"][0], dtype = int).tolist()
                    cat_id = bbox.pop()
                    bbox_offset = self._add_offset(bbox, crop_frame.shape, 20)

                    # Create annotation
                    annotation = f"{cat_id} " + self._create_annotation(kps, bbox_offset, crop_frame.shape)

                    base_name = f"{sub_id}-{clip_id}-{cam_id}-{frame_counter:03d}"
                    cv2.imwrite(os.path.join(self._OIROOT, is_train, f"{base_name}.jpg"), crop_frame)
                    with open(os.path.join(self._OLROOT, is_train, f"{base_name}.txt"), 'w') as f:
                        f.write(annotation)
                        f.close()

                    # Log success
                    print(f"{is_train:>5s} {base_name:>19s}: Frame-{frame_counter:03d}.jpg Saved")
                    print(f"{is_train:>5s} {base_name:>19s}: Frame-{frame_counter:03d}.txt Saved")
                    save_counter += 1

                frame_counter += 1

        return []



if __name__ == "__main__":
    ROOT = "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/aspset/trainval"
    ds = ASPDataPreProcessor(ROOT)
    dl = DataLoader(ds, batch_size=64, num_workers=8, shuffle=False)
    for _ in enumerate(ds):
        break

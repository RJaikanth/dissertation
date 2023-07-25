"""
dissertation.asp.py
Author: Raghhuveer Jaikanth
Date  : 25/07/2023

# Enter Description Here
"""
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import albumentations as A

from src.utils.c3d import extract_2d_from_c3d
from src.utils.keypoints import xyxy2xywhn


class ASPPreprocessingDS(Dataset):
    def __init__(self, root: str):
        super().__init__()
        self._BROOT = os.path.join(root, "boxes")
        self._CROOT = os.path.join(root, "cameras")
        self._JROOT = os.path.join(root, "joints_3d")
        self._VROOT = os.path.join(root, "videos")

        col_names = ["sub_id", "clip_id", "trainval", "cam_id"]
        dtypes = {col: str for col in col_names}
        self._splits = pd.read_csv(
            f"{os.path.dirname(root)}/splits.csv",
            names = col_names, dtype = dtypes
        )
        self._splits = self._splits[self._splits['trainval'] != "test"]

        self._resize_transform = A.Compose(
            [A.Resize(640, 640, p = 1, always_apply = True)],
            keypoint_params = A.KeypointParams("xy"),
            bbox_params = A.BboxParams("pascal_voc")
        )

        # Output directories
        self._OIROOT = os.path.join(os.path.dirname(root), "final_ds", "images")
        self._OLROOT = os.path.join(os.path.dirname(root), "final_ds", "labels")
        os.makedirs(self._OLROOT, exist_ok = True)
        os.makedirs(self._OIROOT, exist_ok = True)

    def __len__(self):
        return len(self._splits)

    def __getitem__(self, item):
        # Get clip metadata
        sub_id, clip_id, is_train, cam_ids = self._splits.iloc[item, :].values
        cam_ids = ["left", "right", "mid"] if cam_ids == "all" else [cam_ids]

        # Get joint file
        jnt_file = os.path.join(self._JROOT, sub_id, f"{sub_id}-{clip_id}.c3d")

        # For all camera ids
        for cam_id in cam_ids:
            # Get path
            box_file = os.path.join(self._BROOT, sub_id, f"{sub_id}-{clip_id}-{cam_id}.csv")
            cam_file = os.path.join(self._CROOT, sub_id, f"{sub_id}-{cam_id}.json")
            vid_file = os.path.join(self._VROOT, sub_id, f"{sub_id}-{clip_id}-{cam_id}.mkv")

            # Extract joints
            joints = extract_2d_from_c3d(jnt_file, cam_file)

            # Extract bboxes
            bboxes = pd.read_csv(box_file).values

            # Read video
            frame_counter = 0
            cap = cv2.VideoCapture(vid_file)
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_counter % 5 == 0:
                    bbox = bboxes[frame_counter].tolist()
                    bbox.append("0")

                    # Resize image and adjust keypoints
                    resized = self._resize_transform(image=frame, keypoints=joints[frame_counter], bboxes=[bbox])

                    # Normalize Keypoints
                    kps = np.asarray(resized['keypoints'], dtype = float)
                    kps = np.divide(kps, [640, 640]).reshape(34, )
                    kps = " ".join(map(lambda x: str(x), kps))

                    # Normalize Bounding Boxes
                    bbox = list(resized['bboxes'][0])
                    cat_id = bbox.pop()
                    bbox = xyxy2xywhn(bbox, [640, 640])
                    bbox = " ".join(map(lambda x: str(x), bbox))

                    # Create annotations
                    annotation = f"{cat_id} {bbox} {kps}"

                    # Save file
                    base_name = f"{sub_id}-{clip_id}-{cam_id}-{frame_counter:03d}"
                    cv2.imwrite(os.path.join(self._OIROOT, is_train, f"{base_name}.jpg"), resized['image'])
                    with open(os.path.join(self._OLROOT, is_train, f"{base_name}.txt"), 'w') as f:
                        f.write(annotation)
                        f.close()

                    # Log success
                    print(f"{is_train:>5s} {base_name:>20s}: Frame-{frame_counter:03d}.jpg Saved")
                    print(f"{is_train:>5s} {base_name:>20s}: Frame-{frame_counter:03d}.txt Saved")

                frame_counter += 1

        return []


if __name__ == "__main__":
    ds = ASPPreprocessingDS("/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/aspset/trainval")
    dl = DataLoader(ds, batch_size = 64, num_workers = 8, shuffle = False)
    for _ in enumerate(dl):
        continue

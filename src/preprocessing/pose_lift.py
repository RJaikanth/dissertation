"""
dissertation.pose_lift.py
Author: Raghhuveer Jaikanth
Date  : 12/08/2023

# Enter Description Here
"""

import sys
sys.path.append("/home/rjaikanth97/myspace/dissertation-final/dissertation")

import os
import pandas as pd
from torch.utils.data import DataLoader
from sportspose.dataset import SportsPoseDataset
from src.preprocessing.base import PreprocessingBaseClass
from src.preprocessing.sportspose_new import SportsPosePreProcessor
import numpy as np
import cv2
from src.utils.c3d import extract_2d_from_c3d, load_mocap


class ASPKeyPointPreProcessor(PreprocessingBaseClass):
    def __init__(self, root: str):
        super().__init__(root)

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
        self._OUT = "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/pose_lift/arrays"
        self._3dROOT = "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/pose_lift/3d"
        os.makedirs(self._OUT, exist_ok = True)
        os.makedirs(self._3dROOT, exist_ok = True)

    def __getitem__(self, item):
        sub_id, clip_id, is_train, cam_ids = self._splits.iloc[item, :].values
        cam_ids = ["left", "right", "mid"] if cam_ids == "all" else [cam_ids]

        # Input files
        joint_file = os.path.join(self._JROOT, sub_id, f"{sub_id}-{clip_id}.c3d")
        for cam_id in cam_ids:
            cam_file = os.path.join(self._CROOT, sub_id, f"{sub_id}-{cam_id}.json")
            vid_file = os.path.join(self._VROOT, sub_id, f"{sub_id}-{clip_id}-{cam_id}.mkv")

            # Get video dimensions
            cap = cv2.VideoCapture(vid_file)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

            joints_3d = load_mocap(joint_file)
            joints_2d = extract_2d_from_c3d(joint_file, cam_file)

            base_name = f"{sub_id}-{clip_id}-{cam_id}"
            for i, (j3d, j2d) in enumerate(zip(joints_3d.joint_positions, joints_2d)):
                name = f"{base_name}-{i:03d}"
                np.savez(os.path.join(self._OUT, name), j2d=j2d, j3d=j3d)
            print(f"{base_name:<15s}: Saved")

        return []

    def __len__(self):
        return len(self._splits)


class SportsPoseKeyPointPreProcessor(SportsPosePreProcessor):
    def __init__(self, root):
        super().__init__(root)
        self._ds = SportsPoseDataset(data_dir = self._root, sample_level="frame")
        self._OUT = "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/pose_lift/arrays"
        os.makedirs(self._OUT, exist_ok = True)
        self._frame_size = np.array([1200, 1920])

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, item):
        sample = self._ds[item]
        j2d = np.divide(sample['joints_2d']['right'][0], self._frame_size)
        j3d = sample['joints_3d']['data_points'][0]
        vid_path = sample['video']['right']['path']['right'][0]

        base_name = f"{vid_path.split('/')[-2]}"

        frame = 0
        name = f"{base_name}-%03d.npz"
        while os.path.exists(os.path.join(self._OUT, name % frame)):
            frame += 1

        print(name % frame)
        np.savez(os.path.join(self._OUT, name % frame), j2d = j2d, j3d = j3d)
        return []


if __name__ == "__main__":
    from tqdm import tqdm

    ROOT = "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/sportspose/SportsPose"
    ds = SportsPoseKeyPointPreProcessor(ROOT)
    dl = DataLoader(ds, batch_size = 256, shuffle = False, num_workers = 8)

    for _ in dl:
        continue

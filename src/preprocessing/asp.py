"""
dissertation.asp.py
Author: Raghhuveer Jaikanth
Date  : 17/07/2023

File for creating preprocessing classes for the ASP dataset.
"""
import glob
import os.path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.utils.c3d import extract_2d_from_c3d
from src.utils.config import ASPConfig
from src.utils.config import read_asp_config


class ASPPreprocessingDS(Dataset):
    ALL_CAM = ["left", "right", "mid"]

    def __init__(self, config: ASPConfig):
        super().__init__()

        # Config
        self.conf = config.dataset

        # Input paths
        self.boxes_root = os.path.join(self.conf.data_root, "boxes")
        self.camera_root = os.path.join(self.conf.data_root, "cameras")
        self.kp_root = os.path.join(self.conf.data_root, "joints_3d")
        self.videos_root = os.path.join(self.conf.data_root, "videos")
        self.splits = pd.read_csv(
            os.path.join(os.path.dirname(self.conf.data_root), "splits.csv"),
            names = ["sub_id", "clip_id", "trainval", "camera_id"]
        )

        # Output paths
        self.frames_folder = os.path.join(os.path.dirname(self.conf.data_root), "final_ds", "frames")
        self.annotation_folder = os.path.join(os.path.dirname(self.conf.data_root), "final_ds", "annotations")

        # video files
        self.videos = glob.glob(f"{self.videos_root}/**/*.{self.conf.v_ext}")

        # Create output dirs
        self.out_dir = {
            "train_images"     : os.path.join(self.frames_folder, "train"),
            "train_annotations": os.path.join(self.annotation_folder, "train"),
            "val_images"       : os.path.join(self.frames_folder, "val"),
            "val_annotations"  : os.path.join(self.annotation_folder, "val"),
        }
        for v in self.out_dir.values():
            os.makedirs(v, exist_ok = True)

        # Scale
        self.w_scale = self.conf.i_size[0] / 3840.
        self.h_scale = self.conf.i_size[1] / 2160.

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        subject_id, clip_id, camera_id = os.path.basename(self.videos[item][:-4]).split("-")

        # Get paths of all required inputs
        video_path = self.videos[item]
        camera_config = os.path.join(self.camera_root, subject_id, f"{subject_id}-{camera_id}.json")
        keypoints = os.path.join(self.kp_root, subject_id, f"{subject_id}-{clip_id}.c3d")
        bounding_boxes = pd.read_csv(os.path.join(self.boxes_root, subject_id, f"{subject_id}-{clip_id}-{camera_id}.csv"))

        # Check if row is train
        is_train = self.splits[
                       (self.splits['sub_id'] == subject_id) &
                       (self.splits['clip_id'] == int(clip_id))
                       ]['trainval'].values[0]

        # Extract joints in 2D
        joints_2d = extract_2d_from_c3d(keypoints, camera_config)

        # extract frames
        frame_counter = 0
        video_cap = cv2.VideoCapture(video_path)
        while video_cap.isOpened():
            ret, frame = video_cap.read()

            # Break
            if not ret:
                break

            # Extract based on requirement
            if frame_counter % self.conf.v_frame_frequency == 0:
                try:
                    frame = cv2.resize(frame, self.conf.i_size)

                    # Scale and normalize keypoints
                    keypoints = joints_2d[frame_counter]
                    keypoints = np.multiply(keypoints, [self.w_scale, self.h_scale])
                    keypoints = np.divide(keypoints, self.conf.i_size).reshape(34, )

                    # Scale, add offset, and normalize bbox
                    bbox = bounding_boxes.iloc[frame_counter, :].values.reshape((2, 2)) * [self.w_scale, self.h_scale]
                    bbox = bbox + np.asarray([[-10, -10], [10, 10]])
                    bbox = convert_xywh(bbox, self.conf.i_size)

                    # Create annotations
                    annotations = [0]
                    annotations.extend(bbox)
                    annotations.extend(keypoints)
                    annotations = " ".join(map(lambda x: str(x), annotations))

                    # Save frame and annotations
                    base_name = f"{subject_id}-{clip_id}-{camera_id}"
                    cv2.imwrite(os.path.join(self.out_dir[f"{is_train}_images"], f"{base_name}-Frame-{frame_counter:03d}.{self.conf.i_ext}"), frame)
                    with open(os.path.join(self.out_dir[f"{is_train}_annotations"], f"{base_name}-Frame-{frame_counter:03d}.txt"), 'w') as f:
                        f.write(annotations)
                        f.close()
                    print(f"{is_train}/{base_name:15s}: Frame-{frame_counter:03d}.{self.conf.i_ext} Saved")
                    print(f"{is_train}/{base_name:15s}: Frame-{frame_counter:03d}.txt Saved")

                except Exception as e:
                    base_name = f"{subject_id}-{clip_id}-{camera_id}"
                    print("Error: ", f"{is_train}/{base_name}-Frame-{frame_counter:03d}", e)

            frame_counter += 1

        return []

    @staticmethod
    def _get_attrs(item):
        base_name = os.path.basename(item)[:-4]
        (subject_id, clip_id, camera_id) = base_name.split("-")
        return subject_id, clip_id, camera_id


def convert_xywh(bbox, img_size):
    x = int((bbox[0, 0] + bbox[1, 0]) / 2) / img_size[0]
    y = int((bbox[0, 1] + bbox[1, 1]) / 2) / img_size[1]
    w = int(bbox[1, 0] - bbox[0, 0]) / img_size[0]
    h = int(bbox[1, 1] - bbox[0, 1]) / img_size[1]
    return [x, y, w, h]


def preprocess_asp(config_path: str):
    config = read_asp_config(config_path)
    ds = ASPPreprocessingDS(config)
    dl = DataLoader(ds, **config.dataloader.to_dict())
    for i, _ in enumerate(dl):
        continue


if __name__ == "__main__":
    CONFIG_PATH = "/home/rjaikanth97/myspace/dissertation-final/dissertation/configs/asp_prep.yaml"
    preprocess_asp(CONFIG_PATH)

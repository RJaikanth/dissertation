"""
dissertation.sportspose_new.py
Author: Raghhuveer Jaikanth
Date  : 05/08/2023

# Enter Description Here
"""
import glob
import os.path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sportspose.dataset import SportsPoseDataset

from src.preprocessing.base import PreprocessingBaseClass


class SportsPosePreProcessor(PreprocessingBaseClass):
    def __init__(self, root: str):
        super().__init__(root)

        self._train_list, self._valid_list = self._get_trainval()
        self._ds = SportsPoseDataset(data_dir=self._root, sample_level="frame")

        # Video meta
        self.curr_video = None
        self.frame_counter = 0

        # Output dirs
        self._OIROOT = "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/combined-ds/images"
        self._OLROOT = "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/combined-ds/labels"

    def _get_trainval(self):
        ind_videos = glob.glob(
            os.path.join(self._root, "videos/indoors/**/*.avi"), recursive=True
        )
        out_videos = glob.glob(
            os.path.join(self._root, "videos/outdoors/**/*.avi"), recursive=True
        )

        ind_train_size = 443
        ind_train_videos = ind_videos[:ind_train_size]
        ind_valid_videos = ind_videos[ind_train_size:]

        out_train_size = 80
        out_train_videos = out_videos[:out_train_size]
        out_valid_videos = out_videos[out_train_size:]

        train_videos = ind_train_videos
        train_videos.extend(out_train_videos)

        valid_videos = ind_valid_videos
        valid_videos.extend(out_valid_videos)

        return train_videos, valid_videos

    def __len__(self):
        return len(self._ds)

    def _get_bbox(self, joint, offset=50):
        min_x = int(min(joint, key=lambda x: x[0])[0])
        min_y = int(min(joint, key=lambda x: x[1])[1])
        max_x = int(max(joint, key=lambda x: x[0])[0])
        max_y = int(max(joint, key=lambda x: x[1])[1])

        xyxy = [
            max(0, min_x - offset),
            max(0, min_y - offset),
            min(max_x + offset, 1216),
            min(1936, max_y + offset),
        ]

        return xyxy

    def __getitem__(self, item):
        sample = self._ds[item]

        # Set frame count
        vid_path = sample["video"]["right"]["path"]["right"][0]
        is_train = "train" if vid_path in self._train_list else "val"

        if self.curr_video != vid_path:
            self.curr_video = vid_path
            # print(f"\n\nNew video ({vid_path}) after {self.frame_counter} frames")
            self.frame_counter = 0

        if (self.frame_counter % 9 == 0) or (self.frame_counter == 269):
            try:
                self._extract_frame(sample, is_train)
            except Exception as e:
                pass
        self.frame_counter += 1

        return []

    def _extract_frame(self, sample, is_train):
        # Extract data
        frame = sample["video"]["image"]["right"][0]
        joints = np.asarray(sample["joints_2d"]["right"][0])
        bbox = self._get_bbox(joints, offset=100)

        # Transform
        crop = self._add_offset(bbox, frame.shape, 200)
        bbox.append(0)
        transforms = [
            A.Crop(crop[0], crop[1], crop[2], crop[3], p=1, always_apply=True),
            A.LongestMaxSize(max_size=640, interpolation=1),
            A.PadIfNeeded(min_height=640, min_width=640, border_mode=1),
        ]
        transforms = A.Compose(
            transforms,
            bbox_params=A.BboxParams("pascal_voc"),
            keypoint_params=A.KeypointParams("xy"),
        )
        transformed = transforms(image=frame, keypoints=joints, bboxes=[bbox])

        # Get new image
        crop_frame = transformed["image"]

        # Pose keypoints
        joints = np.asarray(transformed["keypoints"])

        # Bounding Box
        bbox = list(transformed["bboxes"][0])
        cat_id = bbox.pop()

        # Create annotation
        annotation = f"{cat_id} " + self._create_annotation(
            joints, bbox, crop_frame.shape
        )

        # Save files
        base_name = f"{self.curr_video.split('/')[-2]}"
        cv2.imwrite(
            os.path.join(
                self._OIROOT, is_train, f"{base_name}-{self.frame_counter:03d}.jpg"
            ),
            transformed["image"],
        )
        with open(
            os.path.join(
                self._OLROOT, is_train, f"{base_name}-{self.frame_counter:03d}.txt"
            ),
            "w",
        ) as f:
            f.write(annotation)
            f.close()

        # Log success
        print(
            f"{is_train:>5s} {base_name:>20s}: Frame-{self.frame_counter:03d}.jpg Saved"
        )
        print(
            f"{is_train:>5s} {base_name:>20s}: Frame-{self.frame_counter:03d}.txt Saved"
        )


if __name__ == "__main__":
    ROOT = (
        "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/SportsPose"
    )
    ds = SportsPosePreProcessor(ROOT)
    for _ in enumerate(ds):
        continue

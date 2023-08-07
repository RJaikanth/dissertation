"""
dissertation.soccerkicks.py
Author: Raghhuveer Jaikanth
Date  : 06/08/2023

# Enter Description Here
"""
import glob

import pandas as pd
from torch.utils.data import Dataset
from src.preprocessing.base import PreprocessingBaseClass


class SoccerKicksPreprocessor(PreprocessingBaseClass):
    def __init__(self, root: str):
        super().__init__(root)

        # Input roots
        self._video_root = f"{self._root}/VideoClips"
        self._annotation_root = f"{self._root}/Rendered"

        # Video files
        self._video_files = list(map(lambda x: x[:-4].split("/")[-1], glob.glob(f"{self._video_root}/*.mp4")))

    def __getitem__(self, item):
        # Video info
        video_title = self._video_files[item]
        video_path = f"{self._video_root}/{video_title}"
        video_type = video_title.split("_")[-1].capitalize()
        print(video_path)

        # Annotation info
        annotation_path = f"{self._annotation_root}/{video_type}/{video_title}/AlphaPose_output/alphapose-results.json"
        annotations = pd.read_json(annotation_path)
        annotations = annotations[["category_id", "keypoints", "box"]]
        print(annotations)

        return []

    def __len__(self):
        return len(self._video_files)


if __name__ == "__main__":
    ROOT = "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/soccerkicks"
    ds = SoccerKicksPreprocessor(ROOT)
    next(iter(ds))

"""
dissertation.scratch.py
Author: Raghhuveer Jaikanth
Date  : 23/07/2023

# Enter Description Here
"""
import os.path

from torch.utils.data import Dataset
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def xyxy2xywhn(bbox, img_size):
    print(bbox)
    x = int((bbox[0, 0] + bbox[1, 0]) / 2) / img_size[0]
    y = int((bbox[0, 1] + bbox[1, 1]) / 2) / img_size[1]
    w = int(bbox[1, 0] - bbox[0, 0]) / img_size[0]
    h = int(bbox[0, 1] - bbox[1, 1]) / img_size[1]
    return [x, y, w, h]


def xywhn2xyxy(bbox, img_size):
    x1 = int(img_size[1] * (bbox[0] - bbox[2] / 2))
    y1 = int(img_size[0] * (bbox[1] + bbox[3] / 2))
    x2 = int(img_size[1] * (bbox[0] + bbox[2] / 2))
    y2 = int(img_size[0] * (bbox[1] - bbox[3] / 2))
    return [x1, y1, x2, y2]


def sampling_function(measurements, index):
    indices = [k for k in measurements.keys() if k % 5 == 0]
    choice = np.random.choice(indices)
    return measurements[choice]


def get_bbox(joint, img_size, offset = 100):
    min_x = min(joint, key = lambda x: x[0])[0]
    min_y = min(joint, key = lambda x: x[1])[1]
    max_x = max(joint, key = lambda x: x[0])[0]
    max_y = max(joint, key = lambda x: x[1])[1]

    tl = [min_x - offset, max_y + offset]
    br = [max_x + offset, min_y - offset]
    xyxy = np.asarray([tl, br], dtype=int)

    return np.asarray(xyxy2xywhn(xyxy, img_size))


class SPPreProcessingDS(Dataset):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds
        self.frame_size = set()

    def __getitem__(self, item):
        sample = self.ds[item]
        joints = sample['joints_2d']['right']
        frames = sample['video']['image']['right']
        vid_name = sample['video']['right']['path']['right'][0]

        num_frames = frames.shape[0]

        for frame_count in range(num_frames):
            if frame_count % 5 == 0:
                frame = frames[0]
                self.frame_size.add(frame.shape)
                # joint = joints[0]
                #
                # plt.imshow(frame)
                # plt.scatter(joint[:, 0], joint[:, 1], s=1)
                # plt.show()

            else:
                continue

        # print(joints.shape, frames.shape, vid_name)

        return

    def __len__(self):
        return len(self.ds)


if __name__ == "__main__":
    from sportspose.dataset import SportsPoseDataset
    from glob import glob

    # Outputs joints and frames for each video
    ds = SportsPoseDataset(data_dir = "./dataset/SportsPose", sample_level = "video", sample_method = sampling_function)

    for i, sample in enumerate(ds):
        frame = sample['video']['image']['right'][0]
        joint = sample['joints_2d']['right'][0]

        # transformed = transform(image=frame, keypoints=joint)
        # joint = np.asarray(transformed['keypoints'])

        bbox = get_bbox(joint, frame.shape)
        print(bbox)
        bbox = np.asarray(xywhn2xyxy(bbox, frame.shape)).reshape((2, 2))
        print(bbox)
        # print(bbox)

        plt.imshow(frame, origin = 'lower')
        plt.scatter(bbox[:, 0], bbox[:, 1], s = 20, c = 'blue')
        plt.scatter(joint[:, 0], joint[:, 1], s = 5, c = 'red')
        plt.show()
        break
        continue
    # print(frames)

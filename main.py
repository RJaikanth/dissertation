"""
dissertation.main.py
Author: Raghhuveer Jaikanth
Date  : 17/08/2023

# Enter Description Here
"""

import argparse
import os.path

import PIL.Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import torch

from src.poselift.cnn import CNNModel, ResidualBlock
from sportspose.plot_skeleton_3d import plot_skeleton_3d


def args():
    parser = argparse.ArgumentParser(
        prog="Dissertation Final Script",
        description = "Score a freekick/penalty",
    )
    parser.add_argument("--video_file", required = True, help = "path to video_file")
    parser.add_argument("--hpe", required = False, default = "yolov8n", choices = ["yolov8n", "yolov8x"])
    parser.add_argument("--lift", required = False, default = "cnn", choices = ["cnn", "ffn"])
    parser.add_argument("--visualize", required = False, default = True)

    return parser.parse_args()


def init_models(args):
    return {
        "hpe" : YOLO("./models/pose/yolov8n-pose.pt"),
        "lift": torch.load(os.path.join("results", "lifting", f"{main_args.lift}", "best.pt"),  map_location=torch.device('cpu'))
    }


def process_video(video_file, models):
    cap = cv2.VideoCapture(video_file)
    kps_2d = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            result = models['hpe'](frame)[0]
            kps_2d.append(result.keypoints.xyn[0])
        else:
            break
    cap.release()

    kps_2d = torch.stack(kps_2d)
    kps_3d = models['lift'](kps_2d).detach().numpy()
    score(kps_3d)


def score(kps):
    from tslearn.metrics import dtw_path_from_metric

    ref = np.load("./results/freekick_ref.np.npy")
    ref = ref.reshape(len(ref), -1)
    kps = np.load("./results/query.npy")
    kps = kps.reshape(len(kps), -1)
    print(f"Query DTW distance from reference video: {dtw_path_from_metric(ref, kps, metric='cosine')[-1]}")


def write_video():
    images = os.listdir("./plots/output/")
    size = list(cv2.imread(f"./plots/output/{images[0]}").shape)
    del size[2]
    size.reverse()

    writer = cv2.VideoWriter("./check.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 24, size)

    for img in images:
        writer.write(cv2.imread(f"plots/output/{img}"))
    writer.release()


if __name__ == "__main__":
    main_args = args()
    models = init_models(main_args)
    video_path = "13_freekick2 (1).mp4"
    process_video(video_path, models)
    write_video()

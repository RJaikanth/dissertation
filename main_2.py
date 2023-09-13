"""
dissertation.main.py
Author: Raghhuveer Jaikanth
Date  : 17/08/2023

# Enter Description Here
"""
import warnings

warnings.filterwarnings("ignore")

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

import matplotlib

matplotlib.use('TkAgg')


def args():
    parser = argparse.ArgumentParser(
        prog = "Dissertation Final Script",
        description = "Score a freekick/penalty",
    )
    parser.add_argument("--query", required = True, help = "path to query video")
    parser.add_argument("--reference", required = True, help = "path to reference video")
    parser.add_argument("--hpe", required = False, default = "yolov8n", choices = ["yolov8n", "yolov8x"])
    parser.add_argument("--lift", required = False, default = "cnn", choices = ["cnn", "ffn"])
    parser.add_argument("--visualize", required = False, default = True)

    return parser.parse_args()


def init_models(args):
    return {
        "hpe" : YOLO(f"./results/{args.hpe}/best.pt"),
        "lift": torch.load(os.path.join("results", "lifting", f"{args.lift}", "best.pt"), map_location = torch.device('cpu'))
    }


def process_video(video_file, models):
    cap = cv2.VideoCapture(video_file)
    kps_2d = []

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

    return kps_3d


def score(query, ref):
    from tslearn.metrics import dtw_path_from_metric

    ref = ref.reshape(len(ref), -1)
    query = query.reshape(len(query), -1)
    print(f"Query DTW distance from reference video: {dtw_path_from_metric(ref, query, metric = 'cosine')[-1]}")


def save_figs(query_kps, ref_kps):
    for i, (r, q) in enumerate(zip(r_kps, q_kps)):
        fig = plt.figure(figsize=(7.5, 7.5))
        ax = fig.add_subplot(1, 1, 1, projection = '3d')

        plot_skeleton_3d(r, ax = ax, kt_color = "green", joint_color = "green", label = "Reference", marker = 'o')
        plot_skeleton_3d(q, ax = ax, kt_color = "red", joint_color = "red", label = "Query", marker = 'x')

        plt.legend()
        plt.savefig(f"./plots_new/output/frame-{i:03d}.jpg", dpi = 100)
        plt.close(fig)


def visualize():
    images = sorted(os.listdir("./plots_new/output/"))
    for i, f in enumerate(images):
        im = cv2.imread(f"./plots_new/output/{f}")
        cv2.imshow("Final Comparison Video", im)
        if i == len(images) - 1:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.waitKey(100)


if __name__ == "__main__":
    # cmd line args
    main_args = args()

    # Init variables
    reference = main_args.reference
    query = main_args.query
    models = init_models(main_args)

    # Extract keypoints
    r_kps = process_video(reference, models)
    q_kps = process_video(query, models)

    # Visualize
    if main_args.visualize:
        visualize()

    score(q_kps, r_kps)

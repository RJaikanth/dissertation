import os.path

import matplotlib.pyplot as plt
import torch
from sportspose.dataset import SportsPoseDataset
from sportspose.plot_skeleton_3d import plot_skeleton_3d
import numpy as np
from ultralytics import YOLO
from src.poselift.cnn import CNNModel, ResidualBlock
from src.poselift.ffn import FFNModel


def sample_fn(measurements, index):
    max_val = len(measurements)
    choices = range(150, max_val)
    # Uniform random sample
    choice = np.random.choice(choices)
    # Return the frame index
    return measurements[choice]


if __name__ == "__main__":
    # activities = ["soccer", "volleyball", "jump", "throw_baseball", "tennis"]

    lift_ = "ffn"
    lift = torch.load(os.path.join("results", "lifting", lift_, "best.pt"), map_location = torch.device('cpu'))
    hpe = YOLO("results/yolov8n/runs/pose/YOLOv8N SGD (40-50 epochs)/weights/best.pt")

    activity = "soccer"

    ds = SportsPoseDataset(
        data_dir="./dataset/sportspose",
        sample_level="video",
        whitelist = {"metadata": {"activity": activity}},
        sample_method = sample_fn
    )
    sample = next(iter(ds))
    frame = sample["video"]["image"]["right"][0]
    joints2D = sample["joints_2d"]["right"][0]
    joints3D = sample["joints_3d"]["data_points"][0]

    # Predictions
    with torch.no_grad():
        result2D = hpe(frame)[0].keypoints
        result3D = lift(result2D.xyn).detach().numpy()[0]

    # Set axes
    fig = plt.figure(figsize = (10, 5))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection = "3d")

    # 2D
    ax2d.imshow(frame)
    ax2d.scatter(joints2D[:, 0], joints2D[:, 1], marker='o', c='green', s = 15, label='Ground Truth')
    ax2d.scatter(result2D.xy[0][:, 0], result2D.xy[0][:, 1], marker='x', c='red', s=20, label='YOLOv8X Prediction')
    ax2d.legend()
    ax2d.axis('off')

    # 3D
    if lift_ == "ffn":
        result3D = result3D.reshape(17, 3)
    plot_skeleton_3d(joints3D, ax = ax3d, kt_color = "green", joint_color = "green", label = "Ground Truth", marker = 'o')
    plot_skeleton_3d(result3D, ax = ax3d, kt_color = "red", joint_color = "red", label = f"{lift_.upper()} Prediction", marker = 'x')
    ax3d.legend()

    plt.suptitle(f"YOLOv8-N + {lift_.upper()} Lifting + {activity.capitalize()}")
    plt.savefig(f"plots/yolov8n/yolov8n-{lift_}-{activity}.png", dpi=100)
    plt.show()

"""
dissertation.yolov8.py
Author: Raghhuveer Jaikanth
Date  : 18/07/2023

# Model functions
"""

from ultralytics import YOLO
from torchinfo import summary

def load_yolov8(model_path: str):
    model = YOLO(model_path).model
    summary(model)


if __name__ == "__main__":
    load_yolov8("/home/rjaikanth97/myspace/dissertation-final/dissertation/models/pose/yolov8n-pose.pt")

"""
dissertation.train_scratch.py
Author: Raghhuveer Jaikanth
Date  : 24/07/2023

# Enter Description Here
"""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./models/pose/yolov8n-pose.pt")

    model.train(data="/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/aspset/combined-ds.yaml", epochs=1, imgsz=640)

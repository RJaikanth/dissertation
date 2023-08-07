"""
dissertation.scratch.py
Author: Raghhuveer Jaikanth
Date  : 23/07/2023

# Enter Description Here
"""
import pandas as pd
import re
from ultralytics import YOLO
from ast import literal_eval

import numpy as np

if __name__ == "__main__":
    # model = YOLO("models/pose/yolov8n-pose.pt")
    # model.train(data="combined.yaml")

    # from dtaidistance import dtw_ndim
    # from dtaidistance import dtw_ndim_visualisation as dtwvis
    #
    # cols = ['Nose']#, 'Neck', 'RShoulder',
    #    # 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'MidHip', 'RHip',
    #    # 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'LBigToe', 'LSmallToe',
    #    # 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel', 'Head']
    # convertors = {col: lambda x: np.asarray(literal_eval(re.sub(r" +", ",", x))) for col in cols}
    #
    # s1 = pd.read_csv(seriesBase, usecols = cols, converters = convertors).to_numpy()
    # s2 = pd.read_csv(seriesToCompare, usecols = cols, converters = convertors).to_numpy()
    # path = dtw_ndim.warping_path(s1, s2)
    # dtwvis.plot_warping(s1, s2, path, filename = "warp.png")

    from tslearn import metrics

    seriesBase = "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/socckerkicks-mini/13_freekick.csv"
    seriesToCompare = "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/socckerkicks-mini/AlphaPose_2D_kps.csv"

    cols = ["Nose"]
    converter = {"Nose": lambda x: np.asarray(literal_eval(re.sub(r" +", ",", x)))}
    s1 = pd.read_csv(seriesBase, usecols = cols)
    s1["Nose"] = s1["Nose"].str.replace(r" +", ",", regex = True).apply(literal_eval) #apply(lambda x: re.sub(r" +", ",", x.strip()).split(","))
    print(np.asarray(s1["Nose"].to_list()).shape)
    s1 = np.asarray(s1["Nose"].to_list())

    s2 = pd.read_csv(seriesToCompare, usecols = cols)
    s2["Nose"] = s2["Nose"].str.replace(r" +", ",", regex = True).apply(literal_eval) #apply(lambda x: re.sub(r" +", ",", x.strip()).split(","))
    print(np.asarray(s2["Nose"].to_list()).shape)
    s2 = np.asarray(s2["Nose"].to_list())

    from fastdtw import fastdtw

    from scipy.spatial.distance import euclidean, cosine
    distance, path = fastdtw(s1, s2, dist = cosine)
    print(distance)

    sim = metrics.dtw(s1, s2)
    print(sim)

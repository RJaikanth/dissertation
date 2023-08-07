"""
dissertation.preprocess_and_combine.py
Author: Raghhuveer Jaikanth
Date  : 05/08/2023

# Enter Description Here
"""
import sys
sys.path.append("/home/rjaikanth97/myspace/dissertation-final/dissertation/")

from src.preprocessing.asp_new import ASPDataPreProcessor
from src.preprocessing.sportspose_new import SportsPosePreProcessor
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # Logging
    import logging

    logging.basicConfig(level = logging.INFO, format = '%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler('prep_and_combine.log', 'w'))
    print = logger.info

    # ASP Dataset
    ROOT = "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/aspset/trainval"
    ds = ASPDataPreProcessor(ROOT)
    dl = DataLoader(ds, batch_size = 64, num_workers = 8, shuffle = False)
    for _ in enumerate(dl):
        continue

    # Sports Pose Dataset
    ROOT = "/home/rjaikanth97/myspace/dissertation-final/dissertation/dataset/SportsPose"
    ds = SportsPosePreProcessor(ROOT)
    for _ in enumerate(ds):
        continue

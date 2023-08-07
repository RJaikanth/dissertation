import glob
import os
import shutil

SP_DIR = "/dataset/combined-ds/sportspose"
ASP_DIR = "/dataset/combined-ds/aspset"
TARGET_DIR = "/dataset/combined-ds"

PATHS = ["images/train", "images/val", "labels/train", "labels/val"]


for dir in [ASP_DIR, SP_DIR]:
    for tar in PATHS:
        for file in glob.glob(f"{dir}/{tar}/*"):
            shutil.move(file, f"{TARGET_DIR}/{tar}/{os.path.basename(file)}")
        # for file in glob.glob(f"{dir}/{}")
        # lob.glob(f"{dir}/{tar}/*"):
            # shutil.move(file, f"{TARGET_DIR}/", f"{TARGET_DIR}/{tar}/{os.path.basename(file)}")
        


# for image in glob.glob("images/train/*.jpg"):
#     shutil.move(image, )


# for i in glob.glob('*.jpg'):
#   shutil.move(i, 'new_dir/' + i)

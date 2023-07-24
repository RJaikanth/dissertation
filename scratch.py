"""
dissertation.scratch.py
Author: Raghhuveer Jaikanth
Date  : 23/07/2023

# Enter Description Here
"""


def xywhn2xyxy(bbox, img_size):
    print(img_size)
    x1 = int(img_size[1] * (bbox[0] - bbox[2] / 2))
    y1 = int(img_size[0] * (bbox[1] - bbox[3] / 2))
    x2 = int(img_size[1] * (bbox[0] + bbox[2] / 2))
    y2 = int(img_size[0] * (bbox[1] + bbox[3] / 2))
    return [x1, y1, x2, y2]


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib import patches

    image = cv2.imread("/home/rjaikanth97/myspace/dissertation-final/dissertation/ASPset/final_ds/frames/train/eb61-0018-left-Frame-000.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open("/home/rjaikanth97/myspace/dissertation-final/dissertation/ASPset/final_ds/annotations/train/eb61-0018-left-Frame-000.txt") as f:
        annotations = list(map(lambda x: float(x), f.read().split(" ")))

    bbox = annotations[1:5]
    bbox = xywhn2xyxy(bbox, image.shape)

    keypoints_x = annotations[5::2]
    keypoints_y = annotations[6::2]
    # print(keypoints_x)
    # print(keypoints_y)
    # print(annotations[5:])

    fig, ax = plt.subplots()
    ax.imshow(image)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    for x, y in zip(keypoints_x, keypoints_y):
        ax.scatter(int(x*image.shape[1]), int(y*image.shape[0]), marker='o', color='red', s=1)
    plt.show()

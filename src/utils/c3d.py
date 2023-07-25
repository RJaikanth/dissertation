"""
dissertation.c3d.py
Author: Raghhuveer Jaikanth
Date  : 25/07/2023

# Enter Description Here
"""

"""
dissertation.c3d.py
Author: Raghhuveer Jaikanth
Date  : 17/07/2023

Utility functions for c3d files
"""
import json
import os
from dataclasses import dataclass

import ezc3d
import numpy as np


@dataclass
class Mocap:
    joint_positions: np.ndarray
    skeleton_name: str
    sample_rate: float


def load_mocap(filename: str):
    c3d = ezc3d.c3d(os.fspath(filename))
    sample_rate = c3d['parameters']['POINT']['RATE']['value'][0]
    skeleton_name = c3d['parameters']['POINT']['SKEL']['value'][0]
    joints = c3d['data']['points'].transpose(2, 1, 0)[..., :3].astype(np.float32)
    return Mocap(joints, skeleton_name, sample_rate)


def get_projection_matrix(filename: str):
    with open(filename, 'r') as f:
        cam_json = json.load(f)

    int_mat = np.asarray(cam_json["intrinsic_matrix"]).reshape([3, 4])
    ext_mat = np.asarray(cam_json["extrinsic_matrix"]).reshape([4, 4])
    return int_mat @ ext_mat


def to_homogeneous(points):
    """Cartesian to homogeneous"""
    return np.concatenate([points, np.ones_like(points[..., -1:])], -1)


def ensure_homogeneous(points, d):
    if points.shape[-1] == d + 1:
        return points
    assert points.shape[-1] == d
    return to_homogeneous(points)


def to_cartesian(points):
    return points[..., :-1] / points[..., -1:]


def extract_2d_from_c3d(c3d_file: str, camera_file: str):
    camera_matrix = get_projection_matrix(camera_file)
    c3d_obj = load_mocap(c3d_file)

    return to_cartesian(ensure_homogeneous(c3d_obj.joint_positions, 3) @ camera_matrix.T)

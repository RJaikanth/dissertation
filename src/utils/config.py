"""
dissertation.config.py
Author: Raghhuveer Jaikanth
Date  : 17/07/2023

Functions to read YAML files with dot notation
"""

import typing as T
from dataclasses import dataclass

import yaml


@dataclass
class ASPDataSetConfig:
    data_root: str
    v_ext: str = "mkv"
    i_ext: str = "jpg"
    v_frame_frequency: int = 5
    i_size: T.Tuple[int, int] = (1280, 736)

    @classmethod
    def from_dict(cls: T.Type["ASPDataSetConfig"], conf: dict):
        return cls(**conf)


@dataclass
class ASPDataLoaderConfig:
    batch_size: int
    shuffle: bool
    drop_last: bool
    num_workers: int

    @classmethod
    def from_dict(cls: T.Type["ASPDataLoaderConfig"], conf: dict):
        return cls(**conf)

    def to_dict(self) -> dict:
        return {
                "batch_size": self.batch_size,
                "shuffle": self.shuffle,
                "drop_last": self.drop_last,
                "num_workers": self.num_workers
        }


@dataclass
class ASPConfig:
    dataset: ASPDataSetConfig
    dataloader: ASPDataLoaderConfig

    @classmethod
    def from_dict(cls: T.Type["ASPConfig"], conf: dict):
        return cls(
            dataset = ASPDataSetConfig.from_dict(conf["dataset"]),
            dataloader = ASPDataLoaderConfig.from_dict(conf["dataloader"]),
        )


def read_asp_config(file_path: str) -> ASPConfig:
    with open(file_path, 'r') as f:
        config = yaml.full_load(f)
        f.close()

    return ASPConfig.from_dict(config)

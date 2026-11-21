import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class KITTIDataset(GradSLAMDataset):
    """KITTI odometry RGB + SGBM depth adapter for SplaTAM RGB-D mode.

    Expects:
      basedir/sequence/image_2/*.png
      basedir/sequence/depth_sgbm/*_depth.npy, float32 meters
      dataset_root/poses/{sequence}.txt in KITTI 12-column c2w format
    """

    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 376,
        desired_width: Optional[int] = 1241,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.sequence = str(sequence).zfill(2)
        self.input_folder = os.path.join(basedir, self.sequence)
        dataset_root = os.path.dirname(os.path.dirname(self.input_folder))
        self.depth_folder = config_dict.get(
            "depth_folder", os.path.join(self.input_folder, "depth_sgbm")
        )
        self.pose_path = config_dict.get(
            "pose_path", os.path.join(dataset_root, "poses", f"{self.sequence}.txt")
        )
        self.frame_ids = []
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        image_dir = os.path.join(self.input_folder, "image_2")
        color_candidates = natsorted(str(p) for p in Path(image_dir).glob("*.png"))
        color_paths, depth_paths, frame_ids = [], [], []
        for color_path in color_candidates:
            frame_id = Path(color_path).stem
            depth_path = os.path.join(self.depth_folder, f"{frame_id}_depth.npy")
            if os.path.isfile(depth_path):
                color_paths.append(color_path)
                depth_paths.append(depth_path)
                frame_ids.append(frame_id)
        if not color_paths:
            raise FileNotFoundError(
                f"No KITTI RGB/depth pairs found under {image_dir} and {self.depth_folder}"
            )
        self.frame_ids = frame_ids
        embedding_paths = None
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        if not os.path.isfile(self.pose_path):
            raise FileNotFoundError(f"KITTI pose file not found: {self.pose_path}")
        raw = np.loadtxt(self.pose_path, dtype=np.float64).reshape(-1, 3, 4)
        poses = []
        for frame_id in self.frame_ids:
            idx = int(frame_id)
            if idx >= raw.shape[0]:
                raise IndexError(
                    f"Pose index {idx} out of range for {self.pose_path} with {raw.shape[0]} poses"
                )
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :4] = raw[idx].astype(np.float32)
            poses.append(torch.from_numpy(c2w).float())
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)

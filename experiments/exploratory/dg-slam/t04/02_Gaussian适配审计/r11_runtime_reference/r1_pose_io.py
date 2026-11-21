"""Explicit boundary: DROID/lietorch uses [tx,ty,tz,qx,qy,qz,qw]."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

def matrix_to_droid_pose(matrix):
    value=matrix.detach().cpu().numpy() if isinstance(matrix,torch.Tensor) else np.asarray(matrix)
    assert value.shape==(4,4) and np.isfinite(value).all()
    quaternion_xyzw=Rotation.from_matrix(value[:3,:3]).as_quat()
    return torch.from_numpy(np.concatenate([value[:3,3],quaternion_xyzw])).float()

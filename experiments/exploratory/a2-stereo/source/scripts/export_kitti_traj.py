import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from utils.slam_external import build_rotation


def write_kitti_traj(params_path, out_path):
    params = np.load(params_path, allow_pickle=True)
    rots = torch.from_numpy(params["cam_unnorm_rots"]).float().cuda()
    trans = torch.from_numpy(params["cam_trans"]).float().cuda()
    first_w2c = torch.from_numpy(params["w2c"]).float().cuda()
    n = rots.shape[-1]
    rows = []
    for idx in range(n):
        if idx == 0:
            w2c = first_w2c.clone()
        else:
            q = F.normalize(rots[..., idx])
            w2c = torch.eye(4, device="cuda", dtype=torch.float32)
            w2c[:3, :3] = build_rotation(q)
            w2c[:3, 3] = trans[..., idx].reshape(3)
        c2w = torch.linalg.inv(w2c).detach().cpu().numpy()[:3, :4]
        rows.append(c2w.reshape(-1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, np.stack(rows), fmt="%.9e")
    print(f"wrote {len(rows)} poses to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_kitti_traj(args.params, args.out)

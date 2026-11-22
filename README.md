# Semantic Dynamic 3DGS SLAM

Research code, configurations, and evaluation records for a stereo Gaussian
SLAM final-year project. The studies cover map-assisted pose correction,
historical keyframe replay, stereo depth confidence, appearance refinement,
and semantic and dynamic observation filtering.

## Get started

Clone the repository and install the CUDA 12.8 builds of PyTorch and
TorchVision in a Python 3.12 environment.

```bash
git clone https://github.com/CooperAndy/Semantic-Dynamic-3DGS-SLAM.git
cd Semantic-Dynamic-3DGS-SLAM
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

The main GPU experiments used Ubuntu 22.04.5 LTS, Python 3.12.3,
CUDA 12.8, and an NVIDIA GeForce RTX 5090. Mask2Former preprocessing
used Transformers 4.57.6. Additional dependencies vary by experiment.
Configure the relevant ORB-SLAM3, RAFT-Stereo, Mask2Former, and CUDA
rasterizer components before running the corresponding stages.

## Download and prepare the data

| Dataset | Files needed for the experiments |
| --- | --- |
| [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | Rectified colour stereo images and calibration. Use the official reference poses for trajectory evaluation and the Velodyne scans for LiDAR-based evaluation. |
| [SemanticKITTI](https://www.semantic-kitti.org/dataset.html) | Point labels paired by sequence and frame with the KITTI Odometry Velodyne scans. |
| [Virtual KITTI 2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/) | RGB, depth, and the annotations specified by the selected semantic or motion study. |
| [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/download.php) | Perspective images, calibration, and semantic labels for the selected fixed-map study. |

For KITTI Odometry, set `KITTI_ROOT` to the directory that contains
`sequences/` after extracting the colour images and calibration.
For example, validate the left and right projection matrices for
sequence 00 before generating depth or poses.

```bash
KITTI_ROOT=/path/to/kitti/dataset
python preprocessing/kitti/scripts/kitti_calib_parser.py \
  "$KITTI_ROOT/sequences/00/calib.txt"
```

Obtain dataset archives and pretrained checkpoints from their providers.
Use the saved split and input manifests to prepare only the windows
required by the experiment you want to run.

## Reproduce results

For the S23 E2/E3 matrix, start with the [runnable S23 guide](experiments/online-semantic-dynamic/REPRODUCE.md). It covers the FP64 CUDA build, supplying causal ORB poses, regenerating or restoring predictions, and fixed-view evaluation.

The [reproduction guide](docs/REPRODUCTION.md) links each study to its
source frame range, configuration, run order, and reference outputs.
The directory [preprocessing/](preprocessing/) contains the input
preparation scripts. The experiment directories preserve the original
paths and environment assumptions, so set local dataset, model, and
output paths before running them.

For supplementary work, see [exploratory experiments](experiments/exploratory/).
Individual reports document their scope and results.

## Open the demonstration

See the [T07 viewer guide](demo/README.md) for the scene selector,
launch steps, and asset requirements. The scene profiles include
KITTI03, 07, 09, 10, and 13. Rebuild or provide the maps, rendered
assets, and observation files referenced by the selected scene.


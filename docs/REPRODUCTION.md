# Reproduction guide

Frame ranges below are inclusive and use zero-based source indices.
A single study may contain multiple run-specific configurations.

| Study | Data and protocol | Saved files |
| --- | --- | --- |
| T0.3 SplaTAM baseline | Replica room0 | [Pinned setup, data, and run scripts](../experiments/replica-room0/) |
| Depth-confidence selection and mapping-frame budget | KITTI00, 0–199 | [Threshold study](../experiments/kitti00-depth-confidence/) and [frame-budget study](../experiments/kitti00-frame-budget/) |
| Map-assisted pose correction | KITTI08, 3000–3999, ORB/HARD/MASS × H0/H3, seed 0 | [G30 configs](../experiments/kitti08-pose-correction/source/configs/kitti/), [worker](../experiments/kitti08-pose-correction/scripts/worker.py), [trajectory results](../experiments/kitti08-pose-correction/evaluation_only/trajectory_results.json) |
| Historical replay | KITTI08, 3000–3999, H0/H3, seeds 0–2 | [G14 configs](../experiments/kitti08-historical-replay/source/configs/kitti/), [worker](../experiments/kitti08-historical-replay/scripts/worker.py), [evaluation summary](../experiments/kitti08-historical-replay/evaluation/summary.json) |
| F0/F1 appearance refinement | KITTI21, 1200–1399, Train160/Dev20/Test20. ORB-SLAM3 processed all 200 frames. | [Split](../experiments/kitti21-appearance-refinement/protocol/split.csv), [F0 configuration](../experiments/kitti21-appearance-refinement/splatam_train160_config.py), [F1 runner](../experiments/kitti21-appearance-refinement/run_f1_dev.sh), [test runner](../experiments/kitti21-appearance-refinement/run_test.sh), [test summary](../experiments/kitti21-appearance-refinement/eval/test_pair_summary.json) |
| Fixed-map semantic fusion | KITTI08 with SemanticKITTI, Virtual KITTI 2, and KITTI-360 | [Protocol and evaluation scripts](../experiments/fixed-map-semantic-fusion/) |
| Online semantic and dynamic feedback | KITTI08, 3000–3999, eight E2 groups. KITTI05/10, 0–999 each, B/TM/D/TMD, seeds 0–2. | [Runnable S23 guide](../experiments/online-semantic-dynamic/REPRODUCE.md), [registered configs](../experiments/online-semantic-dynamic/configurations/), [result report](../experiments/online-semantic-dynamic/report/) |

## Prepare a run

1. Download the files listed in the [data guide](../README.md#download-and-prepare-the-data).
   Align the stereo images, calibration, and generated observations by
   source frame. The [preprocessing directory](../preprocessing/) contains
   position, stereo depth, depth confidence, and semantic preparation
   code. The online study uses causal confidence. The offline demo
   confidence can use adjacent earlier and later frames.
2. In a separate working directory, match the environment and set the
   paths expected by the selected configuration and input manifest.
   Original scripts may also refer to specific Python executables,
   local model checkpoints, or compiled rasterizers.
3. Run the saved configuration with its corresponding launcher and
   evaluate the saved outputs on the original split.

## Run the selected study

**KITTI08 G14 and G30.** With the matching registration and inputs
prepared in each experiment directory, run its `scripts/worker.py`
with a configuration basename. For example, G14 uses `B00C0_S0` for
H0 seed 0 and `B03C0_S0` for the matching H3 run. G30 uses
`B03_ORB_S0` and `B03_MASS_S0` for the H3 pose comparison.
Consult the protocol in each directory for the preparation and
evaluation steps.

**KITTI21 F0/F1.** Prepare the Train160 inputs and F0 map using
`splatam_train160_config.py`. Use `run_f1_dev.sh` for F1 training and
development evaluation, then `run_test.sh` for the paired Test20
evaluation. The saved `protocol/split.csv` defines the frame membership.
The F1 comparison uses fixed cameras and geometry and updates
appearance on the Train160 views.

**KITTI05/08/10 online feedback.** Use `reproduce.py inputs` to build the
registered prediction bundles, `reproduce.py check --verify-hashes` to
check them, and `reproduce.py run --evaluate` for mapping and scoring.
The [S23 guide](../experiments/online-semantic-dynamic/REPRODUCE.md)
includes commands for reusing archived prediction bytes and building the
registered CUDA implementation.

## Demo and supplementary studies

The [T07 viewer](../demo/) uses local scene configurations and generated
assets. KITTI10 O3F offline vehicle cleanup uses a separate map from
the KITTI10 online feedback comparison. The
[supplementary studies](../experiments/exploratory/) include the S23-P
Pi³MOS state replacement, the Pi³MOS five-scene study, A2, earlier
TSDD experiments, and DG-SLAM.


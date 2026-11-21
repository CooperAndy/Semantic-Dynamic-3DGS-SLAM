"""S2 mapping-frame efficiency and held-out/no-held-out comparison."""

import os


primary_device = "cuda:0"
seed = 0
scene_name = "00"
group_name = "KITTI"

mode = os.environ.get("S2_MODE", "")
valid_modes = {"m025", "m050", "m100", "m190", "noheldout200"}
if mode not in valid_modes:
    raise ValueError(f"S2_MODE must be one of {sorted(valid_modes)}")

heldout = tuple(range(19, 200, 20))
training_candidates = tuple(index for index in range(200) if index not in heldout)


def nested_farthest_order(candidates):
    """Return deterministic nested temporal coverage, independent of results."""
    selected = {candidates[0], candidates[-1]}
    order = [candidates[0], candidates[-1]]
    while len(order) < len(candidates):
        remaining = (value for value in candidates if value not in selected)
        best = max(
            remaining,
            key=lambda value: (min(abs(value - chosen) for chosen in selected), -value),
        )
        selected.add(best)
        order.append(best)
    return tuple(order)


nested_order = nested_farthest_order(training_candidates)
if mode == "noheldout200":
    mapping_frame_indices = list(range(200))
    eval_only_frames = []
    run_name = f"seq{scene_name}_s2_confhard_t060_noheldout_200_{seed}"
else:
    budget = int(mode[1:])
    mapping_frame_indices = sorted(nested_order[:budget])
    eval_only_frames = list(heldout)
    run_name = f"seq{scene_name}_s2_confhard_t060_{mode}_holdout10_{seed}"

map_every = 1
keyframe_every = 5
mapping_window_size = 24
tracking_iters = 40
mapping_iters = 60

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,
    keyframe_every=keyframe_every,
    mapping_window_size=mapping_window_size,
    report_global_progress_every=50,
    # Formal metrics are computed separately on the same ten views and support.
    eval_every=200,
    eval_only_frames=eval_only_frames,
    mapping_frame_indices=mapping_frame_indices,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=50,
    use_wandb=False,
    wandb=dict(
        entity="theairlab",
        project="SplaTAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=False,
    ),
    data=dict(
        basedir="/root/autodl-tmp/slam_project/data/kitti/dataset/sequences",
        gradslam_data_cfg="./configs/data/kitti_seq00_s1_confhard_t060.yaml",
        sequence=scene_name,
        desired_image_height=376,
        desired_image_width=1241,
        start=0,
        end=-1,
        stride=1,
        num_frames=200,
    ),
    tracking=dict(
        use_gt_poses=True,
        forward_prop=True,
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0,
            cam_trans=0.0,
        ),
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500,
        ),
        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),
    ),
    viz=dict(
        render_mode="color",
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=True,
        viz_w=600,
        viz_h=340,
        viz_near=0.01,
        viz_far=100.0,
        view_scale=2,
        viz_fps=5,
        enter_interactive_post_online=True,
    ),
)

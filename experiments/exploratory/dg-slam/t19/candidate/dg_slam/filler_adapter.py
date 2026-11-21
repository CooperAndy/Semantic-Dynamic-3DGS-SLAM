"""Attach T19 candidate after tracking, before terminate_woBA; no GT access."""
import json
from pathlib import Path
import numpy as np
from dg_slam.filler_contract import anchor_policy
from dg_slam.filler_trace import FillerTrace

def configure_filler(model, mapped_c2w, arm, policy, output,
                     mixed_geometry_diagnostic=False, preserve_keyframes=True,
                     trace_frames=range(144, 176)):
    # Validate before importing runtime libraries or mutating any video state.
    receipt = anchor_policy(arm, policy, mixed_geometry_diagnostic)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh diagnostic output directory')
    import torch
    from lietorch import SE3
    from scipy.spatial.transform import Rotation
    video = model.video
    n = int(video.counter.value)
    if n < 1 or len(mapped_c2w) != n:
        raise ValueError('Mapped anchor count must match the final video keyframe count')
    before = video.poses[:n].detach().clone()
    snapshot = np.asarray(mapped_c2w)
    if snapshot.shape != (n,4,4) or not np.isfinite(snapshot).all():
        raise ValueError('Expected finite mapped c2w matrices')
    if not np.allclose(snapshot[:,3], [0,0,0,1], atol=1e-6, rtol=0):
        raise ValueError('Invalid homogeneous mapped pose')
    R = snapshot[:,:3,:3]
    if not np.allclose(R.transpose(0,2,1)@R, np.eye(3), atol=1e-5, rtol=0) or not np.allclose(np.linalg.det(R), 1, atol=1e-5, rtol=0):
        raise ValueError('Mapped rotation must belong to SO(3)')
    injected = before
    if policy == 'mapping_snapshot':
        seeds = np.concatenate([snapshot[:,:3,3], Rotation.from_matrix(R).as_quat()], axis=1)
        injected = SE3(torch.as_tensor(seeds, dtype=before.dtype, device=before.device)).inv().data
    output.mkdir(parents=True, exist_ok=False)
    trace = FillerTrace(output/'trace', trace_frames)
    np.savez_compressed(output/'anchor_source.npz', native_final_w2c=before.cpu().numpy(),
                        selected_w2c=injected.detach().cpu().numpy(), mapped_c2w=snapshot,
                        frame_indices=video.tstamp[:n].detach().cpu().numpy(),
                        disps=video.disps[:n].detach().cpu().numpy(),
                        disps_sens=video.disps_sens[:n].detach().cpu().numpy())
    if policy == 'mapping_snapshot':
        video.poses[:n] = injected
    model.traj_filler.preserve_keyframes = bool(preserve_keyframes)
    model.traj_filler.diagnostic_trace = trace
    receipt.update(arm=arm, preserve_keyframes=bool(preserve_keyframes),
                   anchor_preservation=bool(preserve_keyframes),
                   mode='preserve_KF' if preserve_keyframes else 'instrumented_legacy_all_frames',
                   trace_frames=sorted(trace.frames), trace_max_reserved_bytes=trace.max_bytes,
                   n_keyframes=n, GPU_validated=False, formal=False,
                   native_geometry_not_recomputed=True)
    (output/'anchor_policy.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt

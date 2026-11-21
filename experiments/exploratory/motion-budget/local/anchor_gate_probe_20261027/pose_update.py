"""Camera-centred incremental w2c pose for a *candidate* M1-2 recovery control.

For T0=[R0,t0] and camera-local increment (omega,v), use
T=[R(omega) R0, R(omega) t0 + v].  A pure rotation therefore leaves the
world-space camera centre unchanged.  Quaternion order is (w,x,y,z), as in
the archived SplaTAM renderer.  This module has no renderer dependency.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, scalar first; broadcasts over leading dimensions."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dim=-1)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Differentiable (w,x,y,z) quaternion to rotation matrix."""
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(q.shape[:-1] + (3, 3))


def rotation_vector_to_quaternion(omega: torch.Tensor) -> torch.Tensor:
    """Stable exponential map at omega=0; omega is in radians."""
    theta = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    half = theta * 0.5
    vector_scale = 0.5 * torch.sinc(theta / (2 * torch.pi))
    return torch.cat((torch.cos(half), vector_scale * omega), dim=-1)


def compose_local_increment(
    base_q: torch.Tensor,
    base_t: torch.Tensor,
    omega: torch.Tensor,
    translation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return quaternion and w2c translation after a camera-local SE(3) step."""
    q_inc = rotation_vector_to_quaternion(omega)
    q_new = F.normalize(quaternion_multiply(q_inc, F.normalize(base_q, dim=-1)), dim=-1)
    t_new = (quaternion_to_matrix(q_inc) @ base_t.unsqueeze(-1)).squeeze(-1) + translation
    return q_new, t_new


def insert_camera_at_frame(
    params: dict[str, torch.Tensor], frame: int, q: torch.Tensor, t: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Shallow map view with differentiable replacement of one camera column."""
    base_q = params["cam_unnorm_rots"].detach()
    base_t = params["cam_trans"].detach()
    if base_q.shape[0:2] != (1, 4) or base_t.shape[0:2] != (1, 3):
        raise ValueError("Expected G11 camera arrays of shape 1x4xN and 1x3xN")
    if not 0 <= frame < base_q.shape[-1] or base_q.shape[-1] != base_t.shape[-1]:
        raise ValueError("Invalid frame index or camera array mismatch")
    out = dict(params)
    out["cam_unnorm_rots"] = torch.cat((base_q[..., :frame], q.reshape(1, 4, 1), base_q[..., frame + 1:]), dim=-1)
    out["cam_trans"] = torch.cat((base_t[..., :frame], t.reshape(1, 3, 1), base_t[..., frame + 1:]), dim=-1)
    return out

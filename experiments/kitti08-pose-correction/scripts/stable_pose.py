"""Keep the native quaternion encoding while preserving the requested camera centre."""
import torch
import torch.nn.functional as F

def install(external):
    original=external.set_external
    assert not getattr(original,'_stable_camera_center',False)
    def set_external(params,w2c,index,core):
        with torch.no_grad():
            matrix=w2c.to(torch.float64)
            assert matrix.shape==(4,4) and torch.isfinite(matrix[:3,:4]).all()
            center=torch.linalg.solve(matrix[:3,:3],-matrix[:3,3])
            result=original(params,w2c,index,core)
            rotation=core.build_rotation(F.normalize(params['cam_unnorm_rots'][...,index].detach()))[0]
            translation=-(rotation.to(torch.float64)@center)
            params['cam_trans'][...,index]=translation.to(params['cam_trans'].dtype)
            return result
    set_external._stable_camera_center=True
    external.set_external=set_external
    return original

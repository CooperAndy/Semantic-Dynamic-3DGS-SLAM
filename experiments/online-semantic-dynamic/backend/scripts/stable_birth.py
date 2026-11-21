"""Keep native camera unprojection/RGB/scales; solve only the affine world transform in FP64."""
import torch

def install(core):
    original = core.get_pointcloud
    def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True,
                       mask=None, compute_mean_sq_dist=False, mean_sq_dist_method='projective'):
        result = original(color, depth, intrinsics, w2c, transform_pts=False,
                          mask=mask, compute_mean_sq_dist=compute_mean_sq_dist,
                          mean_sq_dist_method=mean_sq_dist_method)
        cloud, distance = result if compute_mean_sq_dist else (result, None)
        if transform_pts:
            assert w2c.shape == (4, 4) and cloud.device == w2c.device
            assert torch.equal(w2c[3], w2c.new_tensor([0., 0., 0., 1.]))
            matrix = w2c.to(torch.float64)
            camera = cloud[:, :3].to(torch.float64)
            world = torch.linalg.solve(matrix[:3, :3], (camera - matrix[:3, 3]).T).T
            cloud = torch.cat((world.to(cloud.dtype), cloud[:, 3:]), dim=1)
        return (cloud, distance) if compute_mean_sq_dist else cloud
    core.get_pointcloud = get_pointcloud
    return original

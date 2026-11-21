"""Actual rasterizer pixel-center check, independent of SLAM/mapping."""
from pathlib import Path
import sys,json,math
import torch
ROOT=Path('/root/autodl-tmp/t04_chy_kitti00_200_20261012')
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'source/dg_slam')]
from dg_slam.gaussian.gaussian_render import render
from dg_slam.gaussian.graphics_utils import getProjectionMatrix
p=json.loads((ROOT/'runtime200/protocol.json').read_text());fx,fy,cx,cy=p['K'];W,H=640,192
view=torch.eye(4,device='cuda');camera=torch.zeros(3,device='cuda');results=[]
for corrected in [False,True]:
    for x,y,z in [(0.,0.,10.),(.6,-.2,10.),(-.7,.3,15.)]:
        projection=getProjectionMatrix(znear=.01,zfar=100.,fovX=2*math.atan(W/(2*fx)),fovY=2*math.atan(H/(2*fy))).T.cuda()
        projection[2,0]=(2*cx+(1 if corrected else 0))/W-1;projection[2,1]=(2*cy+(1 if corrected else 0))/H-1
        xyz=torch.tensor([[x,y,z]],device='cuda');zero=torch.zeros((1,1,3),device='cuda');rest=torch.zeros((1,15,3),device='cuda')
        image=render(xyz,zero,rest,torch.tensor([[.8]],device='cuda'),torch.full((1,3),.04,device='cuda'),torch.tensor([[1.,0,0,0]],device='cuda'),0,3,camera,view,projection,2*math.atan(W/(2*fx)),2*math.atan(H/(2*fy)),H,W,override_color=torch.ones((1,3),device='cuda'))['render'][0].detach()
        yy,xx=torch.meshgrid(torch.arange(H,device='cuda'),torch.arange(W,device='cuda'),indexing='ij')
        u=float((image*xx).sum()/image.sum());v=float((image*yy).sum()/image.sum())
        expected=[fx*x/z+cx,fy*y/z+cy]
        results.append({'corrected':corrected,'xyz':[x,y,z],'centroid':[u,v],'expected':expected,'error_px':[u-expected[0],v-expected[1]]})
assert all(max(abs(v) for v in r['error_px'])<.03 for r in results if r['corrected'])
assert all(max(abs(v+.5) for v in r['error_px'])<.03 for r in results if not r['corrected'])
(ROOT/'projection_rasterizer_audit.json').write_text(json.dumps({'status':'PASS','finding':'old projection is -0.5 pixel on both axes; +1 NDC numerator corrects actual rasterizer','results':results},indent=2));print(json.dumps(results))

"""Real weights, CUDA features/correlation/native BA, on a synthetic engineering input."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from runtime_boot import F,R,bootstrap,guard_gt,dump

def main():
    bootstrap(); audit=guard_gt()
    import torch
    from lietorch import SE3
    from dg_slam.dg_model import dg_model
    from depth_video import DepthVideo
    from trajectory_filler import PoseTrajectoryFiller
    from dg_slam.filler_adapter import configure_filler
    from state_control import NativeState
    from analyze_trace import inspect
    torch.manual_seed(1219); torch.cuda.manual_seed_all(1219)
    model=SimpleNamespace()
    dg_model.load_weights(model,str(R/'01_source_env/weights/droid.pth'))
    model.video=DepthVideo([64,64],32)
    model.traj_filler=PoseTrajectoryFiller(model.net,model.video)
    image=torch.randint(0,256,(1,3,64,64),dtype=torch.uint8)
    depth=torch.ones(64,64); K=torch.tensor([64.,64.,32.,32.]); mask=torch.zeros(64,64,dtype=torch.uint8)
    data=[(t,image.clone(),depth.clone(),K.clone(),None,mask.clone()) for t in [0,1,2]]
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=True):
        x=image[None].cuda().float()/255
        x=(x-model.traj_filler.MEAN)/model.traj_filler.STDV
        fmap=model.net.fnet(x).squeeze(0)
        net,inp=model.net.cnet(x).split([128,128],dim=2)
        for i,t in enumerate([0,2]):
            pose=SE3.Identity(1,device='cuda').data[0]
            model.video.append(t,image[0],pose,depth,depth,K/8,fmap,net.tanh()[0,0],inp.relu()[0,0],mask)
    mapped=SE3(model.video.poses[:2]).inv().matrix().cpu().numpy()
    snap=NativeState(model)
    output=F/'smoke';output.mkdir(exist_ok=False)
    results=[]
    for name,preserve in [('legacy',False),('fixed',True)]:
        hashes=snap.restore(model)
        case=output/name
        configure_filler(model,mapped,'NO_GS_REFINE','mapping_snapshot',case,
                         mixed_geometry_diagnostic=True,preserve_keyframes=preserve,trace_frames=range(3))
        anchors=model.video.poses[:2].clone()
        poses=model.traj_filler(data)
        assert poses.data.shape==(3,7) and torch.isfinite(poses.data).all()
        assert model.video.counter.value==2 and torch.equal(model.video.poses[:2],anchors)
        if preserve: assert torch.equal(poses.data[[0,2]],anchors)
        steps=inspect(case/'trace'); assert len(steps)==6
        events=[json.loads(x) for x in (case/'trace/events.jsonl').read_text().splitlines()]
        assert all(not any(x['nonfinite'].values()) for x in events)
        dump(case/'smoke_result.json',{'status':'PASS','real_CUDA':True,'real_weights':True,'native_BA':True,'same_state_hashes':hashes,'iterations':len(steps),'GT_guard':audit})
        results.append({'case':name,'status':'PASS','iterations':len(steps)})
    assert audit['blocked_GT_opens']==0
    dump(F/'receipts/candidate_gpu_smoke.json',{'status':'PASS','cases':results,'scientific_quality_evidence':False,'GT_guard':audit})
    print('CANDIDATE_GPU_SMOKE_PASS',flush=True)

if __name__=='__main__':main()

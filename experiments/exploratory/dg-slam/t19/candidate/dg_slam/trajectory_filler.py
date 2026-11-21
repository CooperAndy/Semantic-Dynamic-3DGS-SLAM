import torch
import lietorch

from lietorch import SE3
from collections import OrderedDict
from factor_graph import FactorGraph
from droid_net import DroidNet
import geom.projective_ops as pops
from dg_slam.filler_contract import batch_plan, ordered_indices

class PoseTrajectoryFiller:
    """ This class is used to fill in non-keyframe poses """

    def __init__(self, net, video, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.count = 0
        self.video = video
        self.device = device
        self.preserve_keyframes = True
        self.diagnostic_trace = None

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image)

    def __fill(self, tstamps, images, depths, intrinsics, seg_masks):
        """ fill operator """

        tt = torch.as_tensor(tstamps, device="cuda")
        images = torch.stack(images, 0)
        intrinsics = torch.stack(intrinsics, 0)
        
        inputs = images[:,:,[0,1,2]].to(self.device) / 255.0
        seg_masks = torch.stack(seg_masks, 0)
        depths = torch.stack(depths, 0)
        
        ### linear pose interpolation ###
        N = self.video.counter.value
        M = len(tstamps)

        ts = self.video.tstamp[:N]
        Ps = SE3(self.video.poses[:N])

        t0 = torch.as_tensor([ts[ts<=t].shape[0] - 1 for t in tstamps])
        t1 = torch.where(t0<N-1, t0+1, t0)

        dt = ts[t1] - ts[t0] + 1e-3
        dP = Ps[t1] * Ps[t0].inv()

        v = dP.log() / dt.unsqueeze(-1)
        w = v * (tt - ts[t0]).unsqueeze(-1)
        Gs = SE3.exp(w) * Ps[t0]

        # extract features (no need for context features)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        fmap = self.__feature_encoder(inputs)

        if N + M > len(self.video.poses):
            raise ValueError('Video buffer has no capacity for temporary non-keyframes')
        anchors_before = self.video.poses[:N].clone()
        old_hook = getattr(self.video, 'ba_diagnostic_hook', None)
        trace = self.diagnostic_trace
        active_trace = trace is not None and trace.begin(tstamps)
        self.video.counter.value = N + M
        try:
            self.video[N:N+M] = (tt, images[:,0], Gs.data, depths, depths, intrinsics / 8.0, fmap, None, None, seg_masks)
            graph = FactorGraph(self.video, self.update)
            graph.add_factors(t0.cuda(), torch.arange(N, N+M).cuda())
            graph.add_factors(t1.cuda(), torch.arange(N, N+M).cuda())
            if active_trace:
                sources = torch.unique(torch.cat([t0, t1])).to(self.video.poses.device)
                trace.record('initialization', temporary_frames=tt, temporary_w2c=self.video.poses[N:N+M],
                             anchor_frames=ts, anchor_w2c=anchors_before, source_indices=sources,
                             source_disps=self.video.disps[sources], source_disps_sens=self.video.disps_sens[sources],
                             source_features=self.video.fmaps[sources], temporary_features=self.video.fmaps[N:N+M],
                             source_intrinsics=self.video.intrinsics[sources], source_seg_masks=self.video.seg_masks[sources],
                             temporary_depths=depths, temporary_seg_masks=seg_masks)
                self.video.ba_diagnostic_hook = trace.record
            for itr in range(6):
                if active_trace:
                    trace.iteration = itr
                graph.update(N, N+M, motion_only=True,
                             diagnostic_hook=trace.record if active_trace else None)
            Gs = SE3(self.video.poses[N:N+M].clone())
            if not torch.equal(self.video.poses[:N], anchors_before):
                raise RuntimeError('Filler modified fixed anchor poses')
            return Gs
        finally:
            self.video.counter.value = N
            self.video.ba_diagnostic_hook = old_hook

    @torch.no_grad()
    def __call__(self, image_stream):
        """ fill in poses of non-keyframe images """

        n = int(self.video.counter.value)
        keys = ordered_indices(self.video.tstamp[:n].detach().cpu().tolist())
        anchors = self.video.poses[:n].clone()
        pose_list, batch = [], []
        previous = -1

        def flush(rows):
            if int(self.video.counter.value) != n:
                raise RuntimeError('Keyframe count changed during filling')
            slots, positions = batch_plan([row[0] for row in rows], keys)
            if not self.preserve_keyframes:
                positions = list(range(len(rows)))
                slots = [('temporary', i) for i in positions]
            filled = None
            if positions:
                selected = [rows[i] for i in positions]
                filled = self.__fill([r[0] for r in selected], [r[1] for r in selected],
                                     [r[2] for r in selected], [r[3] for r in selected],
                                     [r[5] for r in selected])
            for kind, index in slots:
                pose_list.append(SE3(anchors[index:index+1].clone()) if kind == 'anchor'
                                 else filled[index:index+1])

        for row in image_stream:
            # Keep original 16-input-frame batch boundaries, even when anchors are skipped.
            value = ordered_indices([row[0]])[0]
            if value <= previous:
                raise ValueError('Input stream must be strictly increasing')
            previous = value
            batch.append(row)
            if len(batch) == 16:
                flush(batch)
                batch = []
        if batch:
            flush(batch)
        if not pose_list:
            raise ValueError('Empty image stream')
        if not torch.equal(self.video.poses[:n], anchors):
            raise RuntimeError('Anchor poses changed during filling')
        return lietorch.cat(pose_list, 0)

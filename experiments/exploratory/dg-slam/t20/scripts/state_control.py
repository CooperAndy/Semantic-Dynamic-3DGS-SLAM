"""Exact paired native-video / network / RNG restoration before anchor policy."""
import copy
import hashlib
import json
import random
import numpy as np
import torch


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def gpu_copy(value, device):
    if torch.is_tensor(value):
        return value.to(device).clone()
    if isinstance(value, dict):
        return {k: gpu_copy(v, device) for k, v in value.items()}
    return copy.deepcopy(value)


def digest(value):
    h = hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous().numpy()
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                h.update(str(key).encode()); add(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x: add(item)
        else:
            h.update(repr(x).encode())
    add(value)
    return h.hexdigest()


class NativeState:
    def __init__(self, model, temporary_capacity=16):
        v = model.video
        self.n = int(v.counter.value)
        self.capacity = min(len(v.poses), self.n + temporary_capacity)
        assert self.capacity >= self.n + temporary_capacity
        # All tensor attributes, including stale temporary slots which old code may reuse.
        self.video = {}
        self.sliced = {}
        for key, val in vars(v).items():
            if torch.is_tensor(val):
                sliced = val.ndim > 0 and val.shape[0] == len(v.poses)
                self.video[key] = cpu_copy(val[:self.capacity] if sliced else val)
                self.sliced[key] = sliced
        self.dicts = {key: cpu_copy(getattr(v, key)) for key in ('depth_warp_mask', 'depth_warp_mask_ori')}
        self.flags = {key: getattr(v, key) for key in ('use_segmask','use_depth_warp','use_depth_mask','get_depth_warp_mask_ori','stereo')}
        self.ready = int(v.ready.value)
        self.net = cpu_copy(model.net.state_dict())
        self.training = {k: module.training for k, module in model.net.named_modules()}
        assert not any(self.training.values()), 'Network must be in eval mode'
        self.rng = self.get_rng()
        self.filler_count = model.traj_filler.count
        self.expected = self.hash_current(model)

    @staticmethod
    def get_rng():
        return {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch_cpu': torch.get_rng_state().clone(),
                'torch_cuda': [x.cpu().clone() for x in torch.cuda.get_rng_state_all()]}

    def hash_current(self, model):
        v = model.video
        payload = {key: getattr(v, key)[:self.capacity] if self.sliced[key] else getattr(v,key) for key in self.video}
        active = {key: getattr(v, key)[:self.n] for key in self.video if self.sliced[key]}
        values = {'video_all_captured': payload, 'video_active': active,
                  'video_dicts': {k:getattr(v,k) for k in self.dicts},
                  'flags': {k:getattr(v,k) for k in self.flags},
                  'count_ready': [int(v.counter.value),int(v.ready.value)],
                  'network': model.net.state_dict(),
                  'network_training': {k:x.training for k,x in model.net.named_modules()},
                  'RNG': self.get_rng()}
        return {key: digest(val) for key,val in values.items()}

    @torch.no_grad()
    def restore(self, model):
        v = model.video
        for key, val in self.video.items():
            dst = getattr(v,key)
            if self.sliced[key]: dst[:self.capacity].copy_(val)
            else: dst.copy_(val)
        for key,val in self.dicts.items(): setattr(v,key,gpu_copy(val,v.poses.device))
        for key,val in self.flags.items(): setattr(v,key,val)
        v.counter.value = self.n; v.ready.value = self.ready
        model.net.load_state_dict(self.net)
        for key,module in model.net.named_modules(): module.train(self.training[key])
        model.traj_filler.count = self.filler_count
        model.traj_filler.diagnostic_trace = None
        model.traj_filler.preserve_keyframes = True
        v.ba_diagnostic_hook = None
        random.setstate(self.rng['python']); np.random.set_state(self.rng['numpy'])
        torch.set_rng_state(self.rng['torch_cpu'])
        torch.cuda.set_rng_state_all(self.rng['torch_cuda'])
        actual = self.hash_current(model)
        assert actual == self.expected, {k:(self.expected[k],actual[k]) for k in actual if actual[k]!=self.expected[k]}
        return actual

    def save(self, path):
        torch.save({'n':self.n,'capacity':self.capacity,'video':self.video,'sliced':self.sliced,
                    'dicts':self.dicts,'flags':self.flags,'ready':self.ready,'network':self.net,
                    'network_training':self.training,'rng':self.rng,'expected_hashes':self.expected}, path)

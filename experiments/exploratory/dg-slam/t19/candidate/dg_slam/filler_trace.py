"""Opt-in diagnostic snapshots; never changes optimization targets or accepts poses."""
from pathlib import Path
import json
import numpy as np

class TraceBudgetExceeded(RuntimeError):
    pass

def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()

class FillerTrace:
    def __init__(self, root, frames=range(144, 176), max_bytes=256*1024**2):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.frames = set(int(x) for x in frames)
        self.max_bytes = int(max_bytes)
        self.bytes = 0
        self.records = 0
        self.batch = None
        self.iteration = -1

    def begin(self, frames):
        ids = [int(x) for x in frames]
        self.batch = ids if any(x in self.frames for x in ids) else None
        self.iteration = -1
        return self.batch is not None

    def record(self, stage, **values):
        if self.batch is None:
            return
        arrays = {key: array(value) for key, value in values.items()}
        # Reserve the uncompressed payload before writing. Stop the diagnostic on budget failure.
        reserved = sum(x.nbytes for x in arrays.values()) + 4096
        if self.bytes + reserved > self.max_bytes:
            raise TraceBudgetExceeded('Trace budget exceeded; no silent trace truncation or optimizer fallback')
        stem = f'b{self.batch[0]:04d}_i{self.iteration:02d}_{self.records:04d}_{stage}'
        path = self.root/(stem+'.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        info = {'stage': stage, 'iteration': self.iteration, 'frames': self.batch,
                'file': path.name, 'shapes': {k: list(v.shape) for k,v in arrays.items()},
                'nonfinite': {k: int(np.count_nonzero(~np.isfinite(v))) for k,v in arrays.items() if v.dtype.kind in 'fc'},
                'reserved_bytes': reserved}
        with (self.root/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(info, allow_nan=False)+'\n')
        self.bytes += reserved
        self.records += 1

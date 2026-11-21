"""Audit actual RAFT file access and raw-depth ABI; no GPU or GT scoring."""
from common import *
import argparse, csv, re

def audit_trace(trace, inputs, output_dir, frames):
    """A depth write/close seals a frame; source opens cannot run ahead of it."""
    sources = {os.path.normpath(str(p)): (side, i) for side, paths in inputs.items() for i,p in enumerate(paths)}
    targets = {os.path.normpath(str(output_dir / f'{i:06d}_depth.npy')): i for i in range(frames)}
    open_re = re.compile(r'openat\([^,]+, "((?:\\.|[^"\\])*)", ([^)]*)')
    write_re = re.compile(r'write\(\d+<([^>]+)>')
    close_re = re.compile(r'close\(\d+<([^>]+)>\)\s+=\s+0')
    success_re = re.compile(r'=\s+(\d+)(?:<|\s|$)')
    opened, writing, written, pending = set(), set(), set(), {}
    sealed = -1
    attempts = 0

    def entry(path, flags):
        nonlocal attempts
        path = os.path.normpath(path)
        assert '/poses/' not in path and not path.endswith('groundtruth.txt') and '/evaluation_only/' not in path, ('Forbidden GT/evaluation open',path)
        if path in sources:
            side,i = sources[path]
            assert i <= sealed+1, ('Future image open', side, i, sealed)
            attempts += 1
        if path in targets and 'O_WRONLY' in flags:
            assert targets[path] == sealed+1, ('Out-of-order depth output',path,sealed)
            writing.add(path)
        return path

    for line in Path(trace).open():
        fields=line.split()
        assert fields, 'Empty strace line'
        pid=fields[0]
        m=open_re.search(line)
        if m:
            path=entry(json.loads('"'+m.group(1)+'"'),m.group(2))
            if '<unfinished ...>' in line:
                assert pid not in pending
                pending[pid]=path
            elif success_re.search(line) and path in sources:
                opened.add(sources[path])
        elif '<... openat resumed>' in line:
            assert pid in pending, ('Unmatched resumed open',line)
            path=pending.pop(pid)
            if success_re.search(line) and path in sources:
                opened.add(sources[path])
        m=write_re.search(line)
        if m:
            path=os.path.normpath(m.group(1))
            if path in writing:
                written.add(path)
        m=close_re.search(line)
        if m:
            path=os.path.normpath(m.group(1))
            if path in writing:
                i=targets[path]
                assert path in written and i==sealed+1 and all((side,i) in opened for side in inputs), ('Depth not sealed after both images',i)
                sealed=i
                writing.remove(path)
    assert not pending and not writing
    assert sealed==frames-1 and len(opened)==frames*len(inputs), ('Incomplete source/output trace',sealed,len(opened))
    return dict(frames=frames, unique_image_sources=len(opened), image_open_attempts=attempts,
                seals=sealed+1, GT_forbidden_opens=0, future_image_opens=0,
                unfinished_open_entries_checked_at_entry=True)

def main():
    import numpy as np
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window',required=True)
    parser.add_argument('--side',choices=['left','right'],required=True)
    args=parser.parse_args()
    reg=json.loads((R/'registration_cpu.json').read_text())
    win=next(x for x in reg['windows'] if x['id']==args.window)
    W=R/'windows'/args.window
    output=W/f'raw_{args.side}'
    view=Path(win['input_view'])
    sources={side:[view/f'image_{2 if side=="left" else 3}'/f'{i:06d}.png' for i in range(1000)] for side in ['left','right']}
    trace=W/f'{args.side}_file_activity.log'
    proof=audit_trace(trace,sources,output,1000)
    stats=list(csv.DictReader((output/'depth_stats.csv').open()))
    summary=json.loads((output/'generation_summary.json').read_text())
    assert len(stats)==summary['frames']==1000 and summary['shape']==win['shape']
    assert summary['valid_iters']==32 and summary['checkpoint_sha256']==reg['checkpoint_sha256']
    assert abs(summary['calibration']['fx']*summary['calibration']['baseline_m']-win['projection_beta'])<1e-9
    if args.side=='left':
        assert summary['raw_depth_range_m']==[.5,80.] and summary['mixed_precision'] and summary['corr_implementation']=='alt'
    else:
        assert summary['mirrored_inputs_before_inference'] and summary['reference_view']=='image_3'
    files=[]
    for i,row in enumerate(stats):
        assert int(row['index'])==i and row['frame_id']==f'{i:06d}'
        p=output/f'{i:06d}_depth.npy';array=np.load(p,allow_pickle=False)
        assert array.dtype==np.float32 and array.shape==tuple(win['shape']) and np.isfinite(array).all()
        assert np.all((array==0)|((array>=.5)&(array<=80.)))
        assert abs(float((array>0).mean())-float(row['valid_ratio']))<1e-12
        if args.side=='right':assert int(row['flow_sign_to_positive_disparity']) in [-1,1]
        files.append(dict(index=i,path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    assert len(list(output.glob('*_depth.npy')))==1000
    dump(W/'receipts'/f'stereo_{args.side}_audit.json',dict(
        status='PASS_1000_STEREO_CURRENT_PAIR_TRACE_AND_FLOAT32_METRIC_DEPTH',at=now(),side=args.side,
        window=args.window,proof=proof,files=files,trace_sha256=sha(trace),
        CPU_registration_sha256=sha(R/'registration_cpu.json'),script_sha256=sha(__file__),
        timing_scope='Shared recorded inference milliseconds are not fairFPS',
        GPU_calls_in_auditor=0,GT_scoring=False,quality_adoption=False))
    print('PASS_STEREO_TRACE_AND_DEPTH',args.window,args.side,flush=True)

if __name__=='__main__':main()

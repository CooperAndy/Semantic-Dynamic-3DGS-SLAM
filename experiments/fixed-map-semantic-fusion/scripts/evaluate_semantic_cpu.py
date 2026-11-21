"""Evaluate already-produced class arrays against independent mapped 2D GT."""
import argparse,json
from pathlib import Path
import numpy as np
from semantic_cpu import score_semantics,risk_coverage

def main():
    p=argparse.ArgumentParser();p.add_argument('--gt',required=True);p.add_argument('--prediction',required=True)
    p.add_argument('--classes',required=True,type=int);p.add_argument('--out',required=True);p.add_argument('--gt-provenance',required=True)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError('use a new evaluation output')
    gt=np.load(a.gt,allow_pickle=False)
    with np.load(a.prediction,allow_pickle=False) as v:
        result=score_semantics(gt,v['label'],a.classes);curve=risk_coverage(gt,v['label'],v['score'])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'gt_provenance':a.gt_provenance,'metrics':result,'risk_coverage':curve},indent=2),encoding='utf-8')

if __name__=='__main__':main()

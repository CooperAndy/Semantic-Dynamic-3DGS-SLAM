"""Post-hoc CPU inspection of saved effective weights; never changes BA inputs."""
import csv
import json
from pathlib import Path
import numpy as np
F=Path(__file__).resolve().parents[1]
def load_npz(p):
    with np.load(p) as z:return {k:a.copy() for k,a in z.items()}
def sample_mask(mask,xy):
    h,w=mask.shape
    x=np.rint(xy[...,0]).astype(int);y=np.rint(xy[...,1]).astype(int)
    inside=(x>=0)&(x<w)&(y>=0)&(y<h)
    sampled=mask[np.clip(y,0,h-1),np.clip(x,0,w-1)] & inside
    return sampled,inside
def write(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def main():
    assert json.loads((F/'receipts/raw_seal.json').read_text())['status']=='SEALED_BEFORE_CPU_GT'
    cases=json.loads((F/'protocol.json').read_text())['future_filler_cases']
    output=[];frame_totals={}
    for case in cases:
        root=F/f"runs/DEV200_s2718_{case['arm']}"/case['case']/'trace'
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        inits={};groups={}
        for event in events:
            batch=tuple(event['frames'])
            if event['stage']=='initialization':inits[batch]=load_npz(root/event['file'])
            elif event['stage'] in ('native_before_ba','graph_before_ba'):
                groups.setdefault((batch,event['iteration']),{})[event['stage']]=load_npz(root/event['file'])
        for (batch,iteration),group in sorted(groups.items()):
            initial=inits[batch];n=len(initial['anchor_frames'])
            native=group['native_before_ba'];graph=group['graph_before_ba']
            masks=initial['temporary_seg_masks'][:,3::8,3::8].astype(bool)
            targets=native['target'].transpose(0,2,3,1)
            weights=native['effective_weight'].transpose(0,2,3,1)
            projected=graph['projected_before'][0]
            residual=np.sum(weights*(targets-projected)**2,axis=-1)
            scalar=weights.sum(axis=-1)
            aggregate={key:0. for key in ['weight','residual','source_invalid_weight','source_invalid_residual','target_person_weight','target_person_residual','target_outside_weight','target_outside_residual']}
            for edge,j in enumerate(native['jj']):
                local=int(j)-n;assert 0<=local<len(batch)
                frame=int(initial['temporary_frames'][local]);assert frame==batch[local]
                person,inside=sample_mask(masks[local],targets[edge])
                invalid=native['source_disps_sens'][edge]<=0
                row={'weight':float(scalar[edge].sum()),'residual':float(residual[edge].sum()),
                     'source_invalid_weight':float(scalar[edge][invalid].sum()),'source_invalid_residual':float(residual[edge][invalid].sum()),
                     'target_person_weight':float(scalar[edge][person].sum()),'target_person_residual':float(residual[edge][person].sum()),
                     'target_outside_weight':float(scalar[edge][~inside].sum()),'target_outside_residual':float(residual[edge][~inside].sum())}
                for key in aggregate:aggregate[key]+=row[key]
                if iteration==0:
                    key=(case['case'],frame)
                    dest=frame_totals.setdefault(key,{k:0. for k in aggregate})
                    for field in dest:dest[field]+=row[field]
            report={'case':case['case'],'batch_first_frame':batch[0],'iteration':iteration,
                    'effective_weight_sum':aggregate['weight'],'fixed_target_residual':aggregate['residual']}
            for kind in ['source_invalid','target_person','target_outside']:
                report[kind+'_weight_percent']=100*aggregate[kind+'_weight']/max(aggregate['weight'],1e-30)
                report[kind+'_residual_percent']=100*aggregate[kind+'_residual']/max(aggregate['residual'],1e-30)
            output.append(report)
    frame_rows=[]
    for (case,frame),values in sorted(frame_totals.items()):
        item={'case':case,'frame':frame,'iteration':0,'effective_weight_sum':values['weight']}
        for kind in ['source_invalid','target_person','target_outside']:
            item[kind+'_weight_percent']=100*values[kind+'_weight']/max(values['weight'],1e-30)
            item[kind+'_residual_percent']=100*values[kind+'_residual']/max(values['residual'],1e-30)
        frame_rows.append(item)
    write(F/'T20_有效权重诊断.csv',output);write(F/'T20_首轮逐帧权重诊断.csv',frame_rows)
    receipt={'status':'PASS','iterations':len(output),'first_iteration_frames':len(frame_rows),
             'scope':'Post-hoc saved-trace inspection, no input/weight changes, no additional GPU runs',
             'limits':['Person masks are segmentation proxies, not GT motion labels','Target masks sampled at rounded learned-target feature coordinates; outside pixels are counted separately','Fractions describe existing weights; they do not establish causal benefits of filtering']}
    (F/'receipts/weight_inspection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()

from pathlib import Path
from PIL import Image,ImageDraw
import json
root=Path(__file__).resolve().parent.parent.parent
p=root/'特调地图/vehicle_state_experiment/seq03_m3f_20261116'
rows=json.loads((p/'object_trial/deduplicated_observations.json').read_text(encoding='utf8'))
rows={r['source_frame']:r for r in rows if r['object_id']=='V005'}
src=root/'slam_project/data/kitti/sequences/03/image_2'
frames=[140,148,156,164,172,180,188];w=900;h=round(375*w/1242)
out=Image.new('RGB',(w,h*len(frames)));d=ImageDraw.Draw(out)
for i,n in enumerate(frames):
 r=rows[n];out.paste(Image.open(src/f'{n:06d}.png').resize((w,h)),(0,i*h))
 x1,y1,x2,y2=r['bbox'];d.rectangle((x1*w/1242,y1*h/375+i*h,x2*w/1242,y2*h/375+i*h),outline='red',width=2)
 d.text((5,i*h+5),f"frame {n} score={r['raw_score']:.3f}",fill='yellow')
out.save(p/'v005_diagnostic.jpg',quality=90)

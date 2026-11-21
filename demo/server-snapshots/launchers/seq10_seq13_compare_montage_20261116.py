from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path('/root/autodl-tmp'); DEST=ROOT/'seq10_seq13_visual_comparison_20261116';DEST.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
def run(seq,frames,preview):
 if seq=='10':
  r=ROOT/'seq10_m3f_20261116';runtime=json.loads((r/'bundle/runtime.json').read_text());common=r/'comparison20'; dirs=[common/'old_orb_f1_common',common/'m3_common',common/'m3f_common',common/'m3f_selected']; labels=['Original RGB','Old ORB F1 (ORB camera)','M3 (ORB camera)','M3F (ORB camera)','M3F (MASS camera)'];offset=0
 else:
  r=ROOT/'seq13_000780_001039_O3F_260_20261115';runtime=json.loads((r/'bundle/runtime.json').read_text());a=r/'compare_m3f_o3f';b=r/'compare_m3_f1_20261116';dirs=[a/'O3F_common',b/'M3_common',b/'M3F_common',b/'M3F_own'];labels=['Original RGB','Old O3F (ORB camera)','M3 (ORB camera)','M3F (ORB camera)','M3F (MASS camera)'];offset=780
 W=460;H=166;head=52;gap=8;row=H+34; canvas=Image.new('RGB',(5*W+6*gap,head+len(frames)*row+gap),'#f3f5f8');draw=ImageDraw.Draw(canvas)
 for col,label in enumerate(labels):draw.text((gap+col*(W+gap),12),label,font=small,fill='#17283b')
 for j,frame in enumerate(frames):
  rid=frame-offset;imgs=[Path(runtime['rows'][rid]['rgb'])]+[d/f'{frame:06}.png' for d in dirs]
  for col,p in enumerate(imgs):
   with Image.open(p) as im:tile=ImageOps.contain(im.convert('RGB'),(W,H),Image.Resampling.LANCZOS)
   x=gap+col*(W+gap)+(W-tile.width)//2;y=head+j*row+(H-tile.height)//2;canvas.paste(tile,(x,y))
  draw.text((gap,head+j*row+H+4),f'Seq {seq} | frame {frame:06}',font=font,fill='#17283b')
 out=DEST/f'seq{seq}_{preview}.jpg';canvas.save(out,quality=90,subsampling=0);print(out,flush=True)
run('10',list(range(19,200,20))+list(range(819,1000,20)),'all20')
run('10',[39,99,199,879,939,999],'preview6')
run('13',[780,799,819,839,859,879,899,919,939,959,979,1009,1039],'all13')
run('13',[799,839,879,939,979,1039],'preview6')

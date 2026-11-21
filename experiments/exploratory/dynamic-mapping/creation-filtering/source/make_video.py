from pathlib import Path
import cv2
r=Path('/root/autodl-tmp/stage2_round16_scene02_v1');out=Path('/root/autodl-tmp/stage2_round16_gaussian_export_v1/Scene02_D0_D3_trajectory.mp4');fps=5;size=(1600,900);v=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),fps,size);assert v.isOpened()
for i in range(26):
 panels=[]
 for mode in ['D0','D3']:
  im=cv2.imread(str(r/'maps'/mode/'eval/plots'/f'{i:04d}.png'));assert im is not None;im=cv2.resize(im,(800,820),interpolation=cv2.INTER_AREA);panels.append(im)
 canvas=cv2.hconcat(panels);canvas=cv2.copyMakeBorder(canvas,80,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20));cv2.putText(canvas,f'Scene02 DEV frame {206+i} | D0 (left) vs D3 (right)',(25,50),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2);v.write(canvas)
v.release();cap=cv2.VideoCapture(str(out));print('video',out,'opened',cap.isOpened(),'frames',int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'fps',cap.get(cv2.CAP_PROP_FPS),'bytes',out.stat().st_size);assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==26

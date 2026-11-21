from pathlib import Path
p=Path(__file__).parent;s=(p/'install_seq10_o3f_states.py').read_text(encoding='utf8')
changes=[('seq10_o3f_20261116','seq03_m3f_20261116'),('KITTI10_O3F_1000','KITTI03_M3F_801'),('20261116_seq10_o3f_object_states','20261116_seq03_m3f_object_states'),('scene-config.kitti10-o3f-1000.json','scene-config.kitti03-m3f-801.json'),('sequences/10/image_2','sequences/03/image_2'),('09f8afdbe059d61d50feb92c9529276dd3947f3578de1e0c1d047009f92e2a7f','2185763e939269d6b4276fcf509cf1821f1156d714447cbe82fcdad2fffdd58b'),('kitti10_o3f_000000_000999','kitti03_m3f_000000_000800'),("RUN/'object_inputs/manifest.json'","RUN/'prepared/object_inputs/manifest.json'"),("RUN/f'frames/{frame:06d}.json'","RUN/f'prepared/frames/{frame:06d}.json'"),('abs(n-499)','abs(n-400)'),('[0,999]','[0,800]'),('frames 0-999','frames 0-800'),('1000 Mask2Former frames','801 Mask2Former frames'),('Seq 10 · 000000–000999 · O3F · Object States','Seq 03 · 000000–000800 · M3F · Object States'),('O3F semantic vehicle region','M3F semantic vehicle region'),('its O3F semantic vehicle region','its M3F semantic vehicle region'),('O3F fixed-map offline fusion','M3F fixed-map offline fusion')]
for a,b in changes:
 if a == 'its O3F semantic vehicle region' and a not in s: continue
 assert a in s,a;s=s.replace(a,b)
(p/'install_seq03_m3f_states.py').write_text(s,encoding='utf8')

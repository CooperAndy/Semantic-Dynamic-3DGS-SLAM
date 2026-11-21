from pathlib import Path
p=Path(__file__).parent
s=(p/'install_seq10_o3f_states.py').read_text(encoding='utf8')
changes=[('seq10_o3f_20261116','seq13_o3f_20261116'),('KITTI10_O3F_1000','KITTI13_O3F_260'),('20261116_seq10_o3f_object_states','20261116_seq13_o3f_object_states'),('scene-config.kitti10-o3f-1000.json','scene-config.kitti13-o3f-260.json'),('sequences/10/image_2','sequences/13/image_2'),('09f8afdbe059d61d50feb92c9529276dd3947f3578de1e0c1d047009f92e2a7f','6cebfe6efc70888706c17c443e5774e3894bd74e0d0baa6f495b51a33bb97c20'),('kitti10_o3f_000000_000999','kitti13_o3f_780_1039'),("RUN/'object_inputs/manifest.json'","RUN/'prepared/object_inputs/manifest.json'"),("RUN/f'frames/{frame:06d}.json'","RUN/f'prepared/frames/{frame:06d}.json'"),("['frames'][frame]['rgb_sha256']","['frames'][frame-780]['rgb_sha256']"),('abs(n-499)','abs(n-909)'),('[0,999]','[780,1039]'),('frames 0-999','frames 780-1039'),('1000 Mask2Former frames','260 Mask2Former frames'),('Seq 10 · 000000–000999 · O3F · Object States','Seq 13 · 000780–001039 · O3F · Object States')]
for a,b in changes:
 assert a in s,a;s=s.replace(a,b)
(p/'install_seq13_o3f_states.py').write_text(s,encoding='utf8')

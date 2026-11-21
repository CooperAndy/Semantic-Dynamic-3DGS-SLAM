from pathlib import Path
p=Path(__file__).parent;s=(p/'prepare_seq13_profile.py').read_text(encoding='utf8')
changes=[('seq13_o3f_20261116','seq03_m3f_20261116'),('KITTI13_O3F_260','KITTI03_M3F_801'),('seq13_o3f_260_delivery','seq03_m3f_801_delivery'),('seq13_o3f_object_trial','seq03_m3f_object_trial'),("sequence='13'","sequence='03'"),('kitti13_o3f_780_1039','kitti03_m3f_000000_000800'),('frame_range=[780,1039]','frame_range=[0,800]'),('Seq13 O3F','Seq03 M3F')]
for a,b in changes:assert a in s,a;s=s.replace(a,b)
(p/'prepare_seq03_profile.py').write_text(s,encoding='utf8')

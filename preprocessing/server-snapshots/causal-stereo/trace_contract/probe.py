from pathlib import Path
import sys
D=Path(__file__).parent;case=sys.argv[1];out=D/case;out.mkdir()
for i in range(3):
 for side in ['left','right']:(D/side/f'{i:06d}.png').read_bytes()
 if i==0 and case=='future':(D/'left/000002.png').read_bytes()
 if i==0 and case=='forbidden':(D/'poses/08.txt').read_text()
 (out/f'{i:06d}_depth.npy').write_bytes(b'completed output fixture')

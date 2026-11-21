"""Build the semantic legend inventory from the registered semantic PLY maps."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / '源码'
PALETTE = ('804080 f423e8 464646 66669c be9999 999999 faaa1e dcdc00 '
           '6b8e23 98fb98 4682b4 dc143c ff0000 00008e 000046 003c64 '
           '005064 0000e6 770b20 6e6e6e').split()
LOOKUP = np.array([int(color, 16) for color in PALETTE], dtype=np.uint32)


def count_colors(path):
    with path.open('rb') as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f'Incomplete PLY header: {path}')
            header.append(line.decode('ascii').strip())
            if line.strip() == b'end_header':
                offset = stream.tell()
                break
    if 'format binary_little_endian 1.0' not in header:
        raise ValueError(f'Unsupported PLY format: {path}')
    properties = [line.split()[2] for line in header if line.startswith('property float ')]
    if len(properties) != 17 or properties[6:9] != ['f_dc_0', 'f_dc_1', 'f_dc_2']:
        raise ValueError(f'Unsupported PLY layout: {path}')
    count = int(next(line.split()[2] for line in header if line.startswith('element vertex ')))
    if path.stat().st_size != offset + count * 17 * 4:
        raise ValueError(f'Unexpected PLY data size: {path}')
    data = np.memmap(path, dtype='<f4', mode='r', offset=offset, shape=(count, 17))
    counts = np.zeros(len(LOOKUP), dtype=np.int64)
    unmatched = 0
    for start in range(0, count, 500_000):
        dc = np.asarray(data[start:start + 500_000, 6:9])
        rgb = np.rint(np.clip(dc * 0.28209479177387814 + 0.5, 0, 1) * 255).astype(np.uint32)
        packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
        colors, quantities = np.unique(packed, return_counts=True)
        unmatched += int(quantities[~np.isin(colors, LOOKUP)].sum())
        for i, color in enumerate(LOOKUP):
            counts[i] += quantities[colors == color].sum()
    return count, counts, unmatched


def main():
    catalog = json.loads((CODE / 'scene-catalog.json').read_text(encoding='utf-8'))
    result = {}
    for scene in catalog['scenes']:
        config = json.loads((CODE / scene['config']).read_text(encoding='utf-8'))
        asset = config['assets'].get('semantic')
        if not asset:
            continue
        count, counts, unmatched = count_colors(ROOT / asset)
        if unmatched:
            raise ValueError(f'{asset}: {unmatched} Gaussians have unmapped colors')
        result[scene['id']] = {'asset': asset, 'gaussians': count,
            'counts': {PALETTE[i]: int(n) for i, n in enumerate(counts) if n},
            'unmatched': unmatched}
        print(f"{scene['id']}: {int((counts > 0).sum())} labels")
    (CODE / 'semantic-legend-classes.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

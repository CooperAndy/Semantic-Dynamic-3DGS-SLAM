"""Restore only this release's five backed-up files, refusing to overwrite later edits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--demo-root', required=True)
    p.add_argument('--backup-name', default='20261114_seq09_fixed_states_v1')
    a = p.parse_args(); root = Path(a.demo_root).resolve()
    parent = (root/'修改记录').resolve(); backup = (parent/a.backup_name).resolve()
    if not backup.is_relative_to(parent): raise ValueError('Backup must stay within this Demo change-record directory')
    before = json.loads((backup/'backup_manifest.json').read_text(encoding='utf-8'))['before']
    after = json.loads((backup/'installed_state.json').read_text(encoding='utf-8'))['files']
    for name, expected in after.items():
        target = (root/name).resolve(); original = (backup/name).resolve()
        if not target.is_relative_to(root) or not original.is_relative_to(backup): raise ValueError('Unsafe path')
        if sha(target) != expected: raise ValueError(f'Later edits found; restore manually: {name}')
        if sha(original) != before[name]: raise ValueError(f'Backup hash mismatch: {name}')
    for name in after: shutil.copy2(backup/name,root/name)
    print('Restored the previous seq09 TSDD entry and display code. Fixed-state assets remain as inactive experiment records.')


if __name__ == '__main__': main()

"""Check only the handoff artifacts; never run or mutate experiments."""
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
doc = root / '队友交接文档_阶段3截至05_20261010.md'
manifest = root / '交接核验清单_20261010.json'
content = doc.read_text(encoding='utf-8')
metadata = json.loads(manifest.read_text(encoding='utf-8'))
missing_links = []
for target in re.findall(r'\]\(([^)]+)\)', content):
    if '://' not in target and not target.startswith('#'):
        if not (root / target.split('#')[0].strip('<>')).exists():
            missing_links.append(target)
assert not missing_links, missing_links
for item in metadata['source_files']:
    data = (root / item['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
    assert len(data) == item['bytes'], item['path']
for item in metadata['archive_inventory']:
    path = root / item['path']
    assert path.is_file() and path.stat().st_size == item['bytes'], item['path']
assert len(re.findall(r'^## \d+\.', content, re.M)) == 17
assert '\ufffd' not in content
assert not re.search(r'(?i)(?:password|passwd)\s*[:=]\s*\S+', content)
metadata['handoff_sha256'] = hashlib.sha256(doc.read_bytes()).hexdigest()
metadata['handoff_bytes'] = doc.stat().st_size
metadata['handoff_QA'] = {
    'relative_markdown_links_exist': True,
    'source_hashes_rechecked': True,
    'archive_sizes_rechecked': True,
    'large_archive_hashes_recomputed': False,
    'numbered_sections': 17,
    'server_contacted': False,
}
manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status':'PASS', 'characters':len(content), 'bytes':doc.stat().st_size,
                  'source_files':len(metadata['source_files']), 'missing_links':missing_links}))

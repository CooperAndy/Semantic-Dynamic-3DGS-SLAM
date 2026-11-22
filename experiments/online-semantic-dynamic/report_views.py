"""Preserve the two preregistered common-GT views before raw-map cold storage."""
import hashlib,json,shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preserve(root,group,remote):
    root=Path(root);remote=Path(remote);evaluation=root/'runs'/group/'evaluation'
    assert root.parent==remote and group in ['B','T','M','TM','D','TD','MD','TMD']
    registration=json.loads((root/'registration.json').read_text())
    manifest=json.loads((root/'inputs/runtime/input_manifest.json').read_text())
    destination=remote/'report_assets'/root.name/group
    destination.mkdir(parents=True,exist_ok=False)
    rows=[]
    for logical in [19,999]:
        source=manifest['rows'][logical]['source_frame_id']
        file=evaluation/f'typical_{source:06}.png';assert file.is_file()
        target=destination/file.name;shutil.copy2(file,target)
        digest=sha(file);assert sha(target)==digest
        rows.append(dict(logical_frame=logical,source_frame=source,mode='COMMON_GT',
            original_path=file.relative_to(remote).as_posix(),path=target.relative_to(remote).as_posix(),
            bytes=target.stat().st_size,sha256=digest))
    result=dict(status='PRESERVED_PREREGISTERED_EARLY_RECENT_VIEWS',cell=root.name,
        group=group,sequence=registration['cell']['sequence'],seed=registration['cell']['seed'],
        frames=[19,999],selection='Fixed before evaluation; no best-view selection',files=rows)
    (evaluation/'publication_views_receipt.json').write_text(json.dumps(result,indent=2))
    (destination/'manifest.json').write_text(json.dumps(result,indent=2))
    return result


def collect(remote,out,cells,root_index):
    remote=Path(remote);out=Path(out);folder=out/'典型视图';folder.mkdir(exist_ok=False)
    rows=[]
    for cell in cells:
        key=f"{cell['stage']}:K{cell['sequence']}:S{cell['seed']}"
        name=root_index[key]['active_root'];directory=remote/'report_assets'/name/cell['group']
        receipt=json.loads((directory/'manifest.json').read_text())
        assert receipt['status']=='PRESERVED_PREREGISTERED_EARLY_RECENT_VIEWS'
        assert receipt['cell']==name and receipt['group']==cell['group']
        assert receipt['sequence']==cell['sequence'] and receipt['seed']==cell['seed']
        assert receipt['frames']==[19,999] and [r['logical_frame'] for r in receipt['files']]==[19,999]
        for row in receipt['files']:
            assert row['mode']=='COMMON_GT'
            source=(remote/row['path']).resolve();assert source.parent==directory.resolve()
            assert source.stat().st_size==row['bytes'] and sha(source)==row['sha256']
            target=folder/f"{name}_{cell['group']}_{row['source_frame']:06}.png"
            shutil.copy2(source,target);assert sha(target)==row['sha256']
            rows.append(dict(stage=cell['stage'],sequence=cell['sequence'],seed=cell['seed'],group=cell['group'],
                logical_frame=row['logical_frame'],source_frame=row['source_frame'],
                path=target.relative_to(out).as_posix(),sha256=row['sha256'],bytes=row['bytes'],mode=row['mode']))
    assert len(rows)==64
    (folder/'图像清单.json').write_text(json.dumps(dict(status='COMPLETE64_FIXED_COMMON_GT_VIEWS',files=rows),ensure_ascii=False,indent=2))
    return rows

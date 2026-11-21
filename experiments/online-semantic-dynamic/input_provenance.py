"""Archive frozen input identities and baseline prediction cache after sealing.

Hard links retain exact bytes without allocating another cache on the server.
Only links inside the run are eligible for normal post-ACK cleanup; the shared
online cache and all parent artifacts remain in place.
"""
import ast,hashlib,importlib.metadata,json,os,shutil,sys
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(2**20),b''):h.update(data)
    return h.hexdigest()


def capture(root,group,out,remote,references=None):
    root=Path(root);out=Path(out);remote=Path(remote)
    assert root.parent==remote
    folder=out/'input_provenance';folder.mkdir(exist_ok=False)
    files=[]
    def keep(source,name,expected=None,hardlink=False):
        source=Path(source);target=folder/name
        assert target.resolve().is_relative_to(folder.resolve()) and source.is_file()
        target.parent.mkdir(parents=True,exist_ok=True)
        digest=sha(source)
        if expected is not None:assert digest==expected,'Frozen input identity changed: '+str(source)
        linked=False
        if hardlink:
            assert source.resolve().is_relative_to(remote.resolve()) and not source.is_symlink()
            try:os.link(source,target);linked=True
            except OSError:shutil.copy2(source,target)
        else:shutil.copy2(source,target)
        assert (linked and source.stat().st_ino==target.stat().st_ino) or sha(target)==digest
        files.append(dict(path=target.relative_to(folder).as_posix(),source=str(source),
            bytes=target.stat().st_size,sha256=digest,hardlink=linked))
        return target

    manifest_path=root/'inputs/runtime/input_manifest.json'
    m=json.loads(manifest_path.read_text());assert m['GT_runtime'] is False and m['pose_source']=='ORB_estimate'
    assert len(m['rows'])==m['frame_count']
    keep(manifest_path,'runtime_manifest.json')
    config_path=root/f'source/configs/kitti/{group}.py'
    cfg=ast.literal_eval(config_path.read_text().split('=',1)[1].strip())
    keep(config_path,'executed_config.py')
    observation_path=Path(cfg['semantic']['manifest'])
    assert observation_path.resolve().is_relative_to(remote.resolve())
    observation=json.loads(observation_path.read_text());assert observation['GT_runtime'] is False
    assert observation['schema']=='s23_observations_v1'
    assert [r['frame_id'] for r in observation['rows']]==list(range(m['frame_count']))
    for row,frame in zip(observation['rows'],m['rows']):
        assert row['rgb_sha256']==frame['hashes']['rgb'] and row['source_frame']==frame['source_frame_id']
    keep(observation_path,'observation_manifest.json')
    # A baseline supplies the exact shared prediction bytes for its whole cell.
    if group=='B':
        for row in observation['rows']:
            source=observation_path.parent/row['path']
            keep(source,'observation_cache/'+Path(row['path']).name,row['sha256'],hardlink=True)
    pose=Path(m['pose_file']);assert '/dataset/poses/' not in str(pose) and pose.name!='groundtruth.txt'
    keep(pose,'frontend_ORB_poses.txt',m.get('pose_sha256'))
    frozen_path=remote/'stage2_frozen_sources.json';frozen=json.loads(frozen_path.read_text())
    keep(frozen_path,'stage2_frozen_sources.json')
    for i,row in enumerate(frozen['files']):
        source=Path(row['path'])
        if source.suffix in ['.py','.yaml','.json']:
            keep(source,f'stage2_sources/{i:03}_{source.name}',row['sha256'])

    if references is not None:
        references=Path(references);assert references.resolve().is_relative_to(remote.resolve())
        assert m['frame_count']==1000
        for name in ['reference_receipt.json','calib.txt','trajectory_reference.npz']:
            keep(references/name,'evaluation_only/'+name,hardlink=True)
        if group=='B':
            for logical in list(range(19,200,20))+list(range(819,1000,20)):
                source=m['rows'][logical]['source_frame_id']
                for name in [f'velodyne/{source:06}.bin',f'labels/{source:06}.label',f'{source:06}_GT.npz']:
                    keep(references/name,'evaluation_only/'+name,hardlink=True)
        import automatic_gt
        projection=Path(sys.modules[automatic_gt.project.__module__].__file__)
        keep(projection,'evaluation_only/actual_projection_source.py',sha(remote/'semantickitti_projection.py'))

    versions={}
    for name in ['torch','torchvision','numpy','scipy','lpips','transformers','opencv-python']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    result=dict(status='FROZEN_INPUT_PROVENANCE_AND_CACHE_CAPTURED',group=group,frames=m['frame_count'],
        baseline_prediction_cache_included=group=='B',prediction_GT=False,
        evaluation_GT_is_separate=references is not None,runtime_input_mutations=0,
        source_files=files,python=sys.version,packages=versions,
        note='Original absolute identities retained; cached predictions have the same SHA. Model weight identities are in stage2_frozen_sources.json.')
    (folder/'manifest.json').write_text(json.dumps(result,indent=2))
    return dict(status=result['status'],frames=result['frames'],files=len(files),
        baseline_prediction_cache_included=result['baseline_prediction_cache_included'],
        manifest_sha256=sha(folder/'manifest.json'))

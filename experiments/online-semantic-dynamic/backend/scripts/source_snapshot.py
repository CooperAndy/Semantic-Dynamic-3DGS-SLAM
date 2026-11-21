"""Self-contained per-group snapshot of the actually executed source and configs."""
import hashlib,json,tarfile,os
from pathlib import Path

def capture(root,out):
    root=Path(root);out=Path(out);rows=[]
    with tarfile.open(out/'source_snapshot.tar.gz','x:gz') as archive:
        for section in ['source','scripts']:
            for p in sorted((root/section).rglob('*')):
                if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
                relative=p.relative_to(root).as_posix()
                rows.append(dict(path=relative,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
                archive.add(p,arcname=relative,recursive=False)
        # New numerical candidates also retain the actual private CUDA sources
        # and extension binary, rather than only its Python call site.
        marker=root/'registration.json'
        registration=json.loads(marker.read_text()) if marker.exists() else {}
        revision=registration.get('native_numerical_revision')
        if revision:
            raster=Path(revision['raster']).resolve()
            assert os.environ.get('S23_RASTER_ROOT')==str(raster)
            import diff_gaussian_rasterization as module
            assert Path(module.__file__).resolve().is_relative_to(raster)
            for relative,expected in sorted(revision['raster_sha256'].items()):
                p=(raster/relative).resolve()
                assert p.is_relative_to(raster) and p.is_file() and not p.is_symlink()
                digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==expected
                name='private_rasterizer/'+relative
                rows.append(dict(path=name,sha256=digest,bytes=p.stat().st_size))
                archive.add(p,arcname=name,recursive=False)
    (out/'source_snapshot_manifest.json').write_text(json.dumps(dict(status='ACTUAL_EXECUTED_SOURCE_AND_CONFIGS',files=rows),indent=2))

def capture_evaluation(out):
    root=Path(__file__).resolve().parent;out=Path(out);rows=[]
    with tarfile.open(out/'evaluation_source_snapshot.tar.gz','x:gz') as archive:
        for p in sorted(root.glob('*.py')):
            rows.append(dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
            archive.add(p,arcname=p.name,recursive=False)
    (out/'evaluation_source_manifest.json').write_text(json.dumps(dict(status='ACTUAL_EVALUATOR_HELPER_SOURCES',files=rows),indent=2))

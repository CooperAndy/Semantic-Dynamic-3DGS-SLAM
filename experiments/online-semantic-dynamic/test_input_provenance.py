"""Input archive controls; tiny CPU fixtures are not experiment runs."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from input_provenance import capture


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InputArchive(unittest.TestCase):
    def fixture(self,remote):
        root=remote/'cell';out=root/'runs/B/evaluation';out.mkdir(parents=True)
        (root/'inputs/runtime').mkdir(parents=True);(root/'source/configs/kitti').mkdir(parents=True)
        observation=remote/'observations';observation.mkdir()
        rows=[];runtime=[]
        for i in range(3):
            p=observation/f'{i:06}.npz';p.write_bytes(bytes([i,10,20]))
            rows.append(dict(frame_id=i,source_frame=3000+i,rgb_sha256=str(i),path=p.name,sha256=sha(p)))
            runtime.append(dict(frame_id=i,source_frame_id=3000+i,hashes=dict(rgb=str(i))))
        (observation/'manifest.json').write_text(json.dumps(dict(schema='s23_observations_v1',GT_runtime=False,rows=rows)))
        poses=remote/'ORB.txt';poses.write_text('frozen frontend')
        (root/'inputs/runtime/input_manifest.json').write_text(json.dumps(dict(GT_runtime=False,pose_source='ORB_estimate',pose_file=str(poses),pose_sha256=sha(poses),frame_count=3,rows=runtime)))
        (root/'source/configs/kitti/B.py').write_text('config = '+repr(dict(semantic=dict(manifest=str(observation/'manifest.json')))))
        source=remote/'producer.py';source.write_text('VERSION = 1\n')
        (remote/'stage2_frozen_sources.json').write_text(json.dumps(dict(files=[dict(path=str(source),sha256=sha(source),bytes=source.stat().st_size)])))
        return root,out,observation,source

    def test_prediction_bytes_archived_without_mutating_shared_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            remote=Path(temp);root,out,observations,source=self.fixture(remote)
            originals={p.name:sha(p) for p in observations.glob('*.npz')}
            result=capture(root,'B',out,remote)
            self.assertTrue(result['baseline_prediction_cache_included'])
            receipt=json.loads((out/'input_provenance/manifest.json').read_text())
            self.assertEqual(receipt['runtime_input_mutations'],0)
            for row in receipt['source_files']:
                target=out/'input_provenance'/row['path']
                self.assertEqual(sha(target),row['sha256'])
            for p in (out/'input_provenance/observation_cache').glob('*.npz'):p.unlink()
            self.assertEqual(originals,{p.name:sha(p) for p in observations.glob('*.npz')})

    def test_reject_changed_prediction_or_pinned_producer(self):
        for changed in ['prediction','source']:
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as temp:
                remote=Path(temp);root,out,observations,source=self.fixture(remote)
                target=source if changed=='source' else observations/'000001.npz'
                target.write_bytes(b'changed')
                with self.assertRaisesRegex(AssertionError,'Frozen input identity changed'):
                    capture(root,'B',out,remote)


if __name__=='__main__':unittest.main()

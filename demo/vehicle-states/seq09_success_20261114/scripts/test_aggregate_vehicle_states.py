import unittest
from dataclasses import replace
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
from aggregate_vehicle_states import Config, decide, evaluate, load_frames, rejects, save_csv, static_hold


def observation(frame, score, depth=25, pixels=900, area=1000):
    return dict(source_frame=frame, category='car', raw_state='dynamic' if score >= .5 else 'static',
                raw_score=score, depth_z_m=depth, valid_depth_pixels=pixels,
                valid_depth_ratio=pixels/area, history_complete=True, identity_conflict=False)


class StaticHoldTests(unittest.TestCase):
    def test_late_dynamic_while_approaching_can_override_dynamic_summary(self):
        rows = [observation(i, .05, 30-i) for i in range(3)]
        rows += [observation(i, .9, 30-i) for i in range(3, 26)]
        self.assertEqual(decide(rows, Config(), False)['final_state'], 'dynamic')
        result = decide(rows, Config(), True)
        self.assertEqual(result['final_state'], 'static')
        self.assertTrue(result['static_hold']['applied'])

    def test_earlier_dynamic_blocks_static_hold(self):
        rows = [observation(0, .9, 35)] + [observation(i, .05, 30-i) for i in range(1, 5)] + [observation(i, .9, 30-i) for i in range(5, 10)]
        self.assertFalse(static_hold(rows, Config())['applied'])

    def test_single_far_static_does_not_lock(self):
        rows = [observation(0, .01, 30), observation(8, .9, 20), observation(16, .9, 10)]
        self.assertFalse(decide(rows, Config(), True)['static_hold']['applied'])

    def test_invalid_far_depth_cannot_lock(self):
        rows = [observation(i, .01, 30-i, pixels=10) for i in range(3)]
        rows += [observation(i, .9, 30-i) for i in range(3, 26)]
        self.assertFalse(decide(rows, Config(), True)['static_hold']['applied'])

    def test_moving_away_does_not_lock(self):
        rows = [observation(i, .01, 25+i) for i in range(3)] + [observation(i, .9, 25+i) for i in range(3, 10)]
        self.assertFalse(static_hold(rows, Config())['applied'])

    def test_sparse_static_observations_do_not_form_continuous_run(self):
        rows = [observation(i, .01, 30-i/10) for i in (0, 5, 10)] + [observation(i, .9, 20) for i in (15, 16)]
        self.assertFalse(static_hold(rows, Config())['applied'])

    def test_quality_failures_remain_uncertain(self):
        rows = [observation(i, .01, pixels=0) for i in range(25)]
        result = decide(rows, Config(), True)
        self.assertEqual(result['final_state'], 'uncertain')
        self.assertEqual(result['valid_count'], 0)

    def test_overlapping_window_outputs_are_not_counted_as_full_samples(self):
        rows = [observation(i, .01) for i in range(8)]
        result = decide(rows, Config(), False)
        self.assertEqual(result['sample_count'], 1)
        self.assertEqual(result['final_state'], 'uncertain')

    def test_score_band_is_not_forced_to_static(self):
        rows = [observation(i, .5) for i in (0, 8, 16)]
        self.assertEqual(decide(rows, Config(), True)['final_state'], 'uncertain')

    def test_history_and_identity_conflicts_are_excluded(self):
        row = observation(0, .01)
        row.update(history_complete=False, identity_conflict=True)
        self.assertIn('missing_history', rejects(row, Config()))
        self.assertIn('identity_conflict', rejects(row, Config()))

    def test_exact_score_and_depth_boundaries(self):
        rows = [observation(i, .4, depth=40) for i in (0, 8, 16)]
        self.assertEqual(decide(rows, Config(), False)['final_state'], 'static')
        self.assertEqual(rejects(rows[0], Config()), [])
        self.assertIn('depth_outside_gate', rejects(rows[0], replace(Config(), depth_max=30)))

    def test_reappearing_id_after_long_gap_starts_new_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n in range(300, 313):
                objs = [dict(track_id=1, category='car', bbox=[1, 1, 50, 50], score=.1, raw_state='static', depth_m=20, valid_depth_pixels=100, mask_area=100)] if n in (300, 312) else []
                frame = dict(reference_frame=n, source_map_sha256='test', window=list(range(n-7,n+1)), objects=objs, image_width=1226, image_height=370)
                (Path(tmp)/f'{n:06d}.json').write_text(json.dumps(frame))
            _, rows, _ = load_frames(tmp, Config())
            self.assertEqual([r['track_segment_id'] for r in rows], ['1:0', '1:1'])

    def test_dynamic_uncertain_counts_as_false_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            save_csv(tmp/'p.csv', [dict(config='test',track_segment_id='1:0',D_state='dynamic',E_state='uncertain')])
            save_csv(tmp/'l.csv', [dict(track_segment_id='1:0',true_state='dynamic',identity_ok='yes',true_object_id='car1',reviewer='tester',evidence_frames='300-320')])
            evaluate(SimpleNamespace(candidates=tmp/'p.csv', labels=tmp/'l.csv',output=tmp/'out'))
            result = json.loads((tmp/'out/evaluation.json').read_text())
            self.assertEqual(result[0]['dynamic_recall'], 1)
            self.assertEqual(result[1]['dynamic_recall'], 0)
            self.assertEqual(result[1]['decided_coverage'], 0)

    def test_missing_identity_or_evidence_is_not_scored_as_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            save_csv(tmp/'p.csv', [dict(config='test',track_segment_id='1:0',D_state='static',E_state='static')])
            save_csv(tmp/'l.csv', [dict(track_segment_id='1:0',true_state='static',identity_ok='',true_object_id='',reviewer='',evidence_frames='')])
            evaluate(SimpleNamespace(candidates=tmp/'p.csv', labels=tmp/'l.csv',output=tmp/'out'))
            result = json.loads((tmp/'out/evaluation.json').read_text())
            self.assertEqual(result[0]['scored_segments'], 0)
            self.assertIsNone(result[0]['decided_accuracy'])


if __name__ == '__main__': unittest.main()

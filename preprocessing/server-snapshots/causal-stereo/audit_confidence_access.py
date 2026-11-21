"""Add unfinished-open-safe access auditing to the unchanged confidence producer."""
from common import *
from audit_stereo_trace import audit_trace
import argparse

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window',required=True)
    args=parser.parse_args()
    initial=json.loads((R/'registration_cpu.json').read_text())
    assert args.window in {x['id'] for x in initial['windows']}
    W=R/'windows'/args.window;reg=json.loads((W/'registration.json').read_text())
    assert json.loads((W/'receipts/audit.json').read_text())['status']=='PASS_1000_PAST_ONLY_CONFIDENCE_INPUT_REPLAY'
    producer=(W/'scripts/producer.py').read_bytes()
    assert producer==(R/'templates/producer.py').read_bytes()
    # The locked producer saves depth_confhard last. Its completed write/close
    # seals all four output arrays before any next-frame input may be opened.
    sources={name:[Path(row['inputs'][name]['path']) for row in reg['rows']]
             for name in ['left','right','depth_left','depth_right','pose']}
    proof=audit_trace(W/'file_activity.log',sources,W/'cache/depth_confhard',1000)
    dump(W/'receipts/confidence_access_audit.json',dict(
        status='PASS_5000_CURRENT_OR_PAST_INPUT_OPENS_WITH_UNFINISHED_ENTRIES_CHECKED',
        at=now(),window=args.window,proof=proof,
        producer_sha256=sha(W/'scripts/producer.py'),trace_sha256=sha(W/'file_activity.log'),
        parent_full_output_audit_sha256=sha(W/'receipts/audit.json'),
        GPU_calls=0,GT_scoring=False,quality_adoption=False))
    print('PASS_CONFIDENCE_ACCESS',args.window,flush=True)

if __name__=='__main__':main()

"""Small isolated CUDA timing check; no SLAM input, model or map is opened."""
import json,torch
from experiment import REMOTE
from runtime_timing import RenderTimer
t=RenderTimer();x=torch.zeros((),device='cuda')
for _ in range(260):x=t.call(lambda value:value+1,x)
assert x.item()==260 and t.calls==260
report=t.flush();assert not t.events and report['cuda_render_seconds']>=0
report.update(status='PASS_ISOLATED_CUDA_TIMING_VALUE_AND_BUFFER_CONTROL',SLAM_frames=0)
(REMOTE/'timing_control_receipt.json').write_text(json.dumps(report,indent=2))
registration=dict(status='REGISTERED_BEFORE_E2_NATIVE',
    timing='CUDA render events buffered128; synchronized semantic support/gradient/fusion wall times; nested totals not added',
    failures='Report geometric qualification failures and prediction fallbacks separately; formal no-increase gate uses geometric failure counts',
    static_vehicles='Official nonmoving raw IDs10/13/16/18 on fixed support; missing prediction counts against coverage/accuracy',
    inference_timing='Formal producer process timings recorded; old DEV input generation timing requires a separately scoped measurement',
    thresholds_changed=False,GT_online=False)
assert not (REMOTE/'E2_K08_S0').exists()
(REMOTE/'E2_prelaunch_measurement_registration.json').write_text(json.dumps(registration,indent=2))
print(json.dumps(report))

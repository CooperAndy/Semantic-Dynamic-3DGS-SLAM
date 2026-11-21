"""Common engineering repair, before the six registered runs and before GT scores."""
from pathlib import Path
import ast,hashlib,json
F=Path(__file__).resolve().parents[1];B=F.parent/'CHY_正式Tracking采用验收_20261012/archive'
src=B/'source/dg_slam/dg_model.py';old=src.read_text(encoding='utf-8')
needle='''            tracking_min = self.frontend.graph.ii.min().item()
            tracking_max = self.frontend.graph.ii.max().item()'''
replacement='''            final_frame = (tstamp == self.cfg["data"]["n_img"] - 1)
            active_edges = self.frontend.graph.ii
            if active_edges.numel() == 0:
                # No active constraints: defer non-final mapping, flush all saved KFs at EOF.
                tracking_min = 0
                tracking_max = int(self.video.counter.value) - 1
                if not hasattr(self, 'chy_empty_graph_events'):
                    self.chy_empty_graph_events = []
                self.chy_empty_graph_events.append({'tstamp':int(tstamp), 'final':bool(final_frame), 'keyframes':int(self.video.counter.value)})
            else:
                tracking_min = active_edges.min().item()
                tracking_max = active_edges.max().item()
            if final_frame:
                tracking_max = int(self.video.counter.value) - 1'''
assert old.count(needle)==1
dst=F/'source/dg_slam/dg_model.py';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(old.replace(needle,replacement),encoding='utf-8');ast.parse(dst.read_text(encoding='utf-8'))
boot=F/'scripts/runtime_boot.py';text=boot.read_text(encoding='utf-8');assert "source=BASE/'source'" in text;boot.write_text(text.replace("source=BASE/'source'","source=F/'source'"),encoding='utf-8')
run=F/'scripts/run_tracking.py';text=run.read_text(encoding='utf-8');assert "(BASE/'source').resolve()" in text;text=text.replace("(BASE/'source').resolve()","(F/'source').resolve()")
text=text.replace("meta.update(status='COMPLETED',","meta.update(empty_graph_events=getattr(model,'chy_empty_graph_events',[]),status='COMPLETED',")
run.write_text(text,encoding='utf-8')
receipt={'status':'COMMON_ENGINEERING_REPAIR_BEFORE_REGISTERED_RUNS','cause':'First60-frame smoke failed at final-frame empty active-edge reduction. No GT scores read.','old_source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'new_dg_model_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'changed_scope':'Guard empty graph; do not map pending KFs on empty non-final graph, flush all video keyframes at final frame. Source common across both arms; no pose/objective/optimizer thresholds changed.','protocol_unchanged':True,'six_registered_runs_started':False}
(F/'receipts/engineering_repair_v1.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))

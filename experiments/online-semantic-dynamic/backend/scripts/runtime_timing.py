"""CUDA event totals with bounded buffers; nested totals must not be added."""
import torch

class RenderTimer:
    def __init__(self):self.events=[];self.seconds=0.;self.calls=0
    def call(self,function,*args,**kwargs):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record()
        result=function(*args,**kwargs)
        end.record();self.events.append((begin,end));self.calls+=1
        if len(self.events)>=128:self.flush()
        return result
    def flush(self):
        if self.events:
            self.events[-1][1].synchronize()
            self.seconds+=sum(a.elapsed_time(b) for a,b in self.events)/1000
            self.events.clear()
        return dict(cuda_render_seconds=self.seconds,render_calls=self.calls,
                    scope='CUDA event duration; includes semantic renders; nested in backend and semantic wall times')

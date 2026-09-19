"""CPU-only orchestration checks with clearly labeled simulated subprocesses."""
import argparse,json,subprocess,tempfile
from pathlib import Path
from unittest.mock import patch
from datetime import datetime,timezone
import run_amp_recovery_stage as stage
import analyze_amp_recovery as analysis
R=Path(__file__).resolve().parents[1]
def main():
 out=R/'results/flashmoba-amp-recovery-controller-cpu-v0';out.mkdir(exist_ok=False);checked=[]
 with tempfile.TemporaryDirectory() as td:
  for mode in ['success','failure','timeout']:
   root=Path(td)/mode;calls=[];kills=[]
   class Fake:
    def __init__(self,cmd,**kw):
     self.cmd=cmd;self.failed=False;self.killed=False
     def arg(x):return cmd[cmd.index(x)+1]
     self.phase=arg('--phase');dest=Path(arg('--output'));dest.mkdir(parents=True)
     calls.append(self.phase)
     k=int(arg('--k'));seed=int(arg('--seed'));lr=float(arg('--lr'))
     if self.phase=='report':
      lock=json.loads(Path(arg('--selection')).read_text());assert lock['selected_lrs'][str(k)]==lr
      assert calls.count('train')==12
      assert Path(arg('--resume')).exists()
     data=dict(status='complete',phase=self.phase,step=256,identity=dict(k=k,seed=seed,lr=lr,config_sha256='mock',sources={}),initial_sha256='mock',training_seconds=256,wall_seconds=260,evaluations=[dict(split='report' if self.phase=='report' else 'calibration',step=256,mean_nll=1.,values=[1.]*12)])
     (dest/'result.json').write_text(json.dumps(data));(dest/'checkpoint-256.pt').write_text('SIMULATED - NOT A CHECKPOINT')
    def wait(self,timeout=None):
     if mode=='timeout' and not self.killed:raise subprocess.TimeoutExpired(self.cmd,timeout)
     return 1 if mode=='failure' else 0
    def kill(self):self.killed=True;kills.append(True)
   try:
    with patch.object(stage.subprocess,'Popen',Fake):stage.main(argparse.Namespace(output=root,preflight_only=False))
   except SystemExit as e:assert mode!='success' and e.code==1
   result=json.loads((root/'result.json').read_text())
   if mode=='success':
    assert len(calls)==21 and calls[:3]==['preflight']*3 and calls[-6:]==['report']*6
    analysis.main(argparse.Namespace(stage=root));assert (root/'paired-analysis.json').exists()
    checked.append('selection locked before six reports; all twelve training jobs accounted; analysis completes')
   else:
    assert len(calls)==1 and result['status']=='failed'
    if mode=='timeout':assert kills
    checked.append(mode+' aborts all remaining launches')
 result=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),checks=checked,scope='Simulated CPU control-flow tests only; no model, no CUDA, no scientific outcomes.')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result))
if __name__=='__main__':main()

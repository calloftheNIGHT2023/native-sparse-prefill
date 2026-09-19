"""Finish this one already-running experiment after its training launcher exits."""
import json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def main():
    status=ROOT/'logs/zoology-length128-postprocess.json'
    assert not status.exists()
    def record(**fields):status.write_text(json.dumps(dict(utc=now(),**fields),indent=2)+'\n',encoding='utf-8')
    record(status='waiting_for_existing_training')
    while True:
        try:launch=json.loads((ROOT/'logs/zoology-length128-launch.json').read_text(encoding='utf-8'))
        except json.JSONDecodeError:time.sleep(1);continue
        if launch['status']!='running':break
        time.sleep(5)
    if launch.get('exit_code')!=0:
        record(status='training_failed_no_final_evaluation');raise SystemExit(1)
    commands=[
        ('query',[sys.executable,'scripts/probe-zoology-query.py','--run','results/zoology-length128-cpu-v0','--output','results/zoology-length128-query-v0']),
        ('source-values',[sys.executable,'scripts/probe-zoology-source-values.py']),
        ('report',[sys.executable,'scripts/report-zoology-length128.py']),
        ('finalize',[sys.executable,'scripts/finalize-zoology-length128.py'])]
    for name,command in commands:
        record(status='postprocessing',step=name)
        log=ROOT/f'logs/zoology-length128-{name}-v0.log'
        with log.open('x',encoding='utf-8') as f:result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            record(status='postprocessing_failed',step=name,exit_code=result.returncode,log=str(log));raise SystemExit(result.returncode)
    record(status='complete',training_updates_added=0)
    print(status.read_text(),flush=True)
if __name__=='__main__':main()

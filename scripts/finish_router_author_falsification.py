"""Finite immediate job chain for the registered third-fit allocation, not a scheduler."""
import json,subprocess,time,hashlib,datetime,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
    first=R/'results/router-author-falsification-resume-v0';second=R/'results/router-author-falsification-lowlr-v0'
    assert json.loads((first/'fit-result.json').read_text())['development_gate'] is False
    ledger=R/'logs/router-author-falsification-chain-v0.json';assert not ledger.exists();save(ledger,dict(utc=now(),status='waiting_for_low_lr'))
    until=time.monotonic()+1900
    while not (second/'SUCCESS.json').exists():
        if (second/'FAILURE.json').exists():raise RuntimeError('Second fit failed; no automatic retry')
        if time.monotonic()>until:raise TimeoutError('Second fit wait cap')
        time.sleep(5)
    f=json.loads((second/'fit-result.json').read_text());passed=f['development_gate']
    third_name='router-author-falsification-confirm-v0' if passed else 'router-author-falsification-lowerlr-v0';lr=f['lr'] if passed else .00046415888336127773;seed=124 if passed else 123
    third=R/'results'/third_name;assert not third.exists();log=R/'logs'/(third_name+'.log');assert not log.exists()
    cmd=[sys.executable,str(R/'src/run_router_author_falsification.py'),'--output',str(third),'--lr',str(lr),'--seed',str(seed)]
    record=dict(utc=now(),status='third_fit',second_gate=passed,second_fit_sha256=hashlib.sha256((second/'fit-result.json').read_bytes()).hexdigest(),third=third_name,lr=lr,seed=seed,command=cmd);save(ledger,record);print(json.dumps(record),flush=True)
    with log.open('xb') as h:subprocess.run(cmd,cwd=R,stdout=h,stderr=subprocess.STDOUT,check=True,timeout=1900)
    spec=[dict(name='dense_lr01_epoch19',method='dense',checkpoint='results/router-author-control-v0/lr2/checkpoint.pt'),dict(name='dense_lowlr_epoch64',method='dense',checkpoint='results/router-author-control-v0/lr1/checkpoint.pt'),dict(name='native_lr01_epoch19',method='exact_native',checkpoint='results/router-author-exact-train-v0/checkpoint.pt')]
    for d in [first,second,third]:spec.append(dict(name=d.name,method='exact_native',checkpoint=(d/'checkpoint.pt').relative_to(R).as_posix()))
    spec_path=R/'provenance/router-author-falsification-evaluation-spec-v0.json';assert not spec_path.exists();save(spec_path,spec)
    eval_dir=R/'results/router-author-falsification-evaluation-v0';record.update(utc=now(),status='evaluating');save(ledger,record)
    with (R/'logs/router-author-falsification-evaluation-v0.log').open('xb') as h:subprocess.run([sys.executable,str(R/'scripts/evaluate_router_author_falsification.py'),'--output',str(eval_dir),'--spec',str(spec_path)],cwd=R,stdout=h,stderr=subprocess.STDOUT,check=True,timeout=600)
    record.update(utc=now(),status='fits_and_gpu_evaluation_complete',cpu_replay_pending=True);save(ledger,record);print(json.dumps(record),flush=True)
if __name__=='__main__':main()

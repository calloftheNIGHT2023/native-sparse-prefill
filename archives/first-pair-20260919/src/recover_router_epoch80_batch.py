"""Recover the completed epoch80 first arm after a terminal-log formatting error."""
import argparse,json,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from zoology_entry import ROOT
from resume_joint_token_router import restore_checkpoint
from run_frozen_router import now,save,sha,evaluate,weights_sha

def verified_fit(folder):
    result=json.loads((folder/'result.json').read_text());assert result['status']=='fit_complete' and result['completed_epochs']==80 and result['global_step']==25040
    ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==79 and ck['steps']==25040 and ck['epoch_boundary']
    assert weights_sha(ck['model'])==result['final_backbone_hash']
    events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
    assert [e['global_step'] for e in events if e['event']=='optimizer_step']==list(range(12521,25041))
    assert [e['epoch'] for e in events if e['event']=='epoch']==list(range(41,81))
    return result

def main(a):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.output;source=ROOT/'results/joint-token-router-v0';plan=out/'preregistered-plan.md';env=json.loads((out/'environment.json').read_text());start=datetime.fromisoformat(env['utc'])
    test_sha=sha(out/'new-test.pt');data=torch.load(out/'new-test.pt',map_location='cpu',weights_only=True);assert json.loads((out/'data-audit.json').read_text())['seed']==2026091481
    rec=out/'recovery-source';rec.mkdir(exist_ok=False)
    for name in ['recover_router_epoch80_batch.py','resume_joint_token_router.py']:shutil.copy2(ROOT/'src'/name,rec/name)
    record=dict(utc=now(),reason='The first arm saved a full epoch80 checkpoint and fit result before event(complete, **result) duplicated utc. Only terminal logging failed. Preserve original FAILURE.json and failed-event records.',original_failure_sha256=sha(out/'FAILURE.json'),new_test_sha256=test_sha,scientific_updates_repeated=0,recovered=[])
    completed=[]
    for name in ['exact_r16_shadow','learned_r16','learned_r64']:
        folder=out/name
        if folder.exists():
            r=verified_fit(folder);record['recovered'].append(dict(method=name,checkpoint_sha256=sha(folder/'checkpoint.pt'),global_step=25040));save(out/'recovery.json',record)
            print(json.dumps(dict(utc=now(),event='recovered_completed_fit',method=name,repeated_updates=0)),flush=True)
        else:
            elapsed=(datetime.now(timezone.utc)-start).total_seconds();left=2700-elapsed
            if left<40:raise TimeoutError('Original 45-minute total wall cap reached')
            limit=min(720,int(left)-20)
            cmd=[sys.executable,'-u',str(ROOT/'src/resume_joint_token_router.py'),'--checkpoint',str(source/name/'checkpoint.pt'),'--data',str(source/'data-and-initialization.pt'),'--output',str(folder),'--plan',str(plan),'--total-epochs','80','--max-seconds',str(limit),'--device',a.device]
            print(json.dumps(dict(utc=now(),event='starting',method=name)),flush=True)
            with (out/(name+'-console.log')).open('w',encoding='utf-8') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=limit+15)
            r=verified_fit(folder)
        r['method']=name;completed.append(r);save(out/'progress.json',dict(utc=now(),status='fitting',completed=[x['method'] for x in completed]))
    assert sha(out/'new-test.pt')==test_sha
    for r in completed:
        ck=torch.load(out/r['method']/'checkpoint.pt',map_location='cpu',weights_only=False);m,ix,route,opt,iopt,sch=restore_checkpoint(ck,a.device);m.eval();route.collect_aux=False
        r.update(fresh=evaluate(m,data['fresh'],a.device),swapped=evaluate(m,data['swapped'],a.device));route.restore();r['status']='complete';save(out/r['method']/'final-evaluation.json',r)
    control=completed[0];valid=all(control[k]['accuracy']>=.99 for k in ['fresh','swapped'])
    for r in completed[1:]:r['epoch80_gate']=valid and all(r[k]['accuracy']>=.99 and r[k]['accuracy']>=control[k]['accuracy']-.01 for k in ['fresh','swapped']);save(out/r['method']/'final-evaluation.json',r)
    result=dict(status='complete',started_utc=env['utc'],finished_utc=now(),wall_seconds=(datetime.now(timezone.utc)-start).total_seconds(),fit_wall_seconds_sum=sum(x['wall_seconds'] for x in completed),recovered_terminal_logging_error=True,scientific_updates_repeated=0,additional_backbone_updates=37560,additional_indexer_updates=37560,additional_input_tokens=76800000,additional_supervised_answers=4800000,control_valid=valid,runs=completed,scope='Conditional epoch40-to80 continuation on a new A40, single seed. Original first-arm terminal log error preserved; all updates/checkpoints verified before continuing.')
    save(out/'result.json',result);save(out/'SUCCESS.json',dict(utc=now(),recovered_logging_error=True));save(out/'progress.json',dict(utc=now(),status='complete'));save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(dict(status='complete',runs=[dict(method=x['method'],fresh=x['fresh']['accuracy'],swapped=x['swapped']['accuracy'],gate=x.get('epoch80_gate')) for x in completed])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    try:main(a)
    except Exception:
        import traceback
        save(a.output/'RECOVERY_FAILURE.json',dict(utc=now(),traceback=traceback.format_exc()));raise

"""Collect fixed early-stop controls without launching any training."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];STAGE='fresh-book256-resume'
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    c=json.loads((R/'logs/cloud-connection-current.json').read_text(encoding='utf-8'));host=c['ssh_user']+'@'+c['ssh_host'];root=c['root'];opts=['-i',c['ssh_key_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15'];ssh=['ssh',*opts,'-p',str(c['ssh_port']),host];scp=['scp',*opts,'-P',str(c['ssh_port'])]
    state=R/f'logs/{STAGE}-v0-local-collector.json';start=time.monotonic();errors=0
    try:
        while time.monotonic()-start<4500:
            code="from pathlib import Path\nimport json\nr=Path("+repr(root)+")\nf=r/'logs/fresh-book256-resume-queue-v0.json'\nprint(json.dumps(dict(queue=json.loads(f.read_text()) if f.exists() else None,archive_ready=(r/'exports/fresh-book256-resume-evidence-v0.tar.json').exists())))\n"
            try:
                q=subprocess.run(ssh+['python3','-'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=40);assert q.returncode==0,q.stderr[-1000:];v=json.loads(q.stdout);errors=0
            except Exception:
                errors+=1
                if errors>=5:raise
                save(state,dict(status='connection_retry',utc=utc(),errors=errors));time.sleep(30);continue
            save(state,dict(status='waiting',utc=utc(),cloud=v))
            if v['archive_ready']:break
            time.sleep(30)
        else:raise RuntimeError('Collector timeout: inspect existing stage before retrying')
        for ext in ['json','gz']:
            n=f'{STAGE}-evidence-v0.tar.{ext}';subprocess.run(scp+[host+':'+root+'/exports/'+n,str(R/'exports'/n)],check=True,timeout=300)
        with (R/f'logs/{STAGE}-v0-audit.log').open('w',encoding='utf-8') as f:subprocess.run([sys.executable,str(R/'scripts/report_fresh_book256_resume_v0.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((R/f'results/{STAGE}-audit-v0/result.json').read_text(encoding='utf-8'));assert result['status']=='verified'
        c=json.loads((R/'logs/control-state.json').read_text(encoding='utf-8'));already=c.get('fresh_book256_resume',{}).get('status')=='complete_verified'
        if not already:
            assert c['cumulative_task_predictions']==53382
            c['cumulative_task_predictions']+=result['new_task_predictions']+result['diagnostic_replay_predictions']
        c.update(status='fresh_book256_resume_v0_complete_verified',updated_utc=utc(),active_evaluation_jobs=0,next_action='Inspect completed fixed fresh-book256 base gate and three contrasts. Combine with immutable64/128 natural-text and cost audits, preserving scopes. If noninferiority is supported, the remaining requirement is a specific original contribution and replication beyond two seeds/one0.5BLoRA model. Do not announce a paper or launch broad training automatically. If base gate fails, report task invalidity rather than sparse failure, and use an independently motivated task. Avoid idle GPU; preserve backups before any stop.')
        c['fresh_book256_resume']=dict(status='complete_verified',report='docs/fresh-book256-resume-results-2026-09-17.md',task_predictions=result['task_predictions'],new_task_predictions=result['new_task_predictions'],diagnostic_replay_predictions=result['diagnostic_replay_predictions'],nll_forwards=result['nll_forwards'],base_gate=result['base_gate'],contrasts=result['contrasts'],archive=result['archive']);c['fresh_book256']['status']='complete_via_verified_resume';save(R/'logs/control-state.json',c)
        cost=dict(utc=utc(),seconds=result['control']['seconds'],hourly_gpu_usd=.74,estimated_gpu_usd=result['control']['seconds']/3600*.74,scope='Controller wall only, excludes setup idle storage; not invoice.',optimizer_updates=0,new_task_predictions=result['new_task_predictions'],diagnostic_replay_predictions=result['diagnostic_replay_predictions'],nll_forwards=result['nll_forwards']);save(R/f'logs/{STAGE}-v0-cost-estimate.json',cost)
        (R/'STATE.md').write_text('# 当前状态：跨Pod续跑与256本新背景确认已核验\n\n'+utc()+'。主目标仍是稀疏继续训练成本与接近密集质量。\n\n报告docs/fresh-book256-resume-results-2026-09-17.md；基座门槛='+json.dumps(result['base_gate'])+'；三项比较='+json.dumps(result['contrasts'])+'。固定旧模板的新书背景，仅条件于两个旧训练种子，不是原创方法证明。\n\n此前同64步恢复省时7.29%/7.14%，同128步省时7.73%/7.62%。累计6784科学更新、299诊断更新、'+str(c['cumulative_task_predictions'])+'记录预测；本轮0更新，'+str(result['new_task_predictions'])+'新增预测及2条诊断重放。历史失败单独保留。\n\n下一步：'+c['next_action']+'\n\nPod运行，heartbeat ACTIVE。\n',encoding='utf-8')
        if not already:
            with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：256本新背景评测完成核验\n\n0训练更新，'+str(result['new_task_predictions'])+'新增预测及2条诊断重放，'+str(result['nll_forwards'])+'校准NLL。基座门槛与三个配对比较提前冻结，98.333%区间与5pp容差不变。逐题结果、恢复参数校验、原始来源、哈希和费用留存。\n')
        for n in ['STATE.md','TIMELINE.md','logs/control-state.json','docs/fresh-book256-resume-results-2026-09-17.md',f'logs/{STAGE}-v0-cost-estimate.json']:subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
        save(state,dict(status='complete_verified',utc=utc(),report='docs/fresh-book256-resume-results-2026-09-17.md'))
    except Exception:
        save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),scope='Do not duplicate launch or change frozen protocol.'));raise
if __name__=='__main__':main()

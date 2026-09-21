"""Collect once, verify evidence, save counts/cost; heartbeat selects next work."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    c=json.loads((R/'logs/cloud-connection-current.json').read_text(encoding='utf-8'));host=c['ssh_user']+'@'+c['ssh_host'];root=c['root']
    opts=['-i',c['ssh_key_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15']
    ssh=['ssh',*opts,'-p',str(c['ssh_port']),host];scp=['scp',*opts,'-P',str(c['ssh_port'])]
    state=R/'logs/word-route-v0-local-collector.json';start=time.monotonic();errors=0
    try:
        while time.monotonic()-start<3300:
            code="from pathlib import Path\nimport json\nr=Path("+repr(root)+")\nf=r/'logs/word-route-queue-v0.json'\nprint(json.dumps(dict(queue=json.loads(f.read_text()) if f.exists() else None,archive_ready=(r/'exports/word-route-evidence-v0.tar.json').exists())))\n"
            try:
                p=subprocess.run(ssh+['python3','-'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=40)
                assert p.returncode==0,p.stderr[-1000:];v=json.loads(p.stdout);errors=0
            except Exception:
                errors+=1
                if errors>=5:raise
                save(state,dict(status='connection_retry',utc=utc(),errors=errors));time.sleep(30);continue
            save(state,dict(status='waiting',utc=utc(),cloud=v))
            if v['archive_ready']:break
            time.sleep(30)
        else:raise RuntimeError('Collector timeout; inspect existing cloud process before retrying')
        for n in ['word-route-evidence-v0.tar.json','word-route-evidence-v0.tar.gz']:
            subprocess.run(scp+[host+':'+root+'/exports/'+n,str(R/'exports'/n)],check=True,timeout=300)
        with (R/'logs/word-route-v0-audit.log').open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,str(R/'scripts/report_word_route_v0.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((R/'results/word-route-audit-v0/result.json').read_text(encoding='utf-8'));assert result['status']=='verified'
        c=json.loads((R/'logs/control-state.json').read_text(encoding='utf-8'))
        already=c.get('word_route',{}).get('status')=='complete_verified'
        if not already:
            assert c['cumulative_task_predictions']==30840
            c['cumulative_task_predictions']+=384
        c.update(status='word_route_v0_complete_verified',updated_utc=utc(),active_evaluation_jobs=0,next_action='Review all fixed-budget target/sham paired contrasts; choose a bounded next experiment based on both seed results, preserving prior negative RACE evidence and no deployable method claim from privileged routing.')
        c['word_route'].update(status='complete_verified',report='docs/word-route-results-2026-09-16.md',archive=result['archive']);save(R/'logs/control-state.json',c)
        cost=dict(utc=utc(),seconds=result['control']['seconds'],hourly_gpu_usd=.74,estimated_gpu_usd=result['control']['seconds']/3600*.74,scope='Controller wall time only; excludes setup, idle, storage. Not a provider invoice.',optimizer_updates=0,task_predictions=384)
        save(R/'logs/word-route-v0-cost-estimate.json',cost)
        (R/'STATE.md').write_text('# 当前状态：事实词固定预算路由干预完成并核验\n\n'+utc()+'。新增384次计分前向、0训练更新。累计5760科学更新、235诊断更新、31224已记录任务前向，另1次失败未完整记录。\n\n结果见docs/word-route-results-2026-09-16.md。\n\n下一步：'+c['next_action']+'\n\nPod仍运行，连接见logs/cloud-connection-current.json；heartbeat ACTIVE。\n',encoding='utf-8')
        if not already:
            with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：事实词固定预算路由干预完成核验\n\n六个作业，384次前向，0训练。结果见docs/word-route-results-2026-09-16.md，计时与费用估算见logs/word-route-v0-cost-estimate.json。\n')
        for n in ['STATE.md','TIMELINE.md','logs/control-state.json','docs/word-route-results-2026-09-16.md','logs/word-route-v0-cost-estimate.json']:
            subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
        save(state,dict(status='complete_verified',utc=utc(),report='docs/word-route-results-2026-09-16.md'))
    except Exception:
        save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),scope='No duplicate launch. Preserve archive and inspect cloud state.'));raise
if __name__=='__main__':main()

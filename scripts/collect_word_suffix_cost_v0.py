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
    state=R/'logs/word-suffix-cost-v0-local-collector.json';start=time.monotonic();errors=0
    try:
        while time.monotonic()-start<2100:
            code="from pathlib import Path\nimport json\nr=Path("+repr(root)+")\nf=r/'logs/word-suffix-cost-queue-v0.json'\nprint(json.dumps(dict(queue=json.loads(f.read_text()) if f.exists() else None,archive_ready=(r/'exports/word-suffix-cost-evidence-v0.tar.json').exists())))\n"
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
        for n in ['word-suffix-cost-evidence-v0.tar.json','word-suffix-cost-evidence-v0.tar.gz']:
            subprocess.run(scp+[host+':'+root+'/exports/'+n,str(R/'exports'/n)],check=True,timeout=300)
        with (R/'logs/word-suffix-cost-v0-audit.log').open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,str(R/'scripts/report_word_suffix_cost_v0.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((R/'results/word-suffix-cost-audit-v0/result.json').read_text(encoding='utf-8'));assert result['status']=='verified'
        c=json.loads((R/'logs/control-state.json').read_text(encoding='utf-8'))
        already=c.get('word_suffix_cost',{}).get('status')=='complete_verified'
        if not already:
            assert c['cumulative_task_predictions']==31480
            c['cumulative_task_predictions']+=300
        c.update(status='word_suffix_cost_v0_complete_verified',updated_utc=utc(),active_evaluation_jobs=0,next_action='Review all interleaved cache-producing prefill timings and existing quality gaps. If cost screen passes, design fresh capability-gated quality validation before further training; otherwise inspect measured bottlenecks. No novelty, equivalence or training-speed claim.')
        c['word_suffix_cost'].update(status='complete_verified',report='docs/word-suffix-cost-results-2026-09-16.md',archive=result['archive']);save(R/'logs/control-state.json',c)
        cost=dict(utc=utc(),seconds=result['control']['seconds'],hourly_gpu_usd=.74,estimated_gpu_usd=result['control']['seconds']/3600*.74,scope='Controller wall time only; excludes setup, idle, storage. Not a provider invoice.',optimizer_updates=0,task_predictions=300)
        save(R/'logs/word-suffix-cost-v0-cost-estimate.json',cost)
        (R/'STATE.md').write_text('# 当前状态：密集问题后缀整模型prefill计时完成并核验\n\n'+utc()+'。新增300次计分前向、0训练更新。累计5760科学更新、235诊断更新、31780已记录任务前向，另1次失败未完整记录。\n\n结果见docs/word-suffix-cost-results-2026-09-16.md。\n\n下一步：'+c['next_action']+'\n\nPod仍运行，连接见logs/cloud-connection-current.json；heartbeat ACTIVE。\n',encoding='utf-8')
        if not already:
            with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：密集问题后缀整模型prefill计时完成核验\n\n四个作业，300次前向，0训练。结果见docs/word-suffix-cost-results-2026-09-16.md，计时与费用估算见logs/word-suffix-cost-v0-cost-estimate.json。\n')
        for n in ['STATE.md','TIMELINE.md','logs/control-state.json','docs/word-suffix-cost-results-2026-09-16.md','logs/word-suffix-cost-v0-cost-estimate.json']:
            subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
        save(state,dict(status='complete_verified',utc=utc(),report='docs/word-suffix-cost-results-2026-09-16.md'))
    except Exception:
        save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),scope='No duplicate launch. Preserve archive and inspect cloud state.'));raise
if __name__=='__main__':main()

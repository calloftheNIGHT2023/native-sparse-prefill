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
    state=R/'logs/full-step-cost-v0-local-collector.json';start=time.monotonic();errors=0
    try:
        while time.monotonic()-start<2100:
            code="from pathlib import Path\nimport json\nr=Path("+repr(root)+")\nf=r/'logs/full-step-cost-queue-v0.json'\nprint(json.dumps(dict(queue=json.loads(f.read_text()) if f.exists() else None,archive_ready=(r/'exports/full-step-cost-evidence-v0.tar.json').exists())))\n"
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
        for n in ['full-step-cost-evidence-v0.tar.json','full-step-cost-evidence-v0.tar.gz']:
            subprocess.run(scp+[host+':'+root+'/exports/'+n,str(R/'exports'/n)],check=True,timeout=300)
        with (R/'logs/full-step-cost-v0-audit.log').open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,str(R/'scripts/report_full_step_cost_v0.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((R/'results/full-step-cost-audit-v0/result.json').read_text(encoding='utf-8'));assert result['status']=='verified'
        c=json.loads((R/'logs/control-state.json').read_text(encoding='utf-8'))
        already=c.get('full_step_cost',{}).get('status')=='complete_verified'
        c.update(status='full_step_cost_v0_complete_verified',updated_utc=utc(),active_evaluation_jobs=0,active_benchmark_jobs=0,next_action='Combine common-dense NLL trajectory and factual quality gap with newly verified full optimizer-step costs. Primary lower training cost at similar quality; inference speed is not required. Use results to freeze next bounded quality-improvement/independent-validation batch, preserving failed outcomes and budget. Known suffix protection is not an original method.')
        c['queued_full_step_cost']['status']='complete_verified'
        c['full_step_cost'].update(status='complete_verified',report='docs/full-step-cost-results-2026-09-16.md',diagnostic_updates=64,archive=result['archive'])
        # Retain existing accounting keys; this stage does not change scientific or task counts.
        if not already:
            assert c['cumulative_diagnostic_updates']==235
            c['cumulative_diagnostic_updates']+=64
            c['full_step_cost']['cumulative_diagnostic_updates_after']=299
        save(R/'logs/control-state.json',c)
        cost=dict(utc=utc(),seconds=result['control']['seconds'],hourly_gpu_usd=.74,estimated_gpu_usd=result['control']['seconds']/3600*.74,scope='Controller wall only; excludes setup, idle, storage. Not an invoice.',optimizer_updates=64,scientific_updates=0,diagnostic_updates=64,task_predictions=0)
        save(R/'logs/full-step-cost-v0-cost-estimate.json',cost)
        (R/'STATE.md').write_text('# 当前状态：完整训练步成本已核验\n\n'+utc()+'。立项保持稀疏继续预训练省成本、接近质量，不要求推理加速。\n\n统一密集评测：docs/common-dense-quality-results-2026-09-16.md；完整训练步成本：docs/full-step-cost-results-2026-09-16.md。此轮从同一密集检查点分叉进行64次诊断更新，不能充当全程稀疏训练的质量证据。\n\n累计5760科学更新、299诊断更新、31786任务预测；另18次无更新梯度计算、279次此次统一NLL前向和1次失败未完整记录。\n\n下一步：'+c['next_action']+'\n\nPod运行，heartbeat ACTIVE。\n',encoding='utf-8')
        if not already:
            with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：完整训练步成本完成核验\n\n两种子各从同一检查点分叉，16步乘4作业，共64次诊断更新，正式科学更新不变。逐步日志、初末检查点、UTC和费用估算均保存。报告docs/full-step-cost-results-2026-09-16.md。\n')
        for n in ['STATE.md','TIMELINE.md','logs/control-state.json','docs/full-step-cost-results-2026-09-16.md','logs/full-step-cost-v0-cost-estimate.json']:
            subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
        save(state,dict(status='complete_verified',utc=utc(),report='docs/full-step-cost-results-2026-09-16.md'))

    except Exception:
        save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),scope='No duplicate launch. Preserve archive and inspect cloud state.'));raise
if __name__=='__main__':main()

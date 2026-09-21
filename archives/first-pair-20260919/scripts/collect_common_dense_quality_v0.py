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
    state=R/'logs/common-dense-quality-v0-local-collector.json';start=time.monotonic();errors=0
    try:
        while time.monotonic()-start<2100:
            code="from pathlib import Path\nimport json\nr=Path("+repr(root)+")\nf=r/'logs/common-dense-quality-queue-v0.json'\nprint(json.dumps(dict(queue=json.loads(f.read_text()) if f.exists() else None,archive_ready=(r/'exports/common-dense-quality-evidence-v0.tar.json').exists())))\n"
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
        for n in ['common-dense-quality-evidence-v0.tar.json','common-dense-quality-evidence-v0.tar.gz']:
            subprocess.run(scp+[host+':'+root+'/exports/'+n,str(R/'exports'/n)],check=True,timeout=300)
        with (R/'logs/common-dense-quality-v0-audit.log').open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,str(R/'scripts/report_common_dense_quality_v0.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((R/'results/common-dense-quality-audit-v0/result.json').read_text(encoding='utf-8'));assert result['status']=='verified'
        c=json.loads((R/'logs/control-state.json').read_text(encoding='utf-8'))
        already=c.get('common_dense_quality',{}).get('status')=='complete_verified'
        c.update(status='common_dense_quality_v0_complete_verified',updated_utc=utc(),active_evaluation_jobs=0,next_action='Review common-dense NLL and existing common-dense factual gap together. Primary remains lower training cost at comparable common-evaluation quality; inference speed is not required. Decide next bounded real training or fresh quality validation from these results. Do not select favorable metrics or treat exposed windows as independent confirmation.')
        c['common_dense_quality'].update(status='complete_verified',report='docs/common-dense-quality-results-2026-09-16.md',nll_forwards=279,archive=result['archive']);save(R/'logs/control-state.json',c)
        cost=dict(utc=utc(),seconds=result['control']['seconds'],hourly_gpu_usd=.74,estimated_gpu_usd=result['control']['seconds']/3600*.74,scope='Controller wall only; excludes setup, idle, storage. Not an invoice.',optimizer_updates=0,task_predictions=0,nll_forwards=279)
        save(R/'logs/common-dense-quality-v0-cost-estimate.json',cost)
        (R/'STATE.md').write_text('# 当前状态：统一密集注意力评测已核验\n\n'+utc()+'。原立项不变：稀疏继续预训练以更少训练成本获得接近质量。推理速度不是成败门槛。\n\n结果见docs/common-dense-quality-results-2026-09-16.md；9个旧报告窗口属于开发诊断。此次0参数更新、0任务预测、279次NLL前向。累计5760科学更新、235诊断更新、31786任务预测；另18次无更新梯度计算和1次失败未完整记录。\n\n下一步：'+c['next_action']+'\n\nPod运行，heartbeat ACTIVE。\n',encoding='utf-8')
        if not already:
            with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：统一密集注意力NLL完成核验\n\n9个检查点、279次NLL前向，0更新。来源、逐窗口结果、UTC时间和费用估算均已留存。报告docs/common-dense-quality-results-2026-09-16.md。\n')
        for n in ['STATE.md','TIMELINE.md','logs/control-state.json','docs/common-dense-quality-results-2026-09-16.md','logs/common-dense-quality-v0-cost-estimate.json']:
            subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
        save(state,dict(status='complete_verified',utc=utc(),report='docs/common-dense-quality-results-2026-09-16.md'))

    except Exception:
        save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),scope='No duplicate launch. Preserve archive and inspect cloud state.'));raise
if __name__=='__main__':main()

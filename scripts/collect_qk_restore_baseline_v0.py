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
    state=R/'logs/qk-restore-baseline-v0-local-collector.json';start=time.monotonic();errors=0
    try:
        while time.monotonic()-start<2100:
            code="from pathlib import Path\nimport json\nr=Path("+repr(root)+")\nf=r/'logs/qk-restore-baseline-queue-v0.json'\nprint(json.dumps(dict(queue=json.loads(f.read_text()) if f.exists() else None,archive_ready=(r/'exports/qk-restore-baseline-evidence-v0.tar.json').exists())))\n"
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
        for n in ['qk-restore-baseline-evidence-v0.tar.json','qk-restore-baseline-evidence-v0.tar.gz']:
            subprocess.run(scp+[host+':'+root+'/exports/'+n,str(R/'exports'/n)],check=True,timeout=300)
        with (R/'logs/qk-restore-baseline-v0-audit.log').open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,str(R/'scripts/report_qk_restore_baseline_v0.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((R/'results/qk-restore-baseline-audit-v0/result.json').read_text(encoding='utf-8'));assert result['status']=='verified'
        c=json.loads((R/'logs/control-state.json').read_text(encoding='utf-8'))
        c.setdefault('qk_restore_baseline',{})
        already=c.get('qk_restore_baseline',{}).get('status')=='complete_verified'
        if not already:
            assert c['cumulative_task_predictions']==32451
            c['cumulative_task_predictions']+=768
        c.update(status='qk_restore_baseline_v0_complete_verified',updated_utc=utc(),active_evaluation_jobs=0,next_action='Review existing QK-Restore control effects on both dense and sparse checkpoints, along with NLL tradeoff. This is a known baseline, not novelty. Do not extend a generic QK-forgetting claim already covered by Attention Amnesia. Decide a distinct falsifiable sparse-training hypothesis or a carefully bounded actual-training comparison, screening prior work before allocating. Primary is comparable quality at lower training cost, not inference acceleration. Avoid repeated uninformative evaluations or broad searches while GPU idles.')
        c['qk_restore_baseline'].update(status='complete_verified',report='docs/qk-restore-baseline-results-2026-09-16.md',task_predictions=768,effects=result['effects'],interaction_pp=result['interaction_pp'],archive=result['archive']);save(R/'logs/control-state.json',c)
        cost=dict(utc=utc(),seconds=result['control']['seconds'],hourly_gpu_usd=.74,estimated_gpu_usd=result['control']['seconds']/3600*.74,scope='Controller wall only; excludes setup, idle, storage. Not an invoice.',optimizer_updates=0,task_predictions=768,nll_forwards=52)
        save(R/'logs/qk-restore-baseline-v0-cost-estimate.json',cost)
        (R/'STATE.md').write_text('# 当前状态：已有QK-Restore基线诊断已核验\n\n'+utc()+'。主线仍是训练省成本、质量接近密集，不要求推理加速。\n\n报告docs/qk-restore-baseline-results-2026-09-16.md；已有方法出处和贡献重合见docs/qk-restore-overlap-2026-09-16.md。本轮只是诊断，不能作为新方法。\n\n累计5760科学更新、299诊断更新、33219已记录任务预测；本轮另52次NLL前向、0更新。完整训练步此前省时5.17–7.53%，普通文本PPL高4.03%；新背景事实检查稀疏78.12%、密集85.94%，尚未通过5pp非劣标准。\n\n下一步：'+c['next_action']+'\n\nPod运行，heartbeat ACTIVE。\n',encoding='utf-8')
        if not already:
            with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：已有QK-Restore基线完成核验\n\n4个固定检查点、旧开发任务768次预测和52次NLL前向，0训练。密集和稀疏训练都包含，逐参数撤销范围已核验。恢复QK已有明确前作；不将其包装为原创贡献。报告docs/qk-restore-baseline-results-2026-09-16.md。\n')
        for n in ['STATE.md','TIMELINE.md','logs/control-state.json','docs/qk-restore-baseline-results-2026-09-16.md','logs/qk-restore-baseline-v0-cost-estimate.json']:
            subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
        save(state,dict(status='complete_verified',utc=utc(),report='docs/qk-restore-baseline-results-2026-09-16.md'))

    except Exception:
        save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),scope='No duplicate launch. Preserve archive and inspect cloud state.'));raise
if __name__=='__main__':main()

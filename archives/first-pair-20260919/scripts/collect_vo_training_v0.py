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
    state=R/'logs/vo-training-v0-local-collector.json';start=time.monotonic();errors=0
    try:
        while time.monotonic()-start<4500:
            code="from pathlib import Path\nimport json\nr=Path("+repr(root)+")\nf=r/'logs/vo-training-queue-v0.json'\nprint(json.dumps(dict(queue=json.loads(f.read_text()) if f.exists() else None,archive_ready=(r/'exports/vo-training-evidence-v0.tar.json').exists())))\n"
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
        for n in ['vo-training-evidence-v0.tar.json','vo-training-evidence-v0.tar.gz']:
            subprocess.run(scp+[host+':'+root+'/exports/'+n,str(R/'exports'/n)],check=True,timeout=300)
        with (R/'logs/vo-training-v0-audit.log').open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,str(R/'scripts/report_vo_training_v0.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((R/'results/vo-training-audit-v0/result.json').read_text(encoding='utf-8'));assert result['status']=='verified'
        c=json.loads((R/'logs/control-state.json').read_text(encoding='utf-8'));c.setdefault('vo_training',{})
        already=c['vo_training'].get('status')=='complete_verified'
        if not already:
            assert c['cumulative_task_predictions']==33219 and c['cumulative_scientific_updates']==5760
            c['cumulative_task_predictions']+=768;c['cumulative_scientific_updates']+=512
        c.update(status='vo_training_v0_complete_verified',updated_utc=utc(),active_evaluation_jobs=0,active_training_jobs=0,next_action='Review all4 matched V/O-only training runs at fixed128 endpoint, full-step and loop costs, olddevelopment NLL and factual quality. This is a controlled adaptation screen, not a novel method or independent confirmation. If jointly promising, freeze external natural-quality/new-seed confirmation without outcome-based checkpoint choice. If it fails, preserve mechanism and cost evidence; do not extend same setup without a different falsifiable hypothesis. QK-Restore/SSA/LongLoRA overlap must bound contribution claims. Primary lower training cost at comparable common-evaluation quality; inference speed not required.')
        c['vo_training'].update(status='complete_verified',report='docs/vo-training-results-2026-09-17.md',scientific_updates=512,task_predictions=768,gradient_passes=8,quality=result['quality'],cost=result['cost'],development_screen=result['development_screen'],archive=result['archive']);save(R/'logs/control-state.json',c)
        cost=dict(utc=utc(),seconds=result['control']['seconds'],hourly_gpu_usd=.74,estimated_gpu_usd=result['control']['seconds']/3600*.74,scope='Controller wall only; excludes setup, idle, storage. Not an invoice.',optimizer_updates=512,scientific_updates=512,task_predictions=768,gradient_passes=8)
        save(R/'logs/vo-training-v0-cost-estimate.json',cost)
        (R/'STATE.md').write_text('# 当前状态：从头固定Q/K的V/O继续训练已核验\n\n'+utc()+'。主线保持训练省成本、质量接近；不要求推理加速。\n\n报告docs/vo-training-results-2026-09-17.md。4个匹配作业各128步，所有Q/K张量在0/64/128检查点均与初始值相同，64步重载检查通过。开发筛查='+json.dumps(result['development_screen'],ensure_ascii=False)+'。\n\n累计6272科学更新、299诊断更新、33987已记录任务预测；本轮另8次无更新梯度检查。不把旧开发数据结果充当独立确认或新方法。\n\n下一步：'+c['next_action']+'\n\nPod运行，heartbeat ACTIVE。\n',encoding='utf-8')
        if not already:
            with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：固定Q/K、训练V/O的4作业完成核验\n\n新增512科学更新、768任务预测、8次无更新梯度检查。保存逐步耗时、0/64/128检查点、校准/质量结果、UTC和费用。报告docs/vo-training-results-2026-09-17.md。\n')
        for n in ['STATE.md','TIMELINE.md','logs/control-state.json','docs/vo-training-results-2026-09-17.md','logs/vo-training-v0-cost-estimate.json']:
            subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
        save(state,dict(status='complete_verified',utc=utc(),report='docs/vo-training-results-2026-09-17.md'))

    except Exception:
        save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),scope='No duplicate launch. Preserve archive and inspect cloud state.'));raise
if __name__=='__main__':main()

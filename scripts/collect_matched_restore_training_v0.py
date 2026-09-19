"""Collect complete training evidence once; keep cumulative science counts separate."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];STAGE='matched-restore-training'
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    c=json.loads((R/'logs/cloud-connection-current.json').read_text(encoding='utf-8'));host=c['ssh_user']+'@'+c['ssh_host'];root=c['root'];opts=['-i',c['ssh_key_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15'];ssh=['ssh',*opts,'-p',str(c['ssh_port']),host];scp=['scp',*opts,'-P',str(c['ssh_port'])]
    state=R/f'logs/{STAGE}-v0-local-collector.json';start=time.monotonic();errors=0
    try:
        while time.monotonic()-start<5400:
            code="from pathlib import Path\nimport json\nr=Path("+repr(root)+")\nf=r/'logs/matched-restore-training-queue-v0.json'\nprint(json.dumps(dict(queue=json.loads(f.read_text()) if f.exists() else None,archive_ready=(r/'exports/matched-restore-training-evidence-v0.tar.json').exists())))\n"
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
        with (R/f'logs/{STAGE}-v0-audit.log').open('w',encoding='utf-8') as f:subprocess.run([sys.executable,str(R/'scripts/report_matched_restore_training_v0.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((R/f'results/{STAGE}-audit-v0/result.json').read_text(encoding='utf-8'));assert result['status']=='verified'
        c=json.loads((R/'logs/control-state.json').read_text(encoding='utf-8'));already=c.get('matched_restore_training',{}).get('status')=='complete_verified'
        if not already:
            assert c['cumulative_task_predictions']==34519 and c['cumulative_scientific_updates']==6272
            c['cumulative_task_predictions']+=1064;c['cumulative_scientific_updates']+=512
        c.update(status='matched_restore_training_v0_complete_verified',updated_utc=utc(),active_evaluation_jobs=0,active_training_jobs=0,next_action='Audit matched full training+restored artifact cost against dense original and all symmetric quality comparisons. If replication screen passes, next priority is a genuinely new natural task and independent seed/prospective design with prior-art limits; this is knownQKRestore on Qwen0.5B LoRA, not automatically paper-ready. If it fails, retain results and select a distinct informative hypothesis without re-running same recipe. Preserve original training-cost/comparable-quality goal; no inference-speed requirement.')
        c['matched_restore_training']=dict(status='complete_verified',report='docs/matched-restore-training-results-2026-09-17.md',scientific_updates=512,task_predictions=1064,nll_forwards=504,gradient_passes=8,replication_screen=result['replication_screen'],cost=result['cost'],contrasts=result['contrasts'],archive=result['archive']);save(R/'logs/control-state.json',c)
        cost=dict(utc=utc(),seconds=result['control']['seconds'],hourly_gpu_usd=.74,estimated_gpu_usd=result['control']['seconds']/3600*.74,scope='Controller wall includes all training and final scientific eval; excludes setup idle storage, not invoice. Paired artifact costs separately reported.',optimizer_updates=512,task_predictions=1064,nll_forwards=504,gradient_passes=8);save(R/f'logs/{STAGE}-v0-cost-estimate.json',cost)
        (R/'STATE.md').write_text('# 当前状态：完整匹配训练与QK恢复复验完成\n\n'+utc()+'。原目标：稀疏继续训练省成本、同密集评测质量接近。\n\n报告docs/matched-restore-training-results-2026-09-17.md；预定复验筛查='+json.dumps(result['replication_screen'])+'。同卡实际训练成本，已知恢复操作，评测数据已暴露；不是新方法、独立确认或自动成立论文。\n\n累计6784科学更新、299诊断更新、35583已记录任务预测；本轮另504 NLL前向、8梯度预检。历史1次未完整记录任务失败与PG19类型错误1次书籍主体失败另记，不混入已完成质量结果。\n\n下一步：'+c['next_action']+'\n\nPod运行，heartbeat ACTIVE。\n',encoding='utf-8')
        if not already:
            with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：完整匹配QKVO训练与恢复复验核验完成\n\n新增512科学更新、1064任务预测、504 NLL前向、8梯度预检。保存0/64/128及恢复适配器、逐步计时、逐题/书结果、UTC与费用；已知方法和暴露数据的成本质量复验。\n')
        for n in ['STATE.md','TIMELINE.md','logs/control-state.json','docs/matched-restore-training-results-2026-09-17.md',f'logs/{STAGE}-v0-cost-estimate.json']:subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
        save(state,dict(status='complete_verified',utc=utc(),report='docs/matched-restore-training-results-2026-09-17.md'))
    except Exception:
        save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),scope='Do not duplicate launch or change frozen protocol.'));raise
if __name__=='__main__':main()

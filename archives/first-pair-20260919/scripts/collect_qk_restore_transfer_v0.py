"""Collect and verify a bounded, non-novel development transfer intervention."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1]
STAGE='qk-restore-transfer'
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    c=json.loads((R/'logs/cloud-connection-current.json').read_text(encoding='utf-8'));host=c['ssh_user']+'@'+c['ssh_host'];root=c['root']
    opts=['-i',c['ssh_key_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15'];ssh=['ssh',*opts,'-p',str(c['ssh_port']),host];scp=['scp',*opts,'-P',str(c['ssh_port'])]
    state=R/f'logs/{STAGE}-v0-local-collector.json';start=time.monotonic();errors=0
    try:
        while time.monotonic()-start<1800:
            code="from pathlib import Path\nimport json\nr=Path("+repr(root)+")\nf=r/'logs/qk-restore-transfer-queue-v0.json'\nprint(json.dumps(dict(queue=json.loads(f.read_text()) if f.exists() else None,archive_ready=(r/'exports/qk-restore-transfer-evidence-v0.tar.json').exists())))\n"
            try:
                q=subprocess.run(ssh+['python3','-'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=40)
                assert q.returncode==0,q.stderr[-1000:];v=json.loads(q.stdout);errors=0
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
        with (R/f'logs/{STAGE}-v0-audit.log').open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,str(R/'scripts/report_qk_restore_transfer_v0.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((R/f'results/{STAGE}-audit-v0/result.json').read_text(encoding='utf-8'));assert result['status']=='verified'
        c=json.loads((R/'logs/control-state.json').read_text(encoding='utf-8'));already=c.get('qk_restore_transfer',{}).get('status')=='complete_verified'
        if not already:
            assert c['cumulative_task_predictions']==33987
            c['cumulative_task_predictions']+=result['task_predictions']
        c.update(status='qk_restore_transfer_v0_complete_verified',updated_utc=utc(),active_evaluation_jobs=0,next_action='Inspect known QK-Restore transfer contrasts on the now-exposed second word/background set. If screen passes, freeze truly external natural-quality and matched actual training-cost validation before further training; no new-method claim for known restoration. If screen fails, retain negative evidence and choose a distinct falsifiable hypothesis, not another blind run. Original goal remains cheaper sparse continued training with comparable common-evaluation quality.')
        c['qk_restore_transfer']=dict(status='complete_verified',report='docs/qk-restore-transfer-results-2026-09-17.md',task_predictions=result['task_predictions'],contrasts=result['contrasts'],development_screen_pass=result['development_screen_pass'],archive=result['archive']);save(R/'logs/control-state.json',c)
        cost=dict(utc=utc(),seconds=result['control']['seconds'],hourly_gpu_usd=.74,estimated_gpu_usd=result['control']['seconds']/3600*.74,scope='Controller wall only; excludes setup, idle, storage. Not invoice. Cumulative historical invoice still unverified.',optimizer_updates=0,task_predictions=result['task_predictions'])
        save(R/f'logs/{STAGE}-v0-cost-estimate.json',cost)
        (R/'STATE.md').write_text('# 当前状态：QK-Restore 第二批背景开发验证完成\n\n'+utc()+'。主线：训练省成本、质量接近密集，不要求推理加速。\n\n报告docs/qk-restore-transfer-results-2026-09-17.md；开发筛查='+str(result['development_screen_pass'])+'；主要对照='+json.dumps(result['contrasts']['restored_sparse_vs_original_dense'])+'。已有恢复基线，非原创；第二批题已暴露，非独立确认。\n\n累计6272科学更新、299诊断更新、'+str(c['cumulative_task_predictions'])+'已记录任务预测，另1次历史未完整记录的失败任务前向。本轮0更新、532任务预测、16次校准NLL。\n\n下一步：'+c['next_action']+'\n\nPod运行，heartbeat ACTIVE。\n',encoding='utf-8')
        if not already:
            with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：QK-Restore 第二批背景开发验证核验完成\n\n新增532任务预测、16次校准NLL、0更新。固定128步四检查点；已知基线，不是新方法；此前暴露的背景，不是独立确认。原始证据哈希、逐题结果和费用已保存。\n')
        for n in ['STATE.md','TIMELINE.md','logs/control-state.json','docs/qk-restore-transfer-results-2026-09-17.md',f'logs/{STAGE}-v0-cost-estimate.json']:
            subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
        save(state,dict(status='complete_verified',utc=utc(),report='docs/qk-restore-transfer-results-2026-09-17.md'))
    except Exception:
        save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),scope='Do not duplicate launch or change frozen protocol.'));raise
if __name__=='__main__':main()

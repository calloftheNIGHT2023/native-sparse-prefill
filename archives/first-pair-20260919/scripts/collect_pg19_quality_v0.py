"""One bounded collector; preserve complete raw evidence before reporting external quality."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1]
STAGE='pg19-quality'
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    c=json.loads((R/'logs/cloud-connection-current.json').read_text(encoding='utf-8'));host=c['ssh_user']+'@'+c['ssh_host'];root=c['root']
    opts=['-i',c['ssh_key_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15'];ssh=['ssh',*opts,'-p',str(c['ssh_port']),host];scp=['scp',*opts,'-P',str(c['ssh_port'])]
    state=R/f'logs/{STAGE}-v0-local-collector.json';start=time.monotonic();errors=0
    try:
        while time.monotonic()-start<2700:
            code="from pathlib import Path\nimport json\nr=Path("+repr(root)+")\nf=r/'logs/pg19-quality-queue-v0.json'\nprint(json.dumps(dict(queue=json.loads(f.read_text()) if f.exists() else None,archive_ready=(r/'exports/pg19-quality-evidence-v0.tar.json').exists())))\n"
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
            subprocess.run([sys.executable,str(R/'scripts/report_pg19_quality_v0.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((R/f'results/{STAGE}-audit-v0/result.json').read_text(encoding='utf-8'));assert result['status']=='verified'
        c=json.loads((R/'logs/control-state.json').read_text(encoding='utf-8'));already=c.get('pg19_quality',{}).get('status')=='complete_verified'
        assert c['cumulative_task_predictions']==34519
        c.update(status='pg19_quality_v0_complete_verified',updated_utc=utc(),active_evaluation_jobs=0,next_action='Review external16-book full/tail quality and all symmetric restoration contrasts. If pass, preserve external evidence and design matched actual complete128-step sparse/dense training-cost confirmation with explicit budget, plus natural task evaluation; never reuse old-card speed as current GPU saving. If failed, preserve counterevidence and choose a distinct falsifiable hypothesis before more training. Existing QK-Restore remains prior art; not an original method or paper-ready evidence.')
        c['pg19_quality']=dict(status='complete_verified',report='docs/pg19-quality-results-2026-09-17.md',nll_forwards=result['nll_forwards'],task_predictions=0,external_sample_joint_pass=result['external_sample_joint_pass'],contrasts=result['contrasts'],archive=result['archive']);save(R/'logs/control-state.json',c)
        cost=dict(utc=utc(),seconds=result['control']['seconds'],hourly_gpu_usd=.74,estimated_gpu_usd=result['control']['seconds']/3600*.74,scope='Controller wall only; excludes setup, idle, storage. Not invoice. Cumulative historical invoice still unverified.',optimizer_updates=0,task_predictions=0,nll_forwards=result['nll_forwards']);save(R/f'logs/{STAGE}-v0-cost-estimate.json',cost)
        (R/'STATE.md').write_text('# 当前状态：PG19 外部真实文本评测完成\n\n'+utc()+'。主线：训练省成本、质量接近密集，不要求推理加速。\n\n报告docs/pg19-quality-results-2026-09-17.md；16书外部样本整体/结尾困惑度联合门槛='+str(result['external_sample_joint_pass'])+'。已知QK-Restore，非原创；16本书子集的token困惑度，不是官方完整PG19词困惑度。\n\n累计6272科学更新、299诊断更新、34519已记录任务预测，另1次历史失败未完整记录。本轮468 NLL前向、0更新、0任务预测。\n\n下一步：'+c['next_action']+'\n\nPod运行，heartbeat ACTIVE。\n',encoding='utf-8')
        if not already:
            with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：PG19 外部16书质量检查完成核验\n\n9条件、468 NLL前向、0更新、0任务预测。固定书籍和128步终点，整体/结尾5%困惑度界限预先冻结；原始数据、选择记录、逐书结果、模型张量摘要、校准、UTC与费用已保留。\n')
        for n in ['STATE.md','TIMELINE.md','logs/control-state.json','docs/pg19-quality-results-2026-09-17.md',f'logs/{STAGE}-v0-cost-estimate.json']:
            subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
        save(state,dict(status='complete_verified',utc=utc(),report='docs/pg19-quality-results-2026-09-17.md'))
    except Exception:
        save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),scope='Do not duplicate launch or change frozen protocol.'));raise
if __name__=='__main__':main()

"""Collect fixed early-stop controls without launching any training."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];STAGE='sparse-early64'
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    c=json.loads((R/'logs/cloud-connection-current.json').read_text(encoding='utf-8'));host=c['ssh_user']+'@'+c['ssh_host'];root=c['root'];opts=['-i',c['ssh_key_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15'];ssh=['ssh',*opts,'-p',str(c['ssh_port']),host];scp=['scp',*opts,'-P',str(c['ssh_port'])]
    state=R/f'logs/{STAGE}-v0-local-collector.json';start=time.monotonic();errors=0
    try:
        while time.monotonic()-start<2100:
            code="from pathlib import Path\nimport json\nr=Path("+repr(root)+")\nf=r/'logs/sparse-early64-queue-v0.json'\nprint(json.dumps(dict(queue=json.loads(f.read_text()) if f.exists() else None,archive_ready=(r/'exports/sparse-early64-evidence-v0.tar.json').exists())))\n"
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
        with (R/f'logs/{STAGE}-v0-audit.log').open('w',encoding='utf-8') as f:subprocess.run([sys.executable,str(R/'scripts/report_sparse_early64_v0.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((R/f'results/{STAGE}-audit-v0/result.json').read_text(encoding='utf-8'));assert result['status']=='verified'
        c=json.loads((R/'logs/control-state.json').read_text(encoding='utf-8'));already=c.get('sparse_early64',{}).get('status')=='complete_verified'
        if not already:
            assert c['cumulative_task_predictions']==44307
            c['cumulative_task_predictions']+=2580
        c.update(status='sparse_early64_v0_complete_verified',updated_utc=utc(),active_evaluation_jobs=0,next_action='Consolidate the 64/128 matched training cost-quality curve; inspect all five endpoints and known QKRestore tradeoffs. Do not launch more training just from a pass on exposed data. Next freeze a fresh long-context confirmation with sufficient independent backgrounds and baseline checks, plus an originality audit before model/seed expansion. Main goal remains training cost with comparable quality, no inference speed requirement.')
        c['sparse_early64']=dict(status='complete_verified',report='docs/sparse-early64-results-2026-09-17.md',task_predictions=2580,nll_forwards=244,candidate_screens=result['candidate_screens'],cheaper_sparse_candidate_passes=result['cheaper_sparse_candidate_passes'],cost=result['cost'],archive=result['archive']);save(R/'logs/control-state.json',c)
        cost=dict(utc=utc(),seconds=result['control']['seconds'],hourly_gpu_usd=.74,estimated_gpu_usd=result['control']['seconds']/3600*.74,scope='Controller wall only; excludes setup idle storage; not invoice.',optimizer_updates=0,task_predictions=2580,nll_forwards=244);save(R/f'logs/{STAGE}-v0-cost-estimate.json',cost)
        (R/'STATE.md').write_text('# 当前状态：稀疏64步同预算对照已核验\n\n'+utc()+'。主目标仍是稀疏继续训练成本与接近密集质量。\n\n报告docs/sparse-early64-results-2026-09-17.md；更便宜稀疏候选通过联合筛查='+str(result['cheaper_sparse_candidate_passes'])+'；具体='+json.dumps(result['candidate_screens'])+'。暴露数据的成本曲线开发对照，非新方法或独立确认。\n\n此前同128步训练恢复省时7.73%/7.62%，不能据此自动主张同质量最便宜。当前累计6784科学更新、299诊断更新、46887任务预测；本轮0更新、2580预测、244NLL；历史失败单独保留。\n\n下一步：'+c['next_action']+'\n\nPod运行，heartbeat ACTIVE。\n',encoding='utf-8')
        if not already:
            with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：稀疏64步同预算成本质量对照完成核验\n\n新增2580任务预测、244NLL，0更新。预定原/恢复两候选，联合五项质量和成本，97.5%配对区间；保留不同计时来源与1秒敏感性。所有原始结果、UTC、费用、哈希与参照来源已留存。\n')
        for n in ['STATE.md','TIMELINE.md','logs/control-state.json','docs/sparse-early64-results-2026-09-17.md',f'logs/{STAGE}-v0-cost-estimate.json']:subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
        save(state,dict(status='complete_verified',utc=utc(),report='docs/sparse-early64-results-2026-09-17.md'))
    except Exception:
        save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),scope='Do not duplicate launch or change frozen protocol.'));raise
if __name__=='__main__':main()

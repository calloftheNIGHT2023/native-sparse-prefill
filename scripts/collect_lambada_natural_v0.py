"""Collect natural task evidence once and maintain exact recorded prediction counts."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];STAGE='lambada-natural'
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    c=json.loads((R/'logs/cloud-connection-current.json').read_text(encoding='utf-8'));host=c['ssh_user']+'@'+c['ssh_host'];root=c['root'];opts=['-i',c['ssh_key_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15'];ssh=['ssh',*opts,'-p',str(c['ssh_port']),host];scp=['scp',*opts,'-P',str(c['ssh_port'])]
    state=R/f'logs/{STAGE}-v0-local-collector.json';start=time.monotonic();errors=0
    try:
        while time.monotonic()-start<2400:
            code="from pathlib import Path\nimport json\nr=Path("+repr(root)+")\nf=r/'logs/lambada-natural-queue-v0.json'\nprint(json.dumps(dict(queue=json.loads(f.read_text()) if f.exists() else None,archive_ready=(r/'exports/lambada-natural-evidence-v0.tar.json').exists())))\n"
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
        with (R/f'logs/{STAGE}-v0-audit.log').open('w',encoding='utf-8') as f:subprocess.run([sys.executable,str(R/'scripts/report_lambada_natural_v0.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((R/f'results/{STAGE}-audit-v0/result.json').read_text(encoding='utf-8'));assert result['status']=='verified'
        c=json.loads((R/'logs/control-state.json').read_text(encoding='utf-8'));already=c.get('lambada_natural',{}).get('status')=='complete_verified'
        if not already:
            assert c['cumulative_task_predictions']==35583
            c['cumulative_task_predictions']+=result['task_predictions']
        c.update(status='lambada_natural_v0_complete_verified',updated_utc=utc(),active_evaluation_jobs=0,next_action='Inspect natural full-vocabulary word prediction and base context gate alongside matched7.6-7.7%training artifact cost. If base gate failed, preserve and do not call it sparse failure; choose a documented suitable natural task without outcome-guided revisions. If retention passed, consolidate evidence and screen contribution overlap before independent-seed/generalization expansion; known QK-Restore and LongLoRA are not original contributions. Current0.5B LoRA evidence not automatically sufficient for paper. Keep no inference-speed requirement.')
        c['lambada_natural']=dict(status='complete_verified',report='docs/lambada-natural-results-2026-09-17.md',task_predictions=result['task_predictions'],calibration_nll_forwards=result['calibration_nll_forwards'],base_gate=result['base_gate'],natural_accuracy_screen_pass=result['natural_accuracy_screen_pass'],contrasts=result['contrasts'],archive=result['archive']);save(R/'logs/control-state.json',c)
        cost=dict(utc=utc(),seconds=result['control']['seconds'],hourly_gpu_usd=.74,estimated_gpu_usd=result['control']['seconds']/3600*.74,scope='Controller wall only; excludes setup idle storage; not invoice.',optimizer_updates=0,task_predictions=result['task_predictions'],calibration_nll_forwards=result['calibration_nll_forwards']);save(R/f'logs/{STAGE}-v0-cost-estimate.json',cost)
        (R/'STATE.md').write_text('# 当前状态：LAMBADA自然末词预测已核验\n\n'+utc()+'。原目标仍是稀疏继续训练省成本、同设置质量接近。\n\n报告docs/lambada-natural-results-2026-09-17.md；基座门槛='+json.dumps(result['base_gate'])+'；自然准确率非劣筛查='+str(result['natural_accuracy_screen_pass'])+'。最多147tokens，非32K自然理解测试；既有QK-Restore非原创。\n\n此前完整匹配训练两种子含恢复写盘省时7.73%/7.62%，联合质量/成本筛查通过。当前累计6784科学更新、299诊断更新、'+str(c['cumulative_task_predictions'])+'任务预测；本轮0更新，任务预测'+str(result['task_predictions'])+'。历史失败另记。\n\n下一步：'+c['next_action']+'\n\nPod运行，heartbeat ACTIVE。\n',encoding='utf-8')
        if not already:
            with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：LAMBADA自然末词任务核验完成\n\n新增'+str(result['task_predictions'])+'任务预测，'+str(result['calibration_nll_forwards'])+'校准NLL，0更新。固定512题全词表目标词，基座全上下文/末32token门槛和主要5pp界限事先冻结；所有逐题数据与源哈希、UTC、费用已保存。\n')
        for n in ['STATE.md','TIMELINE.md','logs/control-state.json','docs/lambada-natural-results-2026-09-17.md',f'logs/{STAGE}-v0-cost-estimate.json']:subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
        save(state,dict(status='complete_verified',utc=utc(),report='docs/lambada-natural-results-2026-09-17.md'))
    except Exception:
        save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),scope='Do not duplicate launch or change frozen protocol.'));raise
if __name__=='__main__':main()

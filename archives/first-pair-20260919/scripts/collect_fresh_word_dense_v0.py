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
    state=R/'logs/fresh-word-dense-v0-local-collector.json';start=time.monotonic();errors=0
    try:
        while time.monotonic()-start<2100:
            code="from pathlib import Path\nimport json\nr=Path("+repr(root)+")\nf=r/'logs/fresh-word-dense-queue-v0.json'\nprint(json.dumps(dict(queue=json.loads(f.read_text()) if f.exists() else None,archive_ready=(r/'exports/fresh-word-dense-evidence-v0.tar.json').exists())))\n"
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
        for n in ['fresh-word-dense-evidence-v0.tar.json','fresh-word-dense-evidence-v0.tar.gz']:
            subprocess.run(scp+[host+':'+root+'/exports/'+n,str(R/'exports'/n)],check=True,timeout=300)
        with (R/'logs/fresh-word-dense-v0-audit.log').open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,str(R/'scripts/report_fresh_word_dense_v0.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((R/'results/fresh-word-dense-audit-v0/result.json').read_text(encoding='utf-8'));assert result['status']=='verified'
        c=json.loads((R/'logs/control-state.json').read_text(encoding='utf-8'))
        already=c.get('fresh_word_dense',{}).get('status')=='complete_verified'
        if not already:
            assert c['cumulative_task_predictions']==31786
            c['cumulative_task_predictions']+=result['task_predictions']
        c.update(status='fresh_word_dense_v0_complete_verified',updated_utc=utc(),active_evaluation_jobs=0,next_action='Inspect prospective fresh-background common-dense factual result alongside4.03% development PPL penalty and5.17-7.53% full-step diagnostic savings. If base ability gate failed, preserve it and design an evidence-grounded task before training. If noninferiority fails, choose a new bounded training hypothesis targeting the demonstrated gap, with prior-art screening before method claims. If it passes, proceed to matched all-sparse-vs-dense actual training and genuinely independent natural-quality validation. Never claim end-to-end equal-quality cost from separate diagnostics or require inference speed.')
        c['fresh_word_dense'].update(status='complete_verified',report='docs/fresh-word-dense-results-2026-09-16.md',base_ability_gate_passed=result['base_ability_gate_passed'],task_predictions=result['task_predictions'],comparison=result['comparison'],archive=result['archive']);save(R/'logs/control-state.json',c)
        cost=dict(utc=utc(),seconds=result['control']['seconds'],hourly_gpu_usd=.74,estimated_gpu_usd=result['control']['seconds']/3600*.74,scope='Controller wall only; excludes setup, idle, storage. Not an invoice.',optimizer_updates=0,task_predictions=result['task_predictions'])
        save(R/'logs/fresh-word-dense-v0-cost-estimate.json',cost)
        (R/'STATE.md').write_text('# 当前状态：新背景、新答案值的同设置质量检查已核验\n\n'+utc()+'。主线仍是训练省成本、效果接近密集，不要求推理加速。\n\n报告docs/fresh-word-dense-results-2026-09-16.md；能力门槛通过='+str(result['base_ability_gate_passed'])+'；主对照='+json.dumps(result['comparison'],ensure_ascii=False)+'。\n\n累计5760科学更新、299诊断更新、'+str(c['cumulative_task_predictions'])+'已记录任务预测；另18次无更新梯度计算、279次统一NLL前向和1次失败未完整记录。\n\n下一步：'+c['next_action']+'\n\nPod运行，heartbeat ACTIVE。\n',encoding='utf-8')
        if not already:
            with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：新背景同密集注意力事实检查完成核验\n\n新增'+str(result['task_predictions'])+'任务预测、0更新。32个互不共享文章的背景、4个新事实词，比较门槛事先固定。逐题预测、UTC和费用估算均保存，报告docs/fresh-word-dense-results-2026-09-16.md。\n')
        for n in ['STATE.md','TIMELINE.md','logs/control-state.json','docs/fresh-word-dense-results-2026-09-16.md','logs/fresh-word-dense-v0-cost-estimate.json']:
            subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
        save(state,dict(status='complete_verified',utc=utc(),report='docs/fresh-word-dense-results-2026-09-16.md'))

    except Exception:
        save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),scope='No duplicate launch. Preserve archive and inspect cloud state.'));raise
if __name__=='__main__':main()

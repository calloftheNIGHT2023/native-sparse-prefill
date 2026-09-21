"""Single collector, immutable archive, local audit and idempotent ledger update."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];STAGE='book256-factorial'
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 c=json.loads((R/'logs/cloud-connection-current.json').read_text());host=c['ssh_user']+'@'+c['ssh_host'];root=c['root'];opts=['-i',c['ssh_key_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15'];ssh=['ssh',*opts,'-p',str(c['ssh_port']),host];scp=['scp',*opts,'-P',str(c['ssh_port'])]
 state=R/f'logs/{STAGE}-v0-local-collector.json';start=time.monotonic();errors=0
 try:
  while time.monotonic()-start<8400:
   code="from pathlib import Path\nimport json\nr=Path("+repr(root)+")\nf=r/'logs/book256-factorial-queue-v0.json'\nprint(json.dumps(dict(queue=json.loads(f.read_text()) if f.exists() else None,archive_ready=(r/'exports/book256-factorial-evidence-v0.tar.json').exists())))\n"
   try:
    q=subprocess.run(ssh+['python3','-'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=40);assert q.returncode==0,q.stderr[-1000:];v=json.loads(q.stdout);errors=0
   except Exception:
    errors+=1
    if errors>=5:raise
    save(state,dict(status='connection_retry',utc=utc(),errors=errors));time.sleep(30);continue
   save(state,dict(status='waiting',utc=utc(),cloud=v))
   if v['archive_ready']:break
   time.sleep(30)
  else:raise RuntimeError('Collector timeout; inspect before retry')
  for ext in ['json','gz']:
   n=f'{STAGE}-evidence-v0.tar.{ext}';subprocess.run(scp+[host+':'+root+'/exports/'+n,str(R/'exports'/n)],check=True,timeout=300)
  with (R/f'logs/{STAGE}-v0-audit.log').open('w',encoding='utf-8') as f:subprocess.run([sys.executable,str(R/'scripts/report_book256_factorial_v0.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
  result=json.loads((R/f'results/{STAGE}-audit-v0/result.json').read_text());assert result['status']=='verified'
  c=json.loads((R/'logs/control-state.json').read_text());already=c.get('book256_factorial',{}).get('status')=='complete_verified'
  if not already:
   assert c['cumulative_task_predictions']==56150;c['cumulative_task_predictions']+=8232
  c.update(status='book256_factorial_v0_complete_verified',updated_utc=utc(),active_evaluation_jobs=0,next_action='Inspect symmetric-restored comparison and restore interaction; treat as adaptive exposed-data diagnostics, not new independent confirmation or novelQKRestore. Preserve original3contrasts and dense64 early-stop result. Combine existing natural-quality and same-GPU training-cost evidence before deciding a bounded mechanism or independent replication. No blind extension. Avoid prolonged idleGPU; preserve verified backups before stopping.')
  c['book256_factorial']=dict(status='complete_verified',report='docs/book256-factorial-results-2026-09-17.md',task_predictions=8232,nll_forwards=32,contrasts=result['contrasts'],archive=result['archive']);save(R/'logs/control-state.json',c)
  save(R/f'logs/{STAGE}-v0-cost-estimate.json',dict(utc=utc(),seconds=result['control']['seconds'],hourly_gpu_usd_estimate=.74,hourly_rate_verified=False,estimated_gpu_usd=result['control']['seconds']/3600*.74,scope='Stage wall only, excludes setup idle storage; not invoice.',optimizer_updates=0,task_predictions=8232,nll_forwards=32))
  (R/'STATE.md').write_text('# 当前状态：书籍背景完整恢复对照已核验\n\n'+utc()+'\n\n报告docs/book256-factorial-results-2026-09-17.md；8组新增8232预测，32校准NLL，0训练更新。累计6784科学更新、299诊断更新、'+str(c['cumulative_task_predictions'])+'记录预测。\n\n'+json.dumps(result['contrasts'],ensure_ascii=False)+'\n\n本轮是已暴露书籍上的追加诊断；固定98.75%书级区间、5pp容差。已知QK恢复不作为原创，旧训练计时不等于本轮评测耗时。\n\n下一步：'+c['next_action']+'\n\nPod运行，heartbeat ACTIVE。\n',encoding='utf-8')
  if not already:
   with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：完整恢复对照核验完成\n\n保留17组原始预测，8组新增8232条；0更新，32校准NLL。哈希、检查点、恢复参数、全部标签/logits及4项书级比较核验，累计64382记录预测。诊断为自适应追加，不作新任务或新种子确认。\n')
  for n in ['STATE.md','TIMELINE.md','logs/control-state.json','docs/book256-factorial-results-2026-09-17.md',f'logs/{STAGE}-v0-cost-estimate.json']:subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
  save(state,dict(status='complete_verified',utc=utc(),report='docs/book256-factorial-results-2026-09-17.md'))
 except Exception:
  save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),scope='Preserve archive; do not duplicate launch or loosen protocol.'));raise
if __name__=='__main__':main()

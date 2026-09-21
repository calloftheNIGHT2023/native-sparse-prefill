"""Audit diagnostic timing, including math gates and full optimizer updates."""
import json,hashlib,math,tarfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 a=R/'exports/density-tradeoff-cost-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
 dest=R/'results/cloud-density-tradeoff-cost-evidence-v0';dest.mkdir(exist_ok=True)
 with tarfile.open(a) as t:
  entries=json.loads(t.extractfile('density-tradeoff-cost-manifest.json').read())['files'];names=[x.name for x in t.getmembers()];assert len(names)==len(set(names)) and set(names)=={x['path'] for x in entries}|{'density-tradeoff-cost-manifest.json'} and len(entries)==proof['files']
  for e in entries:
   m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   raw=t.extractfile(m).read();assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
 p=load(dest/'provenance/density-tradeoff-cost-protocol.json');psha=sha(dest/'provenance/density-tradeoff-cost-protocol.json');assert psha==sha(R/'provenance/density-tradeoff-cost-protocol.json')
 for n,h in p['source_sha256'].items():assert sha(dest/n)==h
 parent=R/'results/cloud-32k-adaptation-evidence-v0'/p['parent_checkpoint'];assert sha(parent)==p['parent_checkpoint_sha256']
 import torch
 torch.set_num_threads(4);parent_state=torch.load(parent,map_location='cpu',weights_only=False)
 digest=hashlib.sha256(b''.join(x.numpy().tobytes() for x in parent_state['params'].values())).hexdigest()
 stage=dest/'results/density-tradeoff-cost-stage-v0';control=load(stage/'result.json');rows=[];count=0;order=np.random.default_rng(2026091662).permutation(32).tolist();paired={}
 for j in control['jobs']:
  if j['status']!='complete':continue
  d=stage/j['name'];v=load(d/'result.json');m=load(d/'measurement.json');k=j['k'];ri=j['round'];trace=v['trace']
  assert v['status']=='complete' and v['optimizer_updates_this_process']==6 and m['optimizer_updates']==6 and len(trace)==6
  assert m['protocol_sha256']==psha and m['parent_params_sha256']==digest and m['parent_replay_error']<=1e-6 and m['gate']['passed']
  assert [x['step'] for x in trace]==list(range(65,71)) and all(x['window_index']==order[(x['step']-1)%32] and x['lr']==.0001 and math.isfinite(x['train_nll']) and x['seconds']>0 for x in trace)
  assert m['measured_seconds']==[x['seconds'] for x in trace[2:]] and abs(m['median_seconds']-np.median(m['measured_seconds']))<1e-10
  assert sha(d/'diagnostic-final.pt')==m['diagnostic_checkpoint_sha256'] and sha(d/'source.py')==p['source_sha256']['scripts/benchmark_density_tradeoff_cost.py']
  c=torch.load(d/'diagnostic-final.pt',map_location='cpu',weights_only=False);assert c['step']==c['data_cursor']==c['scheduler']['last_epoch']==70 and c['extra']['rows']==trace
  assert c['identity']['diagnostic'] and c['identity']['mode']==k and c['identity']['round']==ri and c['identity']['parent_identity']==parent_state['identity']
  assert all(float(x['step'])==70 for x in c['optimizer']['state'].values()) and c['optimizer']['param_groups'][0]['lr']==.0001
  assert all(torch.isfinite(x).all() for x in c['params'].values())
  actual=hashlib.sha256(b''.join(x.numpy().tobytes() for x in c['params'].values())).hexdigest();assert actual==m['final_params_sha256'] and actual!=digest
  if k in paired:
   assert paired[k]['digest']==actual and np.max(np.abs(np.array(paired[k]['losses'])-[x['train_nll'] for x in trace]))<=1e-6
  else:paired[k]=dict(digest=actual,losses=[x['train_nll'] for x in trace])
  if k:
   g=m['gate'];assert g['length']==8704 and g['query_heads']==14 and g['kv_heads']==2 and g['selection_regret']<.001 and g['output_relative_l2']<.02 and max(g['gradient_relative_l2'])<.03
  rows.append(m);count+=1
 comparisons=[]
 for k in [32,48,64]:
  ratios=[]
  for ri in range(3):
   dense=next((x for x in rows if x['mode']==0 and x['round']==ri),None);sparse=next((x for x in rows if x['mode']==k and x['round']==ri),None)
   if dense and sparse:ratios.append(sparse['median_seconds']/dense['median_seconds'])
  expected=next(x for x in control['candidates'] if x['k']==k);eligible=len(ratios)==3 and all(x<1 for x in ratios)
  # The frozen controller gathered measurements in filesystem glob order.
  # Independently recompute pairs by explicit round; compare the multiset,
  # then publish the canonical round order without rewriting raw evidence.
  assert sorted(ratios)==sorted(expected['ratios']) and eligible==expected['eligible_for_calibration_training']
  comparisons.append(dict(k=k,rounds=len(ratios),ratios=ratios,saving_percent_by_round=[100*(1-x) for x in ratios],eligible_for_calibration_training=eligible))
 result=dict(status='verified',experiment_status=control['status'],utc=datetime.now(timezone.utc).isoformat(),archive=proof,verified_diagnostic_checkpoints=count,diagnostic_optimizer_updates=control['diagnostic_optimizer_updates'],update_accounting_complete=control['update_accounting_complete'],scientific_updates=0,task_predictions=0,seconds=control['seconds'],rows=rows,comparisons=comparisons,jobs=control['jobs'],scope=p['scope'])
 save(R/'results/density-tradeoff-cost-audit-v0/result.json',result);save(dest/'LOCAL-VERIFICATION.json',dict(status='verified',archive=proof,checkpoints=count))
 lines=['# K32/K48/K64完整更新速度筛选','',f"实验状态：{control['status']}。只做诊断，不把六步诊断更新当成质量训练结果。每个测量单元从同一密集64步断点恢复全部状态；两步预热、四步测量，包含前反向、裁剪、AdamW与scheduler。",'', '|轮次|K（0为密集）|测量中位秒/步|峰值显存GiB|数学检查|','|---|---:|---:|---:|---|']
 for m in rows:lines.append(f"|{m['round']}|{m['mode']}|{m['median_seconds']:.4f}|{m['peak_gpu_bytes']/2**30:.2f}|通过|")
 lines+=['','|档位|三轮相对密集的节时|进入校准训练候选|','|---|---|---|']
 for c in comparisons:lines.append(f"|K{c['k']}|"+' / '.join(f'{x:+.2f}%' for x in c['saving_percent_by_round'])+f"|{'是' if c['eligible_for_calibration_training'] else '否或未完成'}|")
 lines+=['','候选条件为三个配对轮次都比密集快，并通过数学/断点复现检查。三轮是计时重复，不是三个训练种子或正式统计优势证明。K48/K64即使通过速度检查，仍须另外做公平的质量训练；本阶段不读取测试集。','',f"诊断优化更新记录数{control['diagnostic_optimizer_updates']}，记账完整性{control['update_accounting_complete']}；科学训练更新0、任务预测0。若某作业超时且无最终状态，更新数可能不完整，保留原始日志后再核算。全阶段{control['seconds']:.2f}秒，含模型加载/校准/数学验证等；不能用测量中位数代替租卡总账单。",'',f"核验{proof['files']}文件、{count}诊断断点。同档位不同轮次最终参数摘要和逐步损失必须复现；数学检查覆盖真实稀疏的8704长度和GQA14:2，比较FP32选块、因果注意力前向及Q/K/V梯度。包SHA `{proof['sha256']}`。",'', '失败及跳过作业同样保留：','']
 for j in control['jobs']:
  if j['status']!='complete':lines.append('- '+j['name']+'：'+j['status'])
 lines+=['','## 时间轴','','全部UTC。','','|作业|开始|结束|状态|','|---|---|---|---|']
 for j in control['jobs']:lines.append(f"|{j['name']}|{j.get('started_utc',j.get('utc',''))}|{j.get('finished_utc','')}|{j['status']}|")
 (R/'docs/density-tradeoff-cost-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(dict(status='verified',comparisons=comparisons,checkpoints=count,diagnostic_updates=control['diagnostic_optimizer_updates'])))
if __name__=='__main__':main()

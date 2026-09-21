"""Verify the isolated evidence package and audit every scored prediction."""
import hashlib, json, tarfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 archive=R/'exports/task-route-coverage-evidence-v0.tar.gz'
 proof=read(archive.with_suffix('.json'));assert sha(archive)==proof['sha256']
 dest=R/'results/cloud-task-route-coverage-evidence-v0';dest.mkdir(exist_ok=False)
 with tarfile.open(archive) as tar:
  members=tar.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))
  manifest=json.load(tar.extractfile('amp-recovery-results-manifest.json'))
  expected={x['path']:x for x in manifest['files']}
  assert set(names)==set(expected)|{'amp-recovery-results-manifest.json'}
  for item in members:
   assert item.isfile() and not Path(item.name).is_absolute() and '..' not in Path(item.name).parts
   target=(dest/item.name).resolve();assert dest.resolve() in target.parents
   raw=tar.extractfile(item).read()
   if item.name in expected:
    e=expected[item.name];assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256']
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 verification=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),sha256=proof['sha256'],files=len(expected))
 (dest/'LOCAL-VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
 stage=dest/'results/task-route-coverage-stage-v0';controller=read(stage/'result.json')
 protocol_path=dest/'provenance/task-quality-route-coverage-protocol.json';protocol=read(protocol_path)
 assert controller['status']=='complete' and controller['optimizer_updates']==0
 assert controller['protocol_sha256']==sha(protocol_path) and len(controller['runs'])==2
 assert all(x['returncode']==0 for x in controller['runs'])
 for n,h in protocol['sources'].items():assert sha(dest/n)==h
 selected_stage=R/'results/cloud-task-route-selective-evidence-v0/results/task-route-selective-stage-v0'
 placements={x['item_id']:x for x in read(dest/'provenance/task-quality-route-chain-protocol.json')['rows']}
 all_rows=[];count=0
 for s in [0,1]:
  seed=2026091560+s;d=read(stage/f'seed{seed}'/'result.json');old=read(selected_stage/f'seed{seed}'/'result.json')
  assert d['status']=='complete' and d['optimizer_updates']==0 and d['seed']==seed
  for n in ['checkpoint_sha256','training_result_sha256','task_protocol_sha256']:assert d[n]==old[n]
  assert d['route_protocol_sha256']==sha(protocol_path) and d['control_max_logit_error']<=1e-6 and d['calibration_replay_max_abs_error']<=1e-6
  prior={x['item_id']:x for x in old['predictions'] if x['condition']=='baseline'}
  assert len(d['predictions'])==24 and set(x['item_id'] for x in d['predictions'])==set(protocol['screen_ids'])
  for x in d['predictions']:
   item=x['item_id'];p=prior[item];count+=1
   assert x['condition']=='baseline' and x['prediction']==p['prediction'] and x['gold']==p['gold'] and x['correct']==p['correct']
   assert np.max(np.abs(np.array(x['choice_logits'])-p['choice_logits']))<=1e-6
   target=placements[item]['target_blocks']
   for layer in range(24):
    h=d['frozen_heads'][item][str(layer)];before=old['frozen_heads'][item][str(layer)]
    assert h['selected']==before['selected'] and h['random']==before['random']
    assert np.max(np.abs(np.array(h['mass'])-before['mass']))<=1e-6
    ids=h['final_query_native_blocks'];assert len(ids)==14
    recalls=[]
    for blocks in ids:
     assert len(blocks)==len(set(blocks))==16 and all(0<=b<=127 for b in blocks) and 127 in blocks
     recalls.append(len(set(blocks)&set(target))/len(target))
    assert np.max(np.abs(np.array(recalls)-h['final_query_target_recall']))<1e-6
    for group,heads in [('all',list(range(14))),('selected',h['selected']),('random',h['random'])]:
     vals=np.array(recalls)[heads];suffix=np.array(h['suffix_target_recall'])[heads];mass=np.array(h['mass'])[heads]
     assert np.all((suffix>=0)&(suffix<=1))
     all_rows.append(dict(seed=seed,item_id=item,layer=layer,group=group,heads=heads,
                          mean_final_target_recall=float(vals.mean()),mean_suffix_target_recall=float(suffix.mean()),
                          final_all_target_present_fraction=float((vals==1).mean()),mean_target_probability_mass=float(mass.mean())))
 summary={}
 for g in ['all','selected','random']:
  rs=[x for x in all_rows if x['group']==g]
  summary[g]={key:float(np.mean([x[key] for x in rs])) for key in ['mean_final_target_recall','mean_suffix_target_recall','final_all_target_present_fraction','mean_target_probability_mass']}
 result=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,predictions=count,
             cumulative_task_predictions=7184+count,seconds=controller['seconds'],summary=summary,
             scope='Baseline-only per-head coverage audit, not causal importance or new candidate',archive=verification)
 out=R/'results/task-route-coverage-audit-v0';out.mkdir(exist_ok=False)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'per-layer-summary.json').write_text(json.dumps(all_rows,indent=2)+'\n')
 (out/'source.py').write_bytes(Path(__file__).read_bytes())
 lines=['# E1d补充：选中的头原本是否已经读到原文','','只重放原24题、两个种子的基线，新增48计分前向、0更新，不改变选头指标或干预。重新检查基线答案、logit、逐层头排名和随机头列表与E1d一致。','',
 '|头组|最后query目标块平均召回|后缀query目标块平均召回|最后query已覆盖全部目标块的头比例|每头平均目标块概率质量|','|---|---:|---:|---:|---:|']
 for g,label in [('all','全部14头'),('selected','选中的4头'),('random','随机4头')]:
  row=summary[g];lines.append('|'+label+'|'+'|'.join(f'{row[k]*100:.2f}%' for k in ['mean_final_target_recall','mean_suffix_target_recall','final_all_target_present_fraction','mean_target_probability_mass'])+'|')
 lines+=['','各题、层、种子等权；后缀召回先对该题后缀query平均。召回测原生路由实际选块；概率质量是同一稀疏路径Q/K上额外计算的密集归一化分数，两者不是同一种量。只记录最后query的具体块ID，并用它独立复算最后query召回；后缀召回由运行时张量计算，没有存储所有后缀ID。',
 '',f"被选头最后query未完整包含目标段的比例为{(1-summary['selected']['final_all_target_present_fraction'])*100:.2f}%。该值衡量是否还有补块空间，不能量化缺失信息的因果价值，也不能保证强制整个段落是正确修复。",'',
 'E1d的负结果仍只否定本次预定规则获得了恢复信号；不能说所有头干预完全相同、也不能由块召回推导唯一根因。这里不新增模型选择、训练或新题泛化结论。',
 '',f"本次作业{controller['seconds']:.2f}秒，累计计分前向{7184+count}。正式AMP训练仍15条3840更新＋43诊断更新。",'',
 '## UTC时间轴','','|任务|开始|结束|退出码|','|---|---|---|---:|']
 for x in controller['runs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|{x['returncode']}|")
 lines+=['',f"证据包：`exports/task-route-coverage-evidence-v0.tar.gz`；SHA256 `{proof['sha256']}`。",'']
 (R/'docs/task-route-coverage-results-2026-09-15.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()

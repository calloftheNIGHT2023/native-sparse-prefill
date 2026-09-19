"""Verify the isolated evidence package and audit every scored prediction."""
import hashlib, json, tarfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 archive=R/'exports/task-route-selective-evidence-v0.tar.gz'
 proof=read(archive.with_suffix('.json'));assert sha(archive)==proof['sha256']
 dest=R/'results/cloud-task-route-selective-evidence-v0';dest.mkdir(exist_ok=False)
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
 stage=dest/'results/task-route-selective-stage-v0';controller=read(stage/'result.json')
 assert controller['status']=='complete' and controller['optimizer_updates']==0
 protocol_path=dest/'provenance/task-quality-route-selective-protocol.json';protocol=read(protocol_path)
 assert controller['protocol_sha256']==sha(protocol_path)
 assert read(dest/'results/task-route-selective-preflight-v0/result.json')['status']=='passed'
 for name,h in protocol['sources'].items():assert sha(dest/name)==h
 assert len(controller['runs'])==3 and all(x['returncode']==0 for x in controller['runs'])
 ids=protocol['screen_ids'];conditions=protocol['conditions'];models=[];scores=[];prediction_count=0
 placement={x['item_id']:x for x in read(dest/'provenance/task-quality-route-chain-protocol.json')['rows']}
 for s in [0,1]:
  seed=2026091560+s;d=read(stage/f'seed{seed}'/'result.json')
  assert d['status']=='complete' and [d['k'],d['seed'],d['lr']]==[16,seed,.001] and d['optimizer_updates']==0
  assert d['route_protocol_sha256']==sha(protocol_path) and d['task_protocol_sha256']==sha(R/'data/task-quality-v0/protocol.json')
  assert max(d['calibration_replay_max_abs_error'],d['control_max_logit_error'],d['all_target_max_logit_error'])<=1e-6
  old=read(R/f'results/cloud-task-quality-evidence-v0/results/task-quality-stage-v0/sparse{s}/result.json')
  assert d['checkpoint_sha256']==old['checkpoint_sha256'] and d['training_result_sha256']==old['training_result_sha256']
  old={x['item_id']:x for x in old['predictions'] if x['variant']=='long16384'}
  prior=read(R/f'results/cloud-task-route-chain-evidence-v0/results/task-route-chain-stage-v0/screen-seed{seed}/result.json')
  prior={x['item_id']:x for x in prior['predictions'] if x['condition']=='target'}
  mapping={};assert len(d['predictions'])==120
  for c in conditions:
   xs={x['item_id']:x for x in d['predictions'] if x['condition']==c};assert set(xs)==set(ids)
   for item,x in xs.items():
    assert x['gold']==old[item]['gold'] and x['prediction']==int(np.argmax(x['choice_logits']))
    assert x['correct']==(x['prediction']==x['gold'])
    assert x['suffix_start']==placement[item]['suffix_start'] and x['target_blocks']==placement[item]['target_blocks']
    assert x['forced_blocks']==([] if c=='baseline' else placement[item]['sham_blocks' if c=='selected_sham' else 'target_blocks'])
    assert sorted(t['layer'] for t in x['layer_telemetry'])==list(range(24))
    for t in x['layer_telemetry']:
     layer=t['layer'];h=d['frozen_heads'][item][str(layer)]
     selected=sorted(range(14),key=lambda z:(-h['mass'][z],z))[:4];assert h['selected']==selected
     seedtext=f"{protocol['random_seed']}:{seed}:{item}:{layer}".encode()
     random_seed=int(hashlib.sha256(seedtext).hexdigest()[:16],16)
     random_heads=sorted(np.random.default_rng(random_seed).choice(14,4,replace=False).tolist());assert h['random']==random_heads
     intended=[] if c=='baseline' else (list(range(14)) if c=='all_target' else (random_heads if c=='random_target' else selected))
     assert t['heads']==intended
    if c in ['baseline','all_target']:
     ref=(old if c=='baseline' else prior)[item]
     assert x['prediction']==ref['prediction'] and np.max(np.abs(np.array(x['choice_logits'])-ref['choice_logits']))<=1e-6
   mapping[c]=xs
  models.append(mapping);scores.append({c:sum(x['correct'] for x in mapping[c].values()) for c in conditions});prediction_count+=len(d['predictions'])
 delta={c:[s['selected_target']-s[c] for s in scores] for c in protocol['gate']}
 passed=all(min(delta[c])>=g['each_seed'] and np.mean(delta[c])>=g['mean'] for c,g in protocol['gate'].items())
 assert bool(passed)==controller['gate']['passed'] and scores==controller['gate']['scores'] and delta==controller['gate']['delta_correct']
 arrays={c:np.array([[models[s][c][item]['correct'] for item in ids] for s in [0,1]],dtype=float) for c in conditions}
 rng=np.random.default_rng(2026091622);samples=rng.integers(0,24,(10000,24));contrasts=[]
 for control in ['baseline','all_target','selected_sham','random_target']:
  diff=(arrays['selected_target']-arrays[control]).mean(0)
  contrasts.append(dict(comparison='selected_target minus '+control,gap_pp=float(diff.mean()*100),
                        descriptive_95pct_interval_pp=(100*np.quantile(diff[samples].mean(1),[.025,.975])).tolist()))
 out=R/'results/task-route-selective-audit-v0';out.mkdir(exist_ok=False)
 audit=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,predictions=prediction_count,
            cumulative_task_predictions=6944+prediction_count,seconds=controller['seconds'],gate_passed=bool(passed),scores=scores,
            delta_correct=delta,contrasts=contrasts,archive=verification)
 (out/'result.json').write_text(json.dumps(audit,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
 lines=['# E1d结果：只干预更关注原文的少量头','','固定旧24题、两个K16高LR断点、16K输入；0更新。按基线最后query的目标块注意力质量每层选4/14头，然后冻结，目标/无关对照共用同一组头。已知目标位置和额外密集打分仅用于诊断，不是可部署方法。','',
 '|种子|原路由|所有头读原文|选4头读原文|同4头读无关块|随机4头读原文|','|---|---:|---:|---:|---:|---:|']
 for s,row in enumerate(scores):lines.append('|'+str(s)+'|'+'|'.join(f"{row[c]}/24" for c in conditions)+'|')
 lines+=['',f"冻结筛查：{'通过，仅支持进一步开发复核' if passed else '未通过，不据此训练选择器或扩大题集'}。",'',
 '|比较|平均变化pp|描述性95%区间pp|','|---|---:|---|']
 for c in contrasts:
  lo,hi=c['descriptive_95pct_interval_pp'];lines.append(f"|{c['comparison']}|{c['gap_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|")
 lines+=['','## 解释范围','','这是原24题的事后机制诊断。未扫描头数和层数，不把最高条件挑成新方法；按头选择已有Retrieval Head与QRHead等前人工作，详情见本轮协议。区间按题配对且保留两个种子，未做多重比较校正，不证明等价或泛化。',
 '','这里的分数是目标块概率质量，不是因果效应；块边界可能包括无关token。未改变的头在当层路由调用中保持其原生选块，但前层干预会改变后层表示。负结果不能证明所有选择性头方法都无效，更不能单独证明早期表示已损坏。',
 '','原始校准NLL、基线逐题logit、全部头强制目标的旧logit均在1e-6内重放。GPU预检含未干预头/前缀不变、固定预算、因果约束、GQA与独立FP32参照；归档逐项SHA核验并隔离还原。本地审计重新核对全部240预测、每题每层选头/随机种子及冻结门槛。',
 '',f"本轮{prediction_count}计分前向，累计{6944+prediction_count}；正式AMP训练仍15条3840更新＋43诊断更新。作业耗时{controller['seconds']:.2f}秒，不是账单；不据此重新宣称prefill或训练加速。",'',
 '## UTC时间轴','','|任务|开始|结束|退出码|','|---|---|---|---:|']
 for x in controller['runs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|{x['returncode']}|")
 lines+=['',f"证据包：`exports/task-route-selective-evidence-v0.tar.gz`；SHA256 `{proof['sha256']}`。",'',
 '[运行前协议](D:/ChatGPT/projects/native-sparse-prefill/docs/task-route-selective-protocol-2026-09-15.md)','']
 (R/'docs/task-route-selective-results-2026-09-15.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps(audit),flush=True)
if __name__=='__main__':main()

"""Verify the isolated evidence package and audit every scored prediction."""
import hashlib, json, tarfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 archive=R/'exports/task-attention-phases-evidence-v0.tar.gz'
 proof=read(archive.with_suffix('.json'));assert sha(archive)==proof['sha256']
 dest=R/'results/cloud-task-attention-phases-evidence-v0';dest.mkdir(exist_ok=False)
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
 stage=dest/'results/task-attention-phases-stage-v0';controller=read(stage/'result.json')
 protocol_path=dest/'provenance/task-attention-phases-protocol.json';protocol=read(protocol_path)
 assert controller['status']=='complete' and controller['optimizer_updates']==0
 assert controller['protocol_sha256']==sha(protocol_path)
 assert read(dest/'results/task-attention-phases-preflight-v0/result.json')['status']=='passed'
 assert len(controller['runs'])==3 and all(x['returncode']==0 for x in controller['runs'])
 for n,h in protocol['sources'].items():assert sha(dest/n)==h
 ids=protocol['screen_ids'];conditions=protocol['conditions'];placements={x['item_id']:x for x in protocol['rows']}
 models=[];scores=[];count=0
 for s in [0,1]:
  seed=2026091560+s;d=read(stage/f'seed{seed}'/'result.json')
  assert d['status']=='complete' and d['optimizer_updates']==0 and [d['k'],d['seed'],d['lr']]==[16,seed,.001]
  assert d['phase_protocol_sha256']==sha(protocol_path) and d['task_protocol_sha256']==sha(R/'data/task-quality-v0/protocol.json')
  assert max(d['calibration_replay_max_abs_error'],d['control_max_logit_error'],d['dense_control_max_logit_error'])<=1e-6
  sparse=read(R/f'results/cloud-task-quality-evidence-v0/results/task-quality-stage-v0/sparse{s}/result.json')
  dense=read(R/f'results/cloud-task-runtime-evidence-v0/results/task-runtime-stage-v0/sparse_weights_dense_eval{s}/result.json')
  assert d['checkpoint_sha256']==sparse['checkpoint_sha256']==dense['checkpoint_sha256']
  assert d['training_result_sha256']==sparse['training_result_sha256']==dense['training_result_sha256']
  old={c:{x['item_id']:x for x in source['predictions'] if x['variant']=='long16384'} for c,source in [('SS',sparse),('DD',dense)]}
  assert len(d['predictions'])==144;count+=len(d['predictions']);mapping={}
  for c in conditions:
   xs={x['item_id']:x for x in d['predictions'] if x['condition']==c};assert set(xs)==set(ids)
   for item,x in xs.items():
    assert x['gold']==old['SS'][item]['gold'] and x['prediction']==int(np.argmax(x['choice_logits']))
    assert x['correct']==(x['prediction']==x['gold'])
    place=placements[item];assert x['suffix_start']==place['suffix_start']
    span=place['target_span'] if c=='target_D' else (place['sham_span'] if c=='sham_D' else None)
    assert x['span']==span and x['sampled_prefix_positions']==[0,place['suffix_start']//2,place['suffix_start']-1]
    assert len(x['sampled_prefix_hidden_sha256'])==3
    assert sorted(t['layer'] for t in x['layer_telemetry'])==list(range(24))
    assert all(t['condition']==c and t['span']==span for t in x['layer_telemetry'])
    if c in old:
     p=old[c][item];assert x['prediction']==p['prediction'] and np.max(np.abs(np.array(x['choice_logits'])-p['choice_logits']))<=1e-6
   mapping[c]=xs
  for item in ids:
   for a,b in [('SS','SD'),('DS','DD')]:assert mapping[a][item]['sampled_prefix_hidden_sha256']==mapping[b][item]['sampled_prefix_hidden_sha256']
  models.append(mapping);scores.append({c:sum(x['correct'] for x in mapping[c].values()) for c in conditions})
 delta={candidate:{control:[z[candidate]-z[control] for z in scores] for control in controls} for candidate,controls in protocol['gate'].items()}
 passed={candidate:all(min(delta[candidate][control])>=g['each_seed'] and np.mean(delta[candidate][control])>=g['mean'] for control,g in controls.items()) for candidate,controls in protocol['gate'].items()}
 assert passed==controller['gate']['passed'] and scores==controller['gate']['scores'] and delta==controller['gate']['delta_correct']
 arrays={c:np.array([[models[s][c][item]['correct'] for item in ids] for s in [0,1]],dtype=float) for c in conditions}
 rng=np.random.default_rng(2026091624);samples=rng.integers(0,24,(10000,24));contrasts=[]
 for a,b in [('DS','SS'),('SD','SS'),('DD','SS'),('target_D','SS'),('target_D','sham_D')]:
  diff=(arrays[a]-arrays[b]).mean(0)
  contrasts.append(dict(comparison=a+' minus '+b,gap_pp=float(diff.mean()*100),
                        descriptive_95pct_interval_pp=(100*np.quantile(diff[samples].mean(1),[.025,.975])).tolist()))
 interaction=(arrays['DD']-arrays['DS']-arrays['SD']+arrays['SS']).mean(0)
 audit=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,predictions=count,
            cumulative_task_predictions=7232+count,seconds=controller['seconds'],gate_passed=passed,scores=scores,contrasts=contrasts,
            descriptive_accuracy_interaction_pp=float(interaction.mean()*100),archive=verification)
 out=R/'results/task-attention-phases-audit-v0';out.mkdir(exist_ok=False)
 (out/'result.json').write_text(json.dumps(audit,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
 lines=['# E1e结果：文章编码与问题读出分开切换注意力','','原24道16K开发题、两个K16高LR断点，所有条件共享各自稀疏训练权重，0更新。D/S指使用方式，不是新的训练方法。混合条件计算密集与稀疏两条路径后按query位置取输出，不能用本轮计时证明部署加速。','',
 '|种子|全稀疏SS|前缀密集DS|后缀密集SD|全密集DD|仅目标段query密集|等长无关段query密集|',
 '|---|---:|---:|---:|---:|---:|---:|']
 for i,z in enumerate(scores):lines.append('|'+str(i)+'|'+'|'.join(f'{z[c]}/24' for c in conditions)+'|')
 lines+=['',f"冻结门槛：前缀密集DS {'通过' if passed['DS'] else '未通过'}；目标段局部密集 {'通过' if passed['target_D'] else '未通过'}。未自动扩展题集或训练。",'',
 '|比较|准确率变化pp|描述性95%区间pp|','|---|---:|---|']
 for x in contrasts:
  lo,hi=x['descriptive_95pct_interval_pp'];lines.append(f"|{x['comparison']}|{x['gap_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|")
 lines+=['','## 这些结果能说明什么','','前缀指问题之前的全部示例、原文和干扰文本；后缀指问题、选项和Answer提示。target_D只改变目标原文范围内query的编码方式，不强制后缀读目标块。改变前缀会改变后层用于回答的K/V；这是一种路径干预，不能单独断言特定表示“损坏”。',
 '','局部目标组和无关组query数量相同，但位置和可见历史长度不同，不是等计算成本或严格内容因果对照。所有条件仍在旧题上探索，仅两个共享训练顺序的种子；区间按题配对，不校正多重比较，不证明独立泛化或等价。',
 '','SS和DD逐题选项logit分别与旧稀疏/密集运行结果在1e-6内重放。SS/SD与DS/DD在三个采样前缀位置的最终hidden state哈希分别相同；GPU单层检查验证了区间选择、未来K/V扰动不影响之前输出及独立FP32参照。完整前缀所有层激活没有逐元素存档，因此不夸大采样检查。',
 '',f"本轮{count}计分前向，累计{7232+count}；正式AMP训练仍15条3840更新＋43诊断更新。作业耗时{controller['seconds']:.2f}秒，不包含准备、传输、空闲与存储，不是账单。",'',
 '## UTC时间轴','','|任务|开始|结束|退出码|','|---|---|---|---:|']
 for x in controller['runs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|{x['returncode']}|")
 lines+=['',f"证据包：`exports/task-attention-phases-evidence-v0.tar.gz`；SHA256 `{proof['sha256']}`。",'',
 '[运行前协议](D:/ChatGPT/projects/native-sparse-prefill/docs/task-attention-phases-protocol-2026-09-15.md)','']
 (R/'docs/task-attention-phases-results-2026-09-15.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps(audit),flush=True)
if __name__=='__main__':main()

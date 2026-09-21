"""Verify the isolated evidence package and audit every scored prediction."""
import hashlib, json, tarfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 archive=R/'exports/task-attention-phases-full-evidence-v0.tar.gz'
 proof=read(archive.with_suffix('.json'));assert sha(archive)==proof['sha256']
 dest=R/'results/cloud-task-attention-phases-full-evidence-v0';dest.mkdir(exist_ok=False)
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
 stage=dest/'results/task-attention-phases-full-stage-v0';controller=read(stage/'result.json')
 protocol_path=dest/'provenance/task-attention-phases-full-protocol.json';protocol=read(protocol_path)
 assert controller['status']=='complete' and controller['optimizer_updates']==0
 assert controller['protocol_sha256']==sha(protocol_path)
 assert read(dest/'results/task-attention-phases-preflight-v0/result.json')['status']=='passed'
 assert len(controller['runs'])==2 and all(x['returncode']==0 for x in controller['runs'])
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
  assert len(d['predictions'])==len(ids)*len(conditions);count+=len(d['predictions']);mapping={}
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
 assert count==576 and controller['gate']['passed']=={}
 remaining_ids=list(ids);remaining_scores=scores
 prior_root=R/'results/cloud-task-attention-phases-evidence-v0'
 assert read(prior_root/'LOCAL-VERIFICATION.json')['status']=='verified'
 prior_protocol=read(prior_root/'provenance/task-attention-phases-protocol.json')
 assert prior_protocol['sources']['scripts/task_attention_phases.py']==protocol['sources']['scripts/task_attention_phases.py']
 first_ids=prior_protocol['screen_ids'];assert set(first_ids).isdisjoint(ids) and len(first_ids)==24
 normal_dense=[]
 for s in [0,1]:
  first=read(prior_root/f'results/task-attention-phases-stage-v0/seed{2026091560+s}/result.json')
  assert first['status']=='complete'
  for c in conditions:
   xs={x['item_id']:x for x in first['predictions'] if x['condition']==c};assert set(xs)==set(first_ids)
   models[s][c].update(xs)
  dense=read(R/f'results/cloud-task-quality-evidence-v0/results/task-quality-stage-v0/dense{s}/result.json')
  normal_dense.append({x['item_id']:x for x in dense['predictions'] if x['variant']=='long16384'})
 ids=first_ids+remaining_ids;assert len(ids)==len(set(ids))==96
 arrays={c:np.array([[models[s][c][item]['correct'] for item in ids] for s in [0,1]],dtype=float) for c in conditions}
 arrays['normal_dense']=np.array([[normal_dense[s][item]['correct'] for item in ids] for s in [0,1]],dtype=float)
 scores=[{c:int(a[s].sum()) for c,a in arrays.items()} for s in [0,1]]
 rng=np.random.default_rng(2026091625);samples=rng.integers(0,96,(10000,96));contrasts=[]
 for a,b in [('DS','SS'),('SD','SS'),('DD','SS'),('DS','normal_dense'),('SD','normal_dense'),('SS','normal_dense')]:
  diff=(arrays[a]-arrays[b]).mean(0)
  contrasts.append(dict(comparison=a+' minus '+b,gap_pp=float(diff.mean()*100),
                        descriptive_95pct_interval_pp=(100*np.quantile(diff[samples].mean(1),[.025,.975])).tolist()))
 splits=[]
 for name,part in [('first24',first_ids),('remaining72',remaining_ids)]:
  splits.append(dict(part=name,questions=len(part),normal_dense_correct=[sum(normal_dense[s][i]['correct'] for i in part) for s in [0,1]],
                    sparse_correct=[sum(models[s]['SS'][i]['correct'] for i in part) for s in [0,1]]))
 audit=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,new_predictions=count,
            combined_four_condition_predictions=768,phase_diagnostic_predictions_including_local_conditions=864,
            cumulative_task_predictions=7520+count,seconds=controller['seconds'],remaining_scores=remaining_scores,
            combined_scores=scores,contrasts=contrasts,sampling_audit=splits,archive=verification,
            scope='Post-hoc full development-set diagnostic; original 24-item gates remain failed; no new pass criterion')
 out=R/'results/task-attention-phases-full-audit-v0';out.mkdir(exist_ok=False)
 (out/'result.json').write_text(json.dumps(audit,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
 lines=['# 全96题：前缀编码与后缀读出的阶段路径诊断','','先前24题未呈现完整任务的密集优势，故在原筛查失败后，以单独协议补齐剩余72题。原失败门槛保留；这是事后诊断范围修订，不是通过原门槛或独立新题确认。所有旧96题完整保留，不挑密集对/稀疏错的子集。','',
 '## 完整结果','','除最后一行复用原正常密集训练结果外，所有条件共享稀疏训练断点。D/S分别表示使用密集/稀疏注意力，前一个字母为问题前缀，后一个为问题后缀。','',
 '|条件|种子0正确/96|种子1正确/96|平均准确率|','|---|---:|---:|---:|']
 for c,label in [('SS','全稀疏 SS'),('DS','前缀密集、后缀稀疏 DS'),('SD','前缀稀疏、后缀密集 SD'),('DD','稀疏权重全部密集运行 DD'),('normal_dense','正常密集训练、密集运行（旧记录参照）')]:
  lines.append(f"|{label}|{scores[0][c]}|{scores[1][c]}|{arrays[c].mean()*100:.2f}%|")
 lines+=['','## 配对差异','','|比较|差值pp|描述性95%区间pp|','|---|---:|---|']
 for x in contrasts:
  lo,hi=x['descriptive_95pct_interval_pp'];lines.append(f"|{x['comparison']}|{x['gap_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|")
 lines+=['','## 为什么24题筛查不能代表主要退步','','|开发集部分|正常密集种子0/1答对|全稀疏种子0/1答对|','|---|---|---|']
 for x in splits:lines.append(f"|{x['part']}（{x['questions']}题）|{x['normal_dense_correct']}|{x['sparse_correct']}|")
 lines+=['','这一发现限制了此前在同24题上的路由、加法、选头负诊断：它们检验的是该小样本的可恢复性，不能直接解释全96题的主要差距。之前的预测没有错误，但抽样覆盖不够。全96题补齐后仍是旧开发集探索，不是修正成独立验证。',
 '','本轮DS/SD混合路径在每层计算两种注意力后选择query区间，不是高效部署实现，不能继承旧9.2%训练节时作为新方法总收益。所有模型参数保持不变；DD不能替代正常密集训练参照。局部target_D/sham_D只在首24题执行，其结果见首轮报告，不与全96题混算。',
 '','剩余72题逐题SS/DD logit与旧记录误差≤1e-6；校准NLL、模型/数据/代码哈希、三个前缀位置的因果一致性全部核对，使用与首轮相同的已检验内核。两个种子数据顺序相同，置信区间按题配对，未经多重比较校正；差异不显著不等于等价。未新增通过门槛，未启动训练或新题确认。',
 '',f"本轮补齐{count}计分前向、{controller['seconds']:.2f}秒、0更新；连同首24题共864次阶段诊断前向，整个任务累计{7520+count}次计分前向。当前AMP正式训练仍15条3840更新＋43诊断更新。",'',
 '## UTC时间轴','','|任务|开始|结束|退出码|','|---|---|---|---:|']
 for x in controller['runs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|{x['returncode']}|")
 lines+=['',f"证据包：`exports/task-attention-phases-full-evidence-v0.tar.gz`；SHA256 `{proof['sha256']}`。",'',
 '[修订协议](D:/ChatGPT/projects/native-sparse-prefill/docs/task-attention-phases-full-protocol-2026-09-16.md)','']
 (R/'docs/task-attention-phases-full-results-2026-09-16.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps(audit),flush=True)
if __name__=='__main__':main()

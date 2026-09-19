"""Compare preserved-original-route additive control with fixed-budget E1a."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1];E=R/'results/cloud-task-route-additive-evidence-v1';A=R/'results/cloud-task-route-chain-evidence-v0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(E/'LOCAL-VERIFICATION.json')['status']=='verified'
protocol=read(E/'provenance/task-quality-route-additive-protocol.json');stage=E/'results/task-route-additive-stage-v0';controller=read(stage/'result.json')
assert controller['status']=='complete' and len(controller['runs'])==2 and all(x['returncode']==0 for x in controller['runs'])
assert sha(E/'scripts/task_route_additive.py')==protocol['route_source_sha256'] and read(E/'results/task-route-additive-preflight-v0/result.json')['status']=='passed'
ids=protocol['screen_ids'];placements={x['item_id']:x for x in protocol['rows']};models=[];scores=[]
for seed in [2026091560,2026091561]:
 d=read(stage/f'screen-seed{seed}/result.json');old=read(A/f'results/task-route-chain-stage-v0/screen-seed{seed}/result.json')
 assert d['status']=='complete' and d['optimizer_updates']==0 and len(d['predictions'])==72 and d['seed']==seed
 assert d['k']==16 and d['lr']==.001 and d['checkpoint_sha256']==old['checkpoint_sha256'] and d['training_result_sha256']==old['training_result_sha256']
 assert d['calibration_replay_max_abs_error']<=1e-6 and d['control_max_logit_error']<=1e-6
 assert d['route_protocol_sha256']==sha(E/'provenance/task-quality-route-additive-protocol.json') and d['task_protocol_sha256']==old['task_protocol_sha256']
 mapping={}
 for prefix,records in [('replace_',old['predictions']),('add_',d['predictions'])]:
  for c in ['baseline','target','sham']:
   xs={x['item_id']:x for x in records if x['condition']==c};assert set(xs)==set(ids)
   for i,x in xs.items():
    assert x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold'])
    if prefix=='add_':
     p=placements[i];assert x['block_budget_at_suffix']==(16 if c=='baseline' else 16+len(p['target_blocks']))
     assert x['forced_blocks']==(p['target_blocks'] if c=='target' else []) and x['suffix_start']==p['suffix_start']
   mapping[prefix+c]=xs
 for i in ids:
  a,b=mapping['add_baseline'][i],mapping['replace_baseline'][i]
  assert a['gold']==b['gold'] and a['prediction']==b['prediction'] and np.max(np.abs(np.array(a['choice_logits'])-b['choice_logits']))<=1e-6
 models.append(mapping);scores.append({c:sum(x['correct'] for x in xs.values()) for c,xs in mapping.items()})
arrays={c:np.array([[m[c][i]['correct'] for i in ids] for m in models],dtype=float) for c in models[0]}
comparisons=[('add_target','add_baseline'),('add_target','add_sham'),('add_sham','add_baseline'),('add_target','replace_target')];contrasts=[];rng=np.random.default_rng(2026091621);samples=rng.integers(0,24,(10000,24))
for a,b in comparisons:
 diff=(arrays[a]-arrays[b]).mean(0);contrasts.append(dict(comparison=a+' minus '+b,gap_pp=100*float(diff.mean()),descriptive_95pct_interval_pp=(100*np.quantile(diff[samples].mean(1),[.025,.975])).tolist()))
lines=['# E1c：保留原路由，只额外补信息','','E1a固定K16替换目标段没有改善，本支线用于判断是否因为替换挤掉其他有用块。它另用增加预算的干预，不修改E1a失败门槛，不据此直接训练选择器。全部仍是原24题、两种子、事后诊断。','','|种子|原路由K16|替换目标段|替换无关块|额外补目标段|额外补无关块|','|---|---:|---:|---:|---:|---:|']
for i,s in enumerate(scores):lines.append(f"|{i}|{s['add_baseline']}/24|{s['replace_target']}/24|{s['replace_sham']}/24|{s['add_target']}/24|{s['add_sham']}/24|")
budgets=[16+len(placements[i]['target_blocks']) for i in ids]
lines += ['',f'增加预算两组均完整保留原16块，再补F块；本批F={min(budgets)-16}–{max(budgets)-16}，最终每个后缀query/head选{min(budgets)}–{max(budgets)}块。目标组先确保完整目标段再用无关块补足F，无关组只补F个未选无关块；两组每题预算相同。不是K16方案，也不宣称加速。','','|比较|平均差值pp|描述性95%区间pp|','|---|---:|---|']
for x in contrasts:
 lo,hi=x['descriptive_95pct_interval_pp'];lines.append(f"|{x['comparison']}|{x['gap_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|")
lines+=['','归档v0漏收实际加法协议，v1补齐并重新做逐项哈希校验，没有改动预测。实际控制器读取父协议1800秒总上限、单任务750秒，未接入原拟750秒总上限；本次实际75.05秒结束。此处记录实现偏差，未因此重跑模型。','','原路由所有块保留、两加法组预算相同、无重复及无未来块均在每次路由运行中断言。基线与旧题全部选项logit重放一致。额外目标组FP32独立数值参照通过，E1c预检也比较两种增加预算模式的有效选块数相同。','','该干预不是严格最优路由上界；对所有头强制目标段会改变注意力分布。即使不提升，也不能排除特定头/层、原文内部编码或信息传播的损伤。反之，额外预算改善也不能证明存在同K预算的可实现算法。上述区间旧题事后描述、未校正多组比较，不是泛化或等价证明。','','下一步如继续机制研究，应先区分早期表示受损与读出阶段干扰，并与已有因果选择器工作全文对照；不能直接进入选择器训练或扩大GPU预算。','','## UTC时间轴','','|任务|开始|结束|退出码|','|---|---|---|---:|']
for x in controller['runs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|{x['returncode']}|")
result=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,predictions=144,scores=scores,accuracy={c:float(v.mean()) for c,v in arrays.items()},contrasts=contrasts,seconds=controller['seconds'],suffix_block_budget_range=[min(budgets),max(budgets)],automatic_training_authorized_by_result=False)
out=R/'results/task-route-additive-audit-v0';out.mkdir(exist_ok=False);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes());(R/'docs/task-route-additive-results-2026-09-15.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(result))

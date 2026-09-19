"""Audit locked-seed results and report the measured quality/cost tradeoff."""
import json,math,hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
E=R/'results/cloud-amp-confirmation-evidence-v1';S=E/'results/amp-recovery-stage-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 v=read(E/'LOCAL-VERIFICATION.json');assert v['status']=='verified'
 control=read(S/'result.json');assert control['status']=='complete' and len(control['runs'])==4 and control['optimizer_updates']==0
 training_control=read(E/'results/amp-recovery-stage-v0/result.json')
 protocol=read(E/'provenance/amp-recovery-confirmation-protocol.json')
 assert len(protocol['heldout_indices'])==29 and len(set(protocol['heldout_indices']))==29
 idx=set(protocol['heldout_indices']);assert not idx.intersection(protocol['excluded_training_indices']) and not idx.intersection(protocol['excluded_previous_report_indices'])
 assert not any(i*16385<b and (i+1)*16385>a for i in idx for a,b in protocol['excluded_prior_training_intervals'])
 assert sha(E/'data/amp-confirmation-v0/heldout.npz')==protocol['heldout_sha256']
 old=R/'results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2';boundary=R/'results/cloud-amp-lr-boundary-evidence-v0/results/amp-recovery-stage-v0'
 parents={(0,2026091560):old/'cal-k0-lr0.0003',(0,2026091561):old/'repeat-k0',(16,2026091560):boundary/'cal-k16-lr0.001',(16,2026091561):E/'results/amp-recovery-stage-v0/repeat-k16'}
 rows=[];diffs=[];sources=[]
 for seed in [2026091560,2026091561]:
  ev={};train={}
  for k in [0,16]:
   p=S/f'heldout-k{k}-seed{seed}/result.json';ev[k]=read(p);t=parents[k,seed];train[k]=read(t/'result.json')
   assert ev[k]['status']=='complete' and ev[k]['optimizer_updates']==0 and len(ev[k]['values'])==29
   assert ev[k]['k']==k and ev[k]['seed']==seed and ev[k]['lr']==(.0003 if k==0 else .001)
   assert ev[k]['calibration_replay_max_abs_error']<=1e-6
   assert ev[k]['training_result_sha256']==sha(t/'result.json') and ev[k]['checkpoint_sha256']==read(t/'checkpoint-index.json')['sha256']
   assert ev[k]['protocol_sha256']==sha(E/'provenance/amp-recovery-confirmation-protocol.json')
   assert abs(sum(x['seconds'] for x in train[k]['trace'])-train[k]['training_seconds'])<1e-8
   assert abs(np.mean(ev[k]['values'])-ev[k]['mean_nll'])<1e-12
   sources.append(dict(path=p.relative_to(R).as_posix(),sha256=sha(p)))
  assert train[0]['initial_sha256']==train[16]['initial_sha256']
  assert [x['window_index'] for x in train[0]['trace']]==[x['window_index'] for x in train[16]['trace']]
  assert train[0]['identity']['sources']==train[16]['identity']['sources'] and train[0]['environment']==train[16]['environment']
  gap=np.asarray(ev[16]['values'])-np.asarray(ev[0]['values']);diffs.append(gap)
  time_ratio=train[16]['training_seconds']/train[0]['training_seconds']
  rows.append(dict(seed=seed,dense_nll=ev[0]['mean_nll'],sparse_nll=ev[16]['mean_nll'],gap=float(gap.mean()),ppl_increase_percent=100*math.expm1(float(gap.mean())),training_time_ratio=time_ratio,time_saved_percent=100*(1-time_ratio),paired_window_gaps=gap.tolist(),sparse_worse_windows=int((gap>0).sum()),strict_screen_pass=float(gap.mean())<=.03 and time_ratio<=.95))
 array=np.stack(diffs);rng=np.random.default_rng(2026091569);samples=rng.integers(0,29,size=(10000,29));boot=array.mean(0)[samples].mean(1)
 intervals=np.quantile(boot,[.025,.975]).tolist()
 sensitivity=[dict(ppl_tolerance_percent=t,both_seeds_within_tolerance_and_5pct_faster=all(r['gap']<=math.log1p(t/100) and r['training_time_ratio']<=.95 for r in rows)) for t in [3,5,10]]
 search_cost=[]
 for k in [0,16]:
  paths=[old/f'cal-k{k}-lr{lr:g}' for lr in [.00003,.0001,.0003]]+[boundary/f'cal-k{k}-lr0.001',parents[k,2026091561]]
  trials=[read(p/'result.json') for p in paths]
  assert all(t['status']=='complete' and t['optimizer_updates_this_process']==256 for t in trials)
  search_cost.append(dict(k=k,runs=5,training_seconds=sum(t['training_seconds'] for t in trials),process_seconds=sum(t['wall_seconds'] for t in trials)))
 result=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),new_scientific_updates=256,new_diagnostic_updates=6,heldout_conditions=4,windows=29,rows=rows,sensitivity=sensitivity,
  mean_seed_gap=float(array.mean()),descriptive_paired_window_bootstrap_95pct_interval=intervals,strict_screen_both_seeds=all(x['strict_screen_pass'] for x in rows),
  scope=protocol['scope'],bootstrap_limit='Shared window resampling across seeds; correlated text windows and shared training order. Not a guarantee across tasks/models.',controller_seconds=control['seconds']+training_control['seconds'],evaluation_only_seconds=control['seconds'],search_cost=search_cost,sources=sources)
 out=R/'results/amp-confirmation-audit-v0';out.mkdir(exist_ok=False);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
 lines=['# 稀疏适配质量—成本确认实验','','固定dense LR3e-4、K16 LR1e-3，均256步；仅新增K16第二种子一条训练。29个新保留位置在配置与模型选择锁定后评测，排除已知Qwen训练位置和上轮report位置。不是跨语料或下游任务验证。','','|种子|dense NLL|K16 NLL|NLL差|困惑度增加|训练时间减少|原严格筛查|','|---|---:|---:|---:|---:|---:|---|']
 for x in rows:lines.append(f"|{x['seed']}|{x['dense_nll']:.6f}|{x['sparse_nll']:.6f}|{x['gap']:.6f}|{x['ppl_increase_percent']:.2f}%|{x['time_saved_percent']:.2f}%|{'通过' if x['strict_screen_pass'] else '未通过'}|")
 lines+=['','质量接近不要求超过dense。是否值得采用，取决于应用能容忍的损失与时间收益；这张表提供实测取舍，不能把困惑度增加直接解释为任务准确率下降。原严格筛查仍为NLL差≤0.03且训练时间比≤0.95，不改写旧门槛。','','|允许的困惑度增加|两个种子均在范围内且训练至少快5%|','|---|---|']
 for x in sensitivity:lines.append(f"|{x['ppl_tolerance_percent']}%|{'是' if x['both_seeds_within_tolerance_and_5pct_faster'] else '否'}|")
 lines+=['','双方均计入四点学习率搜索和一次选中配置的第二种子复跑，费用口径如下。进程时间包含加载/校准/存档，但不含独立预检、最终评测、安装、GPU空闲和存储。','','|方法|训练轨迹数|累计训练分钟|累计训练进程分钟|','|---|---:|---:|---:|']
 for x in search_cost:lines.append(f"|{'dense' if x['k']==0 else 'K16'}|{x['runs']}|{x['training_seconds']/60:.2f}|{x['process_seconds']/60:.2f}|")
 lines+=['','3/5/10%是本轮预先声明的描述性容忍度，不能代替具体任务需求，也不意味着论文或原标准自动通过。',
  '',f"两种子平均配对NLL差为{array.mean():.6f}。共享窗口抽样的描述性95%bootstrap区间为[{intervals[0]:.6f}, {intervals[1]:.6f}]；29个语料窗口可能相关、仅两个初始化种子且训练数据顺序相同，不将此解释为跨任务泛化保证。",
  '',f"新增256科学更新与6个预检诊断更新分开记录，4项最终评测不更新参数。训练阶段含首次失败评测及补评测控制器合计耗时{result['controller_seconds']/60:.2f}分钟，不含中间空闲、迁移/下载/存储。时间减少比较各选中256步训练轨迹，不包含历次搜索成本；真实账单未知。所有4个断点均先复算校准结果且最大误差≤1e-6，才打开新保留数据。",
  '', '首次评测因原配置CRLF与快照LF的字节哈希不同而在读取保留数据前停止。修复后同时验证原配置字节SHA和快照JSON内容一致，未放宽数值门槛。原失败日志及配置保留；仅补评测，没有重训。',
  '', '这仍是密集预训练0.5B模型的LoRA稀疏适配，尚不是原生全参数稀疏预训练。它能提供特定设置下的质量—时间证据，不能单独确认算法新颖性、下游任务等价性或论文贡献。',
  '', '## UTC时间轴','','|任务|开始|结束|退出码|','|---|---|---|---:|']
 for x in training_control['runs']+control['runs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|{x['returncode']}|")
 (R/'docs/amp-confirmation-results-2026-09-15.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ['rows','sources']}))
if __name__=='__main__':main()

"""Verify preprocessing correction and integrate both formats without outcome-based choice."""
import json,hashlib,tarfile,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 archive=R/'exports/32k-separator-replay-evidence-v0.tar.gz';proof=read(archive.with_suffix('.json'));assert sha(archive)==proof['sha256']
 dest=R/'results/cloud-32k-separator-replay-evidence-v0';dest.mkdir(exist_ok=False)
 with tarfile.open(archive) as t:
  members=t.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))
  manifest=json.load(t.extractfile('32k-separator-replay-manifest.json'));expected={x['path']:x for x in manifest['files']};assert set(names)==set(expected)|{'32k-separator-replay-manifest.json'}
  for m in members:
   assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   p=(dest/m.name).resolve();assert dest.resolve() in p.parents;raw=t.extractfile(m).read()
   if m.name in expected:
    e=expected[m.name];assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256']
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 verification=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),sha256=proof['sha256'],files=len(expected));dump(dest/'LOCAL-VERIFICATION.json',verification)
 pp=dest/'provenance/32k-separator-replay-protocol.json';protocol=read(pp);assert sha(pp)==sha(R/'provenance/32k-separator-replay-protocol.json')
 assert sha(dest/'scripts/eval_32k_separator_replay.py')==protocol['evaluator_sha256'] and sha(dest/'scripts/run_32k_separator_replay.py')==protocol['controller_sha256']
 assert sha(dest/'data/32k-separator-replay-v0/report.npz')==protocol['data_sha256']
 original=R/'results/cloud-32k-adaptation-evidence-v0';audit=read(R/'results/32k-adaptation-audit-v0/result.json');assert audit['status']=='complete'
 cfg=read(original/'data/32k-adaptation-v0/config.json');assert sha(original/'data/32k-adaptation-v0/config.json')==protocol['original_config_sha256']
 oldstage=original/'results/32k-adaptation-stage-v0';oldcontroller=read(oldstage/'result.json')
 first_report=min(x['started_utc'] for x in oldcontroller['runs'] if x['name'].startswith('report-'))
 assert protocol['frozen_utc']<first_report and protocol['controller_frozen_utc']<first_report
 stage=dest/'results/32k-separator-replay-stage-v0';controller=read(stage/'result.json');assert controller['status']=='complete' and controller['optimizer_updates']==0 and len(controller['runs'])==6
 assert controller['protocol_sha256']==sha(pp) and all(x['returncode']==0 for x in controller['runs'])
 values={};meta=[]
 for job in controller['runs']:
  p=stage/job['name'];r=read(p/'result.json');assert r['phase']=='report' and r['status']=='complete' and r['optimizer_updates_this_process']==0
  k=r['identity']['k'];seed=r['identity']['seed'];step=r['step'];lr=audit['selected_lrs'][str(k)]
  assert r['identity']['lr']==lr and r['identity']['sources']==cfg['sources_sha256'] and r['identity']['config_sha256']==protocol['original_config_sha256']
  parent=oldstage/(f'cal-k{k}-lr{lr:g}' if seed==cfg['seeds'][0] else f'repeat-k{k}')
  replay=read(p/'replay-verification.json');assert replay['checkpoint_sha256']==sha(parent/f'checkpoint-{step}.pt') and replay['max_abs_error']<=1e-6
  proofrow=read(p/'separator-replay.json');assert proofrow['protocol_sha256']==sha(pp) and proofrow['data_sha256']==protocol['data_sha256'] and proofrow['checkpoint_sha256']==replay['checkpoint_sha256']
  ev=r['evaluations'];assert len(ev)==1 and ev[0]['step']==step and ev[0]['split']=='report' and len(ev[0]['values'])==9
  assert np.isfinite(ev[0]['values']).all() and abs(np.mean(ev[0]['values'])-ev[0]['mean_nll'])<1e-10
  values[k,seed,step]=ev[0]['values'];meta.append(dict(k=k,seed=seed,step=step,ppl=float(np.exp(np.mean(ev[0]['values'])))))
 arrays={k:np.array([values[k,s,64] for s in cfg['seeds']]) for k in [0,32]}
 diff=(arrays[32]-arrays[0]).mean(0);rng=np.random.default_rng(2026091690);samples=rng.integers(0,9,(10000,9))
 ppl={str(k):float(np.exp(arrays[k].mean())) for k in [0,32]};gap=100*math.expm1(diff.mean());ci=(100*np.expm1(np.quantile(diff[samples].mean(1),[.025,.975]))).tolist()
 result=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),archive=verification,protocol_frozen_before_first_report=True,optimizer_updates=0,task_predictions=0,report_window_forwards=54,calibration_replay_forwards=24,seconds=controller['seconds'],corrected_format_ppl=ppl,corrected_format_ppl_increase_pct=gap,corrected_format_descriptive_95_interval=ci,per_checkpoint=meta,original_format_ppl_increase_pct=audit['comparison']['ppl_increase_pct'])
 out=R/'results/32k-separator-replay-audit-v0';out.mkdir(exist_ok=False);dump(out/'result.json',result);(out/'source.py').write_bytes(Path(__file__).read_bytes())
 comp=audit['comparison'];points=audit['selected_points']
 lines=['# 32K质量与训练成本：最终汇总（含文本格式复核）','',f"本轮完成6条64步训练、双种子选中配置复核。密集与K32都选择LR0.001。固定训练量下K32节时 **{comp['time_saved_pct']:.2f}%**，与训练拼接格式一致的测试PPL增加 **{gap:.2f}%**，32K阅读准确率差 **{comp['task_gap_pp']['long32768']:+.2f}个百分点**。目标仍是接近质量与更低成本，不要求稀疏质量反超。",'',
 '## 主结果','','下表PPL使用与train/validation一致的双换行拼接；任务构造与训练时间沿用原冻结实验，未改动。','', '|方法|平均训练秒|测试PPL|短题|无原文|32K阅读|','|---|---:|---:|---:|---:|---:|']
 for p in points:lines.append(f"|{'密集' if p['k']==0 else 'K32'}|{p['mean_training_seconds']:.2f}|{ppl[str(p['k'])]:.4f}|{p['accuracy']['short']:.2f}%|{p['accuracy']['no_context']:.2f}%|{p['accuracy']['long32768']:.2f}%|")
 lines += ['',f"双换行测试PPL增加的描述性配对95%区间为[{ci[0]:.2f}%, {ci[1]:.2f}%]。只有9个相邻语料窗口、64道题与两个共享数据顺序的初始化种子；区间不是正式等价证明。任务区间、所有候选和早停校准曲线在原完整报告中。",'',
 '## 为什么补做文本格式复核','',
 '复查发现训练/校准采用双换行连接WikiText行，最初测试准备使用单换行。两种模型原本仍在相同输入上比较，但这引入了额外格式变化。发现问题后，在任何最终测试结果出现前冻结了双换行重评，未改变训练、学习率选择或任务题目。两份PPL都保留，不能根据结果挑格式。',
 f"修订协议冻结UTC {protocol['frozen_utc']}，控制器冻结UTC {protocol['controller_frozen_utc']}；原实验首次最终评测开始UTC {first_report}。本地审计核对时序。",'',
 '|测试拼接口径|密集PPL|K32 PPL|PPL增加|','|---|---:|---:|---:|']
 lines.append(f"|原单换行|{points[0]['ppl']:.4f}|{points[1]['ppl']:.4f}|{comp['ppl_increase_pct']:+.2f}%|")
 lines.append(f"|与训练一致的双换行|{ppl['0']:.4f}|{ppl['32']:.4f}|{gap:+.2f}%|")
 lines += ['','相同的test源文件，不同拼接会移动窗口边界，因此两口径不是额外独立样本。新增重评只有54个测试窗口前向和24个校准重放，0新优化更新、0新任务预测。','',
 '## 未适配基线（同格式PPL）','','|注意力|0步测试PPL|','|---|---:|']
 for k in [0,32]:lines.append(f"|K{k}|{np.exp(np.mean(values[k,cfg['seeds'][0],0])):.4f}|")
 lines += ['','## 记录与边界','',f"完整原实验控制器{audit['controller_seconds']:.2f}秒，格式重评{controller['seconds']:.2f}秒；两者均不代表完整租卡账单。科学训练384更新、诊断12更新、任务1152次计分前向；原始断点、训练轨迹和所有预测都已逐项审计，未因格式问题重训。",'这是密集预训练Qwen2.5-0.5B的32K LoRA适配；WikiText为跨行/文章的语料流，RACE任务为带TARGET标记的人工扩展，非原生全参数稀疏训练或官方长文基准。短题K32覆盖所有块，短题相近不能替代32K结果。预训练接触未排除，不据此宣称普遍等价或原创论文已经完成。','',
 '- 原完整训练/任务报告：[32K完整报告](32k-adaptation-results-2026-09-16.md)',
 '- 逐作业UTC与更新计数：[训练时间轴](32k-adaptation-timeline-2026-09-16.md)',
 '- 原实验协议：[冻结协议](32k-adaptation-protocol-2026-09-16.md)',
 f"- 原证据SHA256：{audit['archive']['sha256']}",f"- 格式重评证据SHA256：{verification['sha256']}",'']
 content='\n'.join(lines).replace('## 记录与边界','## 成本结论的准确范围\n\n固定64步、相同训练token曝光下，K32确实节省17.44%的训练循环时间。另一个必须保留的参照是：未适配密集模型的同格式PPL为11.0169，仍略好于当前K32的11.2053。因此，本轮成立的是固定工作量下的节时与质量折中，还没有建立“达到当前稀疏质量时最便宜”的优势。这个比较不要求稀疏质量超过密集，只要求成本结论使用对应的基线。\n\n任务的无原文成绩较高，说明猜题能力不可忽略。原文位置分组使用不同问题，每组仅16题，不能把分组差异直接解释成位置的因果作用。后续优先做相同训练用时对照，并保留密集早停/0步基线；不要通过删掉基线或只挑某个位置组来制造接近效果。\n\n'+'## 记录与边界')
 (R/'docs/32k-adaptation-final-results-2026-09-16.md').write_text(content,encoding='utf-8')
 print(json.dumps(result))
if __name__=='__main__':main()

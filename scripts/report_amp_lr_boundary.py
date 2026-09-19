"""Audit a completed, verified extra-LR probe without opening report data."""
import json,hashlib,math,statistics
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1]
OLD=R/'results/cloud-amp-recovery-final-evidence-v0'
NEW=R/'results/cloud-amp-lr-boundary-evidence-v0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert read(NEW/'LOCAL-VERIFICATION.json')['status']=='verified'
 stage=NEW/'results/amp-recovery-stage-v0';control=read(stage/'result.json')
 assert control['status']=='complete' and len(control['runs'])==4
 oldcfg=read(OLD/'data/flashmoba-amp-recovery-v0/config.json');cfg=read(NEW/'data/flashmoba-amp-recovery-v0/config.json')
 diffs={k for k in oldcfg.keys()|cfg.keys() if oldcfg.get(k)!=cfg.get(k)}
 assert diffs<=set(['learning_rates','methods','seeds','maximum_controller_seconds','prepared_utc','selection_rule','scope']),diffs
 assert not (NEW/'data/flashmoba-amp-recovery-v0/report.npz').exists()
 seed=2026091560;runs=[];evidence=[]
 for k in [0,16]:
  gate=read(stage/f'preflight-k{k}/result.json');assert gate['status']=='complete' and all(x['passed'] for x in gate['gates'])
  for lr in [.00003,.0001,.0003,.001]:
   path=(stage if lr==.001 else OLD/'results/amp-recovery-stage-v2')/f'cal-k{k}-lr{lr:g}'/'result.json'
   d=read(path);assert d['status']=='complete' and d['step']==256 and d['optimizer_updates_this_process']==256
   assert len(d['trace'])==256 and all(x['split']=='calibration' for x in d['evaluations'])
   assert [x['step'] for x in d['evaluations']]==[0,64,128,256]
   assert all(math.isfinite(x['train_nll']) and math.isfinite(x['gradient_norm']) for x in d['trace'])
   assert abs(sum(x['seconds'] for x in d['trace'])-d['training_seconds'])<1e-8
   assert abs(statistics.mean(d['evaluations'][-1]['values'])-d['evaluations'][-1]['mean_nll'])<1e-12
   runs.append(dict(k=k,lr=lr,nll=d['evaluations'][-1]['mean_nll'],values=d['evaluations'][-1]['values'],training_seconds=d['training_seconds'],wall_seconds=d['wall_seconds'],calibration=d['evaluations'],initial_sha=d['initial_sha256'],order=[x['window_index'] for x in d['trace']],sources=d['identity']['sources'],environment=d['environment']))
   evidence.append(dict(path=path.relative_to(R).as_posix(),sha256=sha(path)))
 assert len({r['initial_sha'] for r in runs})==1
 assert len({tuple(r['order']) for r in runs})==1
 assert len({json.dumps(r['sources'],sort_keys=True) for r in runs})==1
 assert len({json.dumps(r['environment'],sort_keys=True) for r in runs})==1
 best={k:min((r for r in runs if r['k']==k),key=lambda r:(r['nll'],r['lr'])) for k in [0,16]}
 prior={k:next(r for r in runs if r['k']==k and r['lr']==.0003) for k in [0,16]}
 high={k:next(r for r in runs if r['k']==k and r['lr']==.001) for k in [0,16]}
 oldgap=prior[16]['nll']-prior[0]['nll'];newgap=best[16]['nll']-best[0]['nll']
 ratio=best[16]['training_seconds']/best[0]['training_seconds']
 result=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),scientific_updates=512,diagnostic_updates=12,report_evaluations=0,
  previous_best_gap=oldgap,new_best_gap=newgap,gap_reduction=oldgap-newgap,
  matched_high_lr_gap=high[16]['nll']-high[0]['nll'],selected_lrs={str(k):v['lr'] for k,v in best.items()},
  selected_training_time_ratio=ratio,second_seed_motivated=oldgap-newgap>=.02,
  calibration_quality_screen=newgap<=.03 and ratio<=.95,controller_seconds=control['seconds'],
  configuration_changed_fields=sorted(diffs),rows=[{k:v for k,v in r.items() if k not in ['order','sources','environment']} for r in runs],inputs=evidence,
  limitations=['Post-hoc development diagnostic','Four reused calibration windows','One initialization seed','No new report evaluation','No claim of novel ML contribution'])
 out=R/'results/amp-lr-boundary-audit-v0';out.mkdir(exist_ok=False)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
 lines=['# 学习率边界诊断结果','','本轮完成两条256步正式训练（512更新）及两次预检（12诊断更新）；仅读校准集，无新report评测。所有新原始文件已逐项校验。','','|方法|峰值LR|256步校准NLL|训练秒数|','|---|---:|---:|---:|']
 for r in runs:lines.append(f"|{'dense' if r['k']==0 else 'K16'}|{r['lr']:g}|{r['nll']:.6f}|{r['training_seconds']:.2f}|")
 lines+=['',f"两方法完整四点网格各选最优终点：dense LR={best[0]['lr']:g}，K16 LR={best[16]['lr']:g}。K16与dense校准NLL差由 {oldgap:.6f} 变为 {newgap:.6f}，缩小 {oldgap-newgap:.6f} nats。相同1e-3对照差为 {result['matched_high_lr_gap']:.6f}。",
  '',f"选中轨迹训练时间比为 {ratio:.6f}，困惑度相对差按exp(NLL差)-1换算约 {100*math.expm1(newgap):.2f}%。都是开发集结果；选中运行的训练时间不包括搜索成本。新增两条进程总耗时 {sum(high[k]['wall_seconds'] for k in [0,16])/60:.2f} 分钟，整个控制器含预检 {control['seconds']/60:.2f} 分钟；启动、下载和存储另计，真实账单未查询。",
  '',('达到事先冻结的差距缩小0.02门槛，值得第二种子复查。' if result['second_seed_motivated'] else '未达到事先冻结的差距缩小0.02门槛，不凭此次学习率诊断扩大训练。'),
  '',('开发集达到0.03质量差与至少5%训练节时的联合筛查，但不能代替新保留数据评测。' if result['calibration_quality_screen'] else '开发集仍未同时达到0.03质量差和至少5%训练节时的联合筛查。'),
  '', '已核验旧新实验的初始化哈希、数据顺序、训练源码、环境相同，配置差异只限于公开的实验范围和元信息。保留全部学习率结果，不挑选中途最低点。',
  '', '本实验在看到上一轮report后设计，因此属于开发诊断。只有一个初始化种子、四个反复使用的校准窗口；不能作为独立泛化验证，更没有证明新的稀疏算法贡献。后续确认必须另锁协议和新的留出评测，旧report不得冒充未见数据。',
  '', '## UTC时间轴','','|任务|开始|结束|退出码|','|---|---|---|---:|']
 for row in control['runs']:lines.append(f"|{row['name']}|{row['started_utc']}|{row['finished_utc']}|{row['returncode']}|")
 (R/'docs/amp-lr-boundary-results-2026-09-15.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ['rows','inputs']}))
if __name__=='__main__':main()

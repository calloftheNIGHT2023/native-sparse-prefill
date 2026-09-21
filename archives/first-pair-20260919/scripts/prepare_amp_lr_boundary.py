"""Freeze isolated input bundle; preserve the already-tested training source."""
import json,hashlib,shutil,tarfile
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=R/'exports/amp-lr-boundary-input-v0';out.mkdir(exist_ok=False)
 for d in ['scripts','docs','data/flashmoba-amp-recovery-v0','provenance']:(out/d).mkdir(parents=True,exist_ok=True)
 names=['run_amp_recovery.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','run_amp_recovery_boundary.py','package_amp_recovery_results.py']
 for n in names:shutil.copy2(R/'scripts'/n,out/'scripts'/n)
 cfg=json.loads((R/'data/flashmoba-amp-recovery-v0/config.json').read_text())
 cfg.update(learning_rates=[.001],methods=[0,16],seeds=[2026091560],maximum_controller_seconds=1500,
  prepared_utc=datetime.now(timezone.utc).isoformat(),selection_rule='Calibration-only extra LR boundary probe; no report evaluation or final selection lock.',
  scope='Post-hoc development diagnostic selected after earlier report was seen. Dense and K16 receive the same additional LR trial; no confirmatory held-out claims.')
 (out/'data/flashmoba-amp-recovery-v0/config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 shutil.copy2(R/'data/flashmoba-amp-recovery-v0/train-calibration.npz',out/'data/flashmoba-amp-recovery-v0/train-calibration.npz')
 protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),question='Was the previous LR grid too small for K16 quality recovery?',
  peak_lr=.001,steps=256,methods=[0,16],seed=2026091560,from_pretrained=True,
  scientific_updates=512,planned_preflight_updates=12,maximum_controller_seconds=1500,per_process_timeout_seconds=750,
  primary='Step256 calibration NLL: compare new K16 to new dense and each method to its prior 3e-4 run.',
  secondary='Full calibration trajectories, finite gradients, measured training seconds, all per-window values.',
  development_decision='A reduction of >=0.02 nats in the best-of-grid K16-minus-dense calibration gap motivates a second seed; otherwise do not scale on this LR probe alone. This is an engineering follow-up rule, not a significance test.',
  previous_grid=[.00003,.0001,.0003],new_trial_equal_for_both=True,
  immutable_training_source={n:sha(out/'scripts'/n) for n in names[:4]},
  exclusions=['No report.npz in bundle','No K4 run','No resume with changed LR','No automatic retry','No claim of a new method or held-out success'])
 (out/'provenance/amp-recovery-lr-boundary-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 doc='''# 学习率边界补充诊断（冻结协议）

目的：判断原网格上界3e-4是否限制K16恢复。dense和K16各增加一次峰值1e-3、256步、种子2026091560的从头训练；其他训练实现及数据不变。K4暂不追加。先分别通过既有数值/恢复预检，12个诊断更新与512个科学更新分开。

本实验在旧report已被查看后设计，只读取原四个校准窗口，不携带或读取report.npz。主要看第256步NLL，不能挑中途最好的点；与两方法各自旧3e-4结果及新1e-3对照一起报告。包含两方法完整四点LR网格的各自最佳终点差距若比原来缩小至少0.02 nats，则值得第二种子复查；否则不凭本轮扩大。0.02只是工程决策阈值，不是显著性检验。

每个进程最多750秒、控制器最多1500秒，无自动重试。相同卡按旧轨迹预计12–15分钟，加备份时间；实际价格与账单未核实。输出全部逐步日志、UTC时间轴、完整断点及失败记录。所有新文件在隔离目录；旧证据不覆盖。此补充只能诊断优化范围，不能证明新颖性或论文成立。
'''
 (out/'docs/flashmoba-amp-recovery-lr-boundary-protocol-2026-09-15.md').write_text(doc,encoding='utf-8')
 (R/'docs/amp-lr-boundary-protocol-2026-09-15.md').write_text(doc,encoding='utf-8')
 files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
 (out/'input-manifest.json').write_text(json.dumps(dict(utc=protocol['frozen_utc'],files=files),indent=2)+'\n')
 archive=R/'exports/amp-lr-boundary-input-v0.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  for p in sorted(out.rglob('*')):
   if p.is_file():t.add(p,arcname=p.relative_to(out).as_posix())
 proof=dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size)
 archive.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()

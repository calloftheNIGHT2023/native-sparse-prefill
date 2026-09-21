"""Freeze selected checkpoints and all eligible new windows before evaluation."""
import hashlib,json,shutil,tarfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=R/'exports/amp-confirmation-input-v0';out.mkdir(exist_ok=False)
 for sub in ['scripts','docs','provenance','data/amp-confirmation-v0','data/flashmoba-amp-recovery-v0']:(out/sub).mkdir(parents=True,exist_ok=True)
 cfg=json.loads((R/'data/flashmoba-amp-recovery-v0/config.json').read_text());original=json.loads(json.dumps(cfg))
 source=R/'data/flashmoba-pool-training-v0/train.txt';assert sha(source)==cfg['sources'][0]['sha256']
 tok=AutoTokenizer.from_pretrained(R/cfg['model_path'],local_files_only=True)
 ids=np.asarray(tok(source.read_text(encoding='utf-8'),add_special_tokens=False)['input_ids'],dtype=np.int64)
 assert len(ids)==cfg['sources'][0]['tokens'];width=16385
 excluded=set(cfg['selection']['train']+cfg['selection']['report']);blocked=cfg['selection']['prior_train_intervals']
 chosen=[i for i in range(len(ids)//width) if i not in excluded and not any(i*width<b and (i+1)*width>a for a,b in blocked)]
 assert len(chosen)==29
 held=np.stack([ids[i*width:(i+1)*width] for i in chosen]);np.savez(out/'data/amp-confirmation-v0/heldout.npz',heldout=held)
 shutil.copy2(R/'data/flashmoba-amp-recovery-v0/train-calibration.npz',out/'data/flashmoba-amp-recovery-v0/train-calibration.npz')
 cfg.update(seeds=[2026091561],methods=[16],learning_rates=[.001],prepared_utc=datetime.now(timezone.utc).isoformat(),maximum_controller_seconds=1500,
  selection_rule='No additional model selection. Replicate locked K16 LR1e-3 on second initialization seed; evaluate both fixed dense3e-4 and sparse1e-3 seeds on new windows.',
  scope='Dense-pretrained 0.5B LoRA sparse adaptation confirmation; not native sparse full-parameter pretraining.')
 (out/'data/flashmoba-amp-recovery-v0/config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 scriptnames=['run_amp_recovery.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','eval_amp_recovery_confirmation.py','run_amp_recovery_confirmation.py','package_amp_recovery_results.py']
 for n in scriptnames:shutil.copy2(R/'scripts'/n,out/'scripts'/n)
 pack=out/'scripts/package_amp_recovery_results.py';s=pack.read_text();s=s.replace("for pat in ['scripts/*amp_recovery*.py',", "for pat in ['data/amp-confirmation-v0/*','input-manifest.json','controller.log','scripts/*amp_recovery*.py',")
 pack.write_text(s)
 parents={
  'dense-seed0':('amp-recovery-20260915-v0','results/amp-recovery-stage-v2/cal-k0-lr0.0003','cloud-amp-recovery-final-evidence-v0'),
  'dense-seed1':('amp-recovery-20260915-v0','results/amp-recovery-stage-v2/repeat-k0','cloud-amp-recovery-final-evidence-v0'),
  'k16-seed0':('amp-lr-boundary-20260915-v0','results/amp-recovery-stage-v0/cal-k16-lr0.001','cloud-amp-lr-boundary-evidence-v0')}
 parentrefs={}
 for name,(root,path,local) in parents.items():
  p=R/'results'/local/path;parentrefs[name]=dict(cloud_path='/workspace/native-sparse-prefill/'+root+'/'+path,result_sha256=sha(p/'result.json'),checkpoint_sha256=json.loads((p/'checkpoint-index.json').read_text())['sha256'])
 protocol=dict(frozen_utc=cfg['prepared_utc'],authorization='User explicitly permits continuing for approximately comparable quality with lower cost; prior LR-improvement stopping rule remains recorded but is superseded for this bounded replication.',
  locked_conditions=[[0,2026091560,.0003],[0,2026091561,.0003],[16,2026091560,.001],[16,2026091561,.001]],
  new_scientific_updates=256,new_diagnostic_updates=6,new_report_conditions=4,maximum_controller_seconds=1500,
  heldout_sha256=sha(out/'data/amp-confirmation-v0/heldout.npz'),heldout_indices=chosen,heldout_windows=29,heldout_source_sha256=sha(source),
  excluded_training_indices=original['selection']['train'],excluded_previous_report_indices=original['selection']['report'],excluded_prior_training_intervals=blocked,
  replay_before_new_evaluation='Recompute four calibration NLLs from exact final checkpoint; max absolute difference <=1e-6, otherwise stop before opening heldout arrays.',
  parents=parentrefs,primary='Paired mean NLL gap and measured training time for both locked seeds. Retain the earlier 0.03 nats and <=0.95 time-ratio screen.',
  secondary='Report exact relative perplexity increase and descriptive sensitivity at 3%, 5%, 10% tolerance; these are not retroactive pass declarations.',
  uncertainty='Paired window bootstrap with shared indices across seeds, descriptive only; contiguous corpus windows may be correlated and there are only two initialization seeds.',
  scope='New stage-held-out WikiText train positions, excluding known Qwen training windows and prior report positions. Historical diagnostics on other models or pretrained corpus exposure are not excluded. One corpus, same data order for both seeds, no claim of independent task generalization or native sparse pretraining.')
 (out/'provenance/amp-recovery-confirmation-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 doc='''# 接近质量与训练成本：确认实验协议

用户授权：只要质量接近、训练更省，仍值得继续。上一轮未达到差距缩小0.02的结果保留，本轮明确作为新的有限确认实验，不改写旧结果。

固定比较dense峰值LR3e-4与K16峰值LR1e-3，各256步、两个相同初始化种子；只缺K16第二种子，因此新增一条256步训练及一项6更新预检，复用其他三个已备份断点。模型、训练窗口顺序、优化器、确定性内核均保持不变。不能根据本轮保留文本再换学习率、断点或K。

评测固定使用29个剩余合法16K窗口，排除本项目已知Qwen训练位置、本轮训练位置和上轮12个report窗口。使用全部29个，不挑选其中有利的文本。仍来自WikiText训练语料，不是跨域任务；不保证从未用于更早的其他模型诊断，不排除预训练接触。

每个最终断点先复算原4个校准窗口，最大绝对NLL差必须≤1e-6，核对权重/源码/环境哈希后才读取新保留文本。不改变模型权重，无评测阶段训练。

报告每个种子的配对NLL差、困惑度变化与完整训练时间，保留原0.03 nats且时间比≤0.95的严格筛查。另展示3%、5%、10%容忍度下的取舍，不将新增容忍度冒充旧标准通过。窗口配对bootstrap只作描述性不确定性分析；29个连续语料窗口可能相关，两个种子共享训练顺序。没有下游任务成绩时，不声称任务效果相同。

最多1500秒，不自动重试或扩张。预计训练约5分钟、评测及检查约2–4分钟，另计备份。输出完整UTC时间轴、训练断点、原始逐窗口结果及SHA清单。
'''
 (out/'docs/flashmoba-amp-recovery-confirmation-protocol-2026-09-15.md').write_text(doc,encoding='utf-8');(R/'docs/amp-confirmation-protocol-2026-09-15.md').write_text(doc,encoding='utf-8')
 manifest=dict(frozen_utc=cfg['prepared_utc'],files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()])
 (out/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 archive=R/'exports/amp-confirmation-input-v0.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  for p in sorted(out.rglob('*')):
   if p.is_file():t.add(p,arcname=p.relative_to(out).as_posix())
 proof=dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size);archive.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()

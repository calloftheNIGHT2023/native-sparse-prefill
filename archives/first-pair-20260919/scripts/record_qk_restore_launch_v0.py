"""Record an already launched, authorized diagnostic; does not launch remote work."""
from pathlib import Path
from datetime import datetime,timezone
import json
R=Path(__file__).resolve().parents[1]
t=datetime.now(timezone.utc).isoformat()
p=R/'logs/control-state.json';c=json.loads(p.read_text(encoding='utf-8'))
assert c['cumulative_task_predictions']==32451
launch=json.loads((R/'logs/qk-restore-baseline-launch-v0.json').read_text());assert launch['pid']==59019
c.update(status='qk_restore_baseline_v0_running',updated_utc=t,active_evaluation_jobs=1)
c['qk_restore_baseline']=dict(status='running',controller_pid=59019,protocol='provenance/qk-restore-baseline-protocol-v0.json',collector='scripts/collect_qk_restore_baseline_v0.py',collector_state='logs/qk-restore-baseline-v0-local-collector.json',expected_task_predictions=768,optimizer_updates=0,maximum_gpu_cost_usd_excluding_setup_storage=1200*.74/3600)
p.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(R/'STATE.md').write_text('# 当前状态：已有QK-Restore基线诊断运行中\n\n'+t+'。主线仍是更低训练成本获得接近质量，不要求推理加速。\n\n新背景结果：密集85.94%、稀疏78.12%，差-7.81pp，95%区间[-19.14,3.91]pp，事先5pp非劣门槛未通过。原始模型能力门槛通过。\n\n现运行qk-restore-baseline-v0：四个128步检查点，只将Q/K的48个LoRA B矩阵归零，其他144个适配器张量不变；全部密集评测，旧32背景和9个旧文本窗口仅作诊断。Attention Amnesia已有QK-Restore，本轮不是新方法，只是对照。0训练，最多768任务预测及52次NLL前向；运行预算上限0.247美元，不含准备/空闲/存储。\n\n累计5760科学更新、299诊断更新、32451任务预测。在途不计完成。控制器PID59019，采集器见logs/control-state.json。Pod运行，heartbeat ACTIVE。\n',encoding='utf-8')
with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+t+'：QK-Restore已有基线启动\n\n已查原文https://arxiv.org/abs/2606.11052第5节，恢复Q/K不是原创。本轮四个检查点、最多768任务预测，0更新；协议a6afb3fa3c6d6b56d833de96d6455360dca7d176bf909a4795702f3cbb7b61a6。\n')
print(json.dumps(dict(status='recorded',utc=t,pid=59019)))

"""Record the launched matched V/O experiment and migration correction."""
from pathlib import Path
from datetime import datetime,timezone
import json
R=Path(__file__).resolve().parents[1];t=datetime.now(timezone.utc).isoformat()
p=R/'logs/control-state.json';c=json.loads(p.read_text(encoding='utf-8'))
assert c['cumulative_scientific_updates']==5760 and c['cumulative_task_predictions']==33219
launch=json.loads((R/'logs/vo-training-launch-v0.json').read_text());assert launch['pid']==59884
c.update(status='vo_training_v0_running',updated_utc=t,active_training_jobs=1,active_evaluation_jobs=0)
c['vo_training']=dict(status='running',controller_pid=59884,protocol='provenance/vo-training-protocol-v0.json',collector='scripts/collect_vo_training_v0.py',collector_state='logs/vo-training-v0-local-collector.json',planned_scientific_updates=512,expected_task_predictions=768,maximum_seconds=3900,maximum_gpu_cost_usd_excluding_setup_storage=3900*.74/3600)
p.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(R/'STATE.md').write_text('# 当前状态：从头固定Q/K、训练V/O的匹配实验运行中\n\n'+t+'。主线保持更低训练成本、质量接近，不要求推理加速。\n\n已有QK恢复对照显示稀疏组旧长题平均提升14.06pp、密集组0.78pp；已查明确前作，不作为新方法。现真正训练固定Q/K策略：两个种子，各128步密集或K32，共512计划科学更新。两组均从未训练适配器开始，只训练540672个V/O参数；相同顺序/LR/梯度检查点，全部主评测密集注意力。基础模型已有密集预训练，不是从头原生预训练。\n\n启动前发现云端缺seed1初始检查点；种子0作业已先启动，随后在种子1作业前补传并核验原始SHA，证据logs/vo-training-seed1-initial-migration.json。\n\n预算3900秒（约0.80美元GPU运行费，不含准备/空闲/存储）；逐步日志和0/64/128检查点保留。控制器PID59884，采集器见logs/control-state.json；不要重复启动。\n\n已核验累计5760科学更新、299诊断更新、33219任务预测；在途更新未加入累计。Pod运行，heartbeat ACTIVE。\n',encoding='utf-8')
with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+t+'：固定Q/K训练V/O的匹配试验启动\n\n协议a9d213be45c05e67dcb4307791c10a7e11ea3a5dcecefce050ff3605b1243f51，4作业×128步，计划512科学更新。seed1初始检查点在启动前检查中发现未迁移；PowerShell未自动阻止随后启动的seed0阶段，已在seed1作业前补传并验证SHA，未修改已冻结训练方案。记录此编排失误，后续依赖性命令应分开执行或显式检查退出码。\n')
print(json.dumps(dict(status='recorded',utc=t,pid=59884)))

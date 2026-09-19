import json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.now(timezone.utc).isoformat();r=json.loads((ROOT/'results/frozen-router-capacity-v0/result.json').read_text());audit=json.loads((ROOT/'results/frozen-router-audit-v0/audit.json').read_text())
ctrl_path=ROOT/'logs/control-state.json';ctrl=json.loads(ctrl_path.read_text(encoding='utf-8-sig'))
ctrl.update(updated_utc=now,status='joint_token_router_running',active_training_jobs=1,current_decision_report='docs/frozen-router-results-2026-09-14.md',next_action='Complete preregistered three-arm joint token-indexer run and verify checkpoints; no novelty claim yet.')
ctrl['frozen_router_capacity']=dict(status='complete_and_cpu_verified',runs=4,indexer_optimizer_updates=20480,backbone_updates=0,indexer_input_tokens=41943040,wall_seconds=r['wall_seconds'],fresh_answers_per_run=4096,swap_answers_per_run=1024,all_fresh_and_swap_accuracy=1.0,manifest_files_verified=audit['manifest_files_verified'],cpu_predictions_rechecked=audit['cpu_reevaluation_answers'],archive_sha256='af5f2dc615b9a0e5e2e9ad7a029f0dea59cc3ef2c33cac5c74b8662a3844a00e')
ctrl['joint_token_router']=dict(status='running',planned_runs=3,epochs=40,planned_backbone_updates=37560,planned_indexer_updates=37560,technical_updates=0,code_sha256='bbcbaa9a7828861595d3befa8b7cad764c2eb8d16bfc1c6da229f1fb3ab760ee',local_output='results/joint-token-router-v0',remote_root='/workspace/native-sparse-prefill/schedule-study-v0',console='logs/joint-token-console-v0.log',gpu='A40',scope='Online KL with full dense detached teacher, single seed toy task, not efficient end-to-end sparse training')
save(ctrl_path,ctrl)
(ROOT/'STATE.md').write_text(f'''# 当前状态：静态索引器检查通过，正在跑从头联合训练

更新：{now}。活动训练1，现有RunPod A40，三条件固定40轮联合训练。见docs/joint-token-router-plan-2026-09-14.md。CUDA前向/梯度与梯度隔离检查通过，技术优化更新0；最终结果尚未产生。

冻结主模型后，16/32/64维随机初始化及32维SVD四组索引器，新题4096答案和源值交换1024答案均100%。四组共20480索引器更新、41943040索引器输入tokens，主模型0更新，墙钟161.40秒。55个文件哈希通过，本机CPU从权重复算57344个诊断答案，与云端全部相同。报告docs/frozen-router-results-2026-09-14.md；证据results/frozen-router-capacity-v0和results/frozen-router-audit-v0。

这个结果只支持本任务上静态表示容量足够，不能证明联合训练困难、新方法、真实文本效果或Qwen/QSA全程稀疏可行性。当前在线联合基线仍计算完整主QK教师，因此不能叫全程高效稀疏训练。

五组单层预热候选此前已关闭，结果及启动器异常仍保留；旧均匀集合外抽查、头间normalizer候选仍关闭。所有训练分别计数，完整UTC时间轴TIMELINE.md。

总预算500美元、首阶段30美元；用户允许稍微增加开销，没有增加总额。本批不新租卡。当前费率/发票未核实，结束Python进程不等于停止Pod计费。
''',encoding='utf-8')
with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as f:
    f.write('\n\n- '+now+' 补记本轮：用户允许略增开销并继续，保留总500美元/首阶段30美元；未新租Pod。冻结索引器准备测试首次因测试模块dropout状态失败，修复后5/5 CPU及云端测试通过；此准备失败0科学更新。\n')
    for item in r['runs']:f.write(f"- {item['started_utc']} → {item['finished_utc']} A40/{item['method']}冻结主干拟合完成，40轮/5120索引器更新/0主干更新，新题与源值交换均100%，墙钟{item['fit_wall_seconds']:.2f}秒。\n")
    f.write(f"- {audit['utc']} 55文件SHA核验通过，CPU独立重算57344答案与GPU预测一致，0更新。\n")
    f.write(f'- {now} 联合训练源包bbcbaa9a…760ee已部署，3条件固定40轮；4/4 CPU及云端测试通过，CUDA数值预检通过，技术更新0。开始exact_r16_shadow；实际精确开始时间将据原始事件补记。\n')
(ROOT/'docs/frozen-router-results-2026-09-14.md').write_text('''# 冻结主模型的索引器容量检查

已经学会任务的模型固定后，16维的小索引器也能把需要的内容挑出来。四组都在1024条全新序列的4096个答案及1024条源值对调题上达到100%；逐层单独替换也均100%。这是一个单种子、小合成任务的容量检查，不是新方法贡献。

| 索引器 | 初始化 | 新题 | 源值对调 | 主干更新 |
|---|---|---:|---:|---:|
| 16维 | 随机 | 100% | 100% | 0 |
| 32维 | 随机 | 100% | 100% | 0 |
| 64维 | 随机 | 100% | 100% | 0 |
| 32维 | 已训练主QK的SVD | 100% | 100% | 0 |

每个条件拟合40轮、5120索引器优化步，四组合计20480步。用4096条原训练序列缓存每层输入和主QK因果分布；拟合不使用任务标签。验证集仅监测，新题只在四组拟合结束后评测，与所有旧数据无整行重复。重复遍历输入共41943040 tokens，这是索引器处理量，不能当作主模型预训练tokens或新的独立数据量。

整个云端脚本161.40秒，不含准备与空闲费用。训练前GPU满秩复制等价性通过；主干最终张量SHA与起始相同。55个结果文件SHA核验通过，本机CPU独立重载四组权重，含逐层替换共重算57344个答案，与云端预测完全一致。证据：results/frozen-router-capacity-v0/、results/frozen-router-audit-v0/audit.json；原始终端logs/frozen-router-console.log；每步events.jsonl留存。完整结果归档exports/frozen-router-results-v0.tar.gz，SHA256 af5f2dc615b9a0e5e2e9ad7a029f0dea59cc3ef2c33cac5c74b8662a3844a00e。

这只支持本任务及本主干表示上小索引器有足够容量。100%是有限样本观测，不等于零总体错误，不包含跨随机种子置信保证。SVD用了已经学成的主干信息，也不能冒充从头训练方法。推理gather实现不代表高效内核。

下一步已按原门槛启动三组从头联合训练。需要先知道主干不断变化时在线索引器是否也能跟上；如果都能跟上，则本配置尚未暴露论文所需的问题。MSA/KSA等已有索引器学习，不能把常规KL及stop-gradient申报新贡献。
''',encoding='utf-8')
print(now)

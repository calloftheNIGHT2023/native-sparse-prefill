import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.now(timezone.utc).isoformat();r=json.loads((ROOT/'results/joint-token-router-v0/result.json').read_text());a=json.loads((ROOT/'results/joint-token-analysis-v0/audit-and-summary.json').read_text());d=json.loads((ROOT/'results/joint-token-analysis-v0/route-replacement.json').read_text())
passing=[x['method'] for x in r['runs'][1:] if x['learned_from_scratch_gate']]
if not r['control_valid']:decision='本批精确选择对照未满足最终门槛，不能把学习索引器的退化解释为特有问题。先定位对照，不扩大。'
elif len(passing)==2:decision='16维、64维索引器都从头学会了当前任务；这批没有形成联合训练失败的证据，不能把这种失败作为论文卖点。'
elif passing:decision='只有部分索引器配置通过，发现的是单种子下的配置差异。需独立种子与只换路由诊断后再解释，不够作为论文结论。'
else:decision='同样40轮预算内，精确对照通过，两个学习索引器未过门槛。64维组最后两轮明显上升，因此应称学习延迟/未达标，不能声称它永远学不会；新机制尚未证明。'
text=f'''# 索引器从头联合训练：三组固定实验结果

{decision}

## 实际结果

每组相同主干随机初始权重、相同训练/开发数据、40轮、top8含最近2token。最终新题1024序列共4096答案；源值交换1024答案，只在全部三组拟合完成后评测。精确组同时拟合不参与选路的16维影子索引器。

| 条件 | 新题正确率 | 源值交换正确率 | 首次开发>=99%轮数 | 最终训练KL(两层和) | 训练/开发墙钟 |
|---|---:|---:|---:|---:|---:|
'''
for row in a['rows']:text+=f"| {row['method']} | {row['fresh']:.4%} | {row['swapped']:.4%} | {row['first_development_99_epoch']} | {row['final_kl']:.6g} | {row['wall_seconds']/60:.2f}分钟 |\n"
text+='''
## 保存的权重换路由（0训练步，事后描述性诊断）

| 训练得到的主干 | 两层精确选路 | 两层学习索引器选路 | 只用第0层索引器 | 只用第1层索引器 |
|---|---:|---:|---:|---:|
'''
for row in d['runs']:text+=f"| {row['method']} | {row['exact']['accuracy']:.4%} | {row['learned']['accuracy']:.4%} | {row['only_layer0_learned']['accuracy']:.4%} | {row['only_layer1_learned']['accuracy']:.4%} |\n"
text+='\n再检查实际原路由下的第1层（第二层）：\n\n| 主干 | 全位置平均KL | 答案查询位置平均KL | 教师给源值的平均概率 | 精确top8含源值 | 学习top8含源值 |\n|---|---:|---:|---:|---:|---:|\n'
for row in d['runs']:
    z=row['answer_query_trace_under_original_route'][1]
    text+=f"| {row['method']} | {z['mean_kl_all']:.6g} | {z['mean_kl_answer_queries']:.6g} | {z['mean_teacher_source_value_mass']:.4%} | {z['exact_source_value_recall']:.4%} | {z['learned_source_value_recall']:.4%} |\n"
text+='\n这项事后检查只用已知合成结构定位查询及源值，不参与训练。命中源值不等于利用源值，也不排除通过别的过去查询传递信息；不能据单个注意力指标证明机制。\n'
text+='''
关键读法：成功对照的16维影子索引器接管两层后仍为99.9756%；从头由16维索引器选路的模型，即使最终换回精确选择也只有26.2695%；64维模型换回精确选择为57.2998%。因此不能将本批终点差距归为最后一步选择误差，主干学习轨迹也不同。该干预仍不足以辨认最早因果事件或区分优化延迟与永久失败。

64维组验证准确率在第38/39/40轮从29.375%升至41.525%、57.450%。下一项必要判别应先延长相同配置的训练、保持已保存优化器与随机状态，并使用另外固定的新测试集；这项只辨认延迟，不自动构成贡献。另一项是加入MSA类既有训练配方作强对照。若已有配方足以解决且无额外可证伪机制或实际收益，就不能以此申报新方法。当前三组已经结束，没有悄悄改变40轮门槛、启动额外训练或租更贵卡。
'''
text+='''
换路由也可能改变后层输入分布，不能把表中差异直接称为不可约容量障碍。此表不用于重新调参后复用同一fresh充当确认测试。

## 能说和不能说

此前冻结已学会主干的四种索引器，新题与源值交换都是100%，本轮则把主干和索引器都从随机初始化开始。小索引器的静态容量已通过自身流程检查；是否存在训练问题应以上表为准，不能预设。

所有组还计算完整因果主QK作为KL教师，因此本轮不是全程高效稀疏训练，也不是Qwen第四代架构、真实文本或长上下文验证。主注意力用了选中KV的gather；未报告加速，不把保留连接数当作实际节省的计算。仅一个训练种子、437760主干参数、长度64的合成MQAR，不能支撑LLM泛化结论。

[RTPurbo §8.2/9.1](https://arxiv.org/html/2605.16928v1)已有低维投影拟合，[MSA §3.2/B.4](https://arxiv.org/html/2606.13392v1)已有KL、梯度隔离与预热，[KSA](https://github.com/awni/k_sparse_attention)已有把路由得分加入主注意力来取得任务梯度的实现。因此这些共同组件不是本项目贡献。定向原文、哈希及阅读范围见literature/router-capacity-2026-09-14/和literature/router-capacity-overlap-2026-09-14.md。

## 可复核性与成本

'''
text+=f"三组共37560主干优化步及37560索引器优化步，处理76800000输入tokens/4800000监督答案；是同一批样本重复40轮，不是7680万独立文本tokens。此前冻结索引器的20480步单列，不混为主模型训练。整批脚本墙钟{r['wall_seconds']/60:.2f}分钟，不含上传、环境准备和Pod空闲计费。技术预检0优化更新，分析0优化更新。\n\n"
text+=f"{a['manifest_files_verified']}个结果文件SHA验证通过，3组最终权重在本机CPU重算{a['original_cpu_predictions_checked']}个原始评测答案，与云端全部一致；逐步日志每组连续12520条，损失/梯度有限，UTC单调，初始主干hash一致。16维精确影子组与16维学习组的初始索引器也一致。配对bootstrap以序列为单位，仅反映固定权重下测试样本不确定性，不覆盖训练种子差异。\n\n"
text+='''结果：results/joint-token-router-v0/；核验和换路由：results/joint-token-analysis-v0/；训练曲线：results/joint-token-analysis-v0/curves.png；原始终端logs/joint-token-console-v0.log；逐步events.jsonl与完整优化器/RNG checkpoint均保存。全过程UTC时间轴TIMELINE.md。

本次沿用已有A40，没有换更贵卡或新租Pod。总500美元、首阶段30美元额度不变；实例费率与实际账单未核实，未把训练计时冒充扣费金额。结束Python进程不等于Pod停止计费。
'''
report=ROOT/'docs/joint-token-router-results-2026-09-14.md';report.write_text(text,encoding='utf-8')
ctrl_path=ROOT/'logs/control-state.json';ctrl=json.loads(ctrl_path.read_text(encoding='utf-8-sig'));ctrl.update(updated_utc=now,status='joint_token_router_complete_and_audited',active_training_jobs=0,current_decision_report='docs/joint-token-router-results-2026-09-14.md',confirmed_original_contributions=0,next_action='Use completed result to decide next bounded mechanism test; do not expand model/GPU before meaningful unresolved gap and targeted novelty check.')
ctrl['joint_token_router'].update(status='complete_and_audited',completed_runs=3,backbone_updates=37560,indexer_updates=37560,input_tokens=76800000,supervised_answers=4800000,wall_seconds=r['wall_seconds'],passing_configurations=passing,control_valid=r['control_valid'],cpu_predictions_rechecked=a['original_cpu_predictions_checked'],manifest_files_verified=a['manifest_files_verified'],analysis_optimizer_updates=0)
save(ctrl_path,ctrl)
(ROOT/'STATE.md').write_text(f'''# 当前状态：静态与联合索引器两批已完成并核验

更新：{now}，活动训练0。{decision}

三组联合40轮共37560主干更新/37560索引器更新，76800000输入tokens、4800000监督答案，墙钟{r['wall_seconds']/60:.2f}分钟。完整报告docs/joint-token-router-results-2026-09-14.md，权重和日志results/joint-token-router-v0/，CPU核验及0步换路由results/joint-token-analysis-v0/。CPU重算{a['original_cpu_predictions_checked']}答案与GPU全一致，{a['manifest_files_verified']}文件SHA通过。

此前冻结主干四组均100%，20480索引器更新/0主干更新，CPU57344答案核验全一致。报告docs/frozen-router-results-2026-09-14.md，结果results/frozen-router-capacity-v0/。

两批都是单种子MQAR小模型定位，尚无论文级原创贡献。完整教师仍有二次成本，没有证明全程稀疏训练、真实文本效果或Qwen4适用性。低维索引器/KL/梯度隔离已见前作，文献边界literature/router-capacity-overlap-2026-09-14.md。旧单层预热、均匀集合外抽查、头间normalizer候选保持关闭。

总500美元、首阶段30美元额度不变，本轮未新租卡。Pod实际费率与账单未知，训练结束不等于停止计费；结果已经备份本机。时间轴TIMELINE.md。
''',encoding='utf-8')
with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as f:
    for run in r['runs']:f.write(f"\n- {run['started_utc']} → {run['finished_utc']} A40/{run['method']}联合训练40轮，12520主干及12520索引器更新，25600000输入tokens；终点评测新题{run['fresh']['accuracy']:.4%}，源值交换{run['swapped']['accuracy']:.4%}（评测在全部拟合后进行）。")
    f.write(f"\n- {now} 三组权重归档并本机重算{a['original_cpu_predictions_checked']}答案与GPU一致，{a['manifest_files_verified']}文件SHA通过，0步换路由完成。{decision} 科学失败0/中断0，活动训练0；Pod未自动停止。\n")
handoff=ROOT/'provenance/router-diagnostics-handoff-2026-09-14';handoff.mkdir(exist_ok=False)
items=['STATE.md','TIMELINE.md','logs/control-state.json','docs/frozen-router-capacity-plan-2026-09-14.md','docs/frozen-router-results-2026-09-14.md','docs/joint-token-router-plan-2026-09-14.md','docs/joint-token-router-results-2026-09-14.md','results/frozen-router-audit-v0/audit.json','results/joint-token-analysis-v0/audit-and-summary.json','literature/router-capacity-overlap-2026-09-14.md']
for item in items:
    dest=handoff/item;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/item,dest)
save(handoff/'manifest.json',[dict(path=p.relative_to(handoff).as_posix(),sha256=sha(p)) for p in sorted(handoff.rglob('*')) if p.is_file()])
save(handoff/'archive-index.json',[dict(path='exports/'+name,sha256=sha(ROOT/'exports'/name)) for name in ['frozen-router-code-v0.tar.gz','frozen-router-results-v0.tar.gz','joint-token-code-v0.tar.gz','joint-token-results-v0.tar.gz']])
print(json.dumps(dict(utc=now,decision=decision,passing=passing,report=str(report)),ensure_ascii=False))

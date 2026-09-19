"""Persist this diagnostic stage without changing historical evidence."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def save(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    audit=json.loads((ROOT/'logs/far-recall-final-audit.json').read_text())
    now=datetime.now(timezone.utc).isoformat()
    backup=ROOT/'provenance/far-recall-handoff-2026-09-14'
    backup.mkdir(exist_ok=False)
    for name in ['STATE.md','README.md','logs/control-state.json']:
        dest=backup/'prior-state'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
    state='''# 当前状态：远距离测试已建好，完整模型尚未学会按键查找

更新：{now}。项目：D:/ChatGPT/projects/native-sparse-prefill。

**本轮本机CPU作业全部结束，没有启动新GPU训练，暂不扩大。** 旧RunPod是否停止本轮未核实；历史云模型和日志已完整备份，若旧Pod还运行，需在控制台Stop。实际账单未知，首阶段30美元、总预算500美元不变。

## 最新结果与决定

- 2048长度关联回忆数据v1：5632条、352个反事实家族。程序查表正确率100%，不等于神经网络成绩。这类任务已有MAD/RULER等前作，不是新贡献。
- 静态局部掩码回溯6层后，查询只依赖0—3和536—2047位置；答案记录在64—447，没有间接路径。16类成组题上的局部答案类别内预测正确率必为6.25%。该界不适用于动态选块或Qwen的Gated DeltaNet路径。
- 已有70M模型的CPU输入干预通过：局部远端输出差0，完整/接通正确块后输出会变。仅验证信息能否影响输出，没有证明70M答对。
- 两轮完整的330752参数、128长度、2层CPU训练各4000步；测试分别35.16%和38.09%，未达预设开发/测试均80%的门槛。不是70M/2K成绩，也不是独立种子复现。
- 最新模型的局部远端/删证据测试均12.5%，移近证据并切局部掩码为100%；位置和掩码同时变化，是分布外干预，不能作为架构比较。
- 开发集只换查询编号：答案应变的查询对中，83.86%保持相同预测。原开发准确率39.45%，忽略编号选最常见数值为39.84%。优先排查键值绑定，不能把当前完整模型与局部对照的差距包装为新方法收益。
- 流式v1第3143步生成时查到随机种子碰撞，完成3142更新后失败，无最终测试。改为PCG64全SHA种子，v2全部64000个训练家族无重复且与留出家族不重叠。29项正确性检查通过。

## 下一步接续

先核对已有关联回忆开源实现的键值绑定结构和监督密度，设计一次有明确区别的本地对照。当前不盲目加步数、不追加种子或新租卡。固定新家族开发和测试均80%门槛不下调；如改变任务/监督须另存版本，不能称单因素归因。

只有完整模型先学会新题，才能进入70M/2K验证，再讨论从第一步稀疏和预热后稀疏。现有新任务运行器仅CPU；云端稠密语言模型运行器就绪不代表新任务CUDA运行器已就绪。尚无确认原创贡献或Qwen复现结果。

## 日志和计数

- 最新报告：docs/far-recall-results-2026-09-14.md；计划与理论前提：docs/far-recall-validation-plan-2026-09-14.md。
- 审计：logs/far-recall-final-audit.json；汇总：results/far-recall-summary.json；图：results/far-recall-figures/。
- 新小模型诊断：2次完成、1次失败，共11142更新/22818816输入tokens/178272受监督答案，含失败前已执行步骤。逐步UTC、损失、梯度、配置、源码、模型与优化器均保留。
- 历史70M主干仍为6次CPU+2次GPU，共6930更新/11440128预测tokens；本轮小模型计数独立保存。
- 历史云数据为重复的小语料，1M和10M从同一随机种子重启，不能算种子重复。10M上下文差门槛未通过：dev0.023199/test0.012304，要求二者均至少0.02。
- 历史报告：docs/cloud-first-results-2026-09-14.md、docs/joint-pilot-results-2026-09-14.md。冻结索引器主实验204次/221994更新等旧记录不改。
- TIMELINE.md保留准备、成功、失败和修复时间轴；logs/control-state.json为接续状态。

均匀集合外抽查和头间normalizer两条候选继续关闭。QSA/MSA的原生训练、预热、辅助监督均有前作；任务有效不代表新方法已成立。
'''.format(now=now)
    (ROOT/'STATE.md').write_text(state,encoding='utf-8')
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    paragraphs=readme.split('\n\n')
    paragraphs[3]='均匀抽查和头间归一化校正已关闭。最新完成远距离关联回忆测试、本机小模型训练和70M模型的CPU输入干预。测试正确率38.09%，未达到完整模型可学性门槛，暂不扩大。29项CPU正确性检查通过；本轮未启动GPU，旧Pod停止状态未核实。当前仍无确认原创贡献，具体见 STATE.md。'
    readme='\n\n'.join(paragraphs)
    readme=readme.replace('- [当前状态](STATE.md)：已完成事项、下一步和实验启动条件。','- [当前状态](STATE.md)：已完成事项、下一步和实验启动条件。\n- [最新远距离测试报告](docs/far-recall-results-2026-09-14.md)：两次完整训练、一次失败、查询干预及全部时间轴。\n- [远距离测试设计与前作](docs/far-recall-validation-plan-2026-09-14.md)：计算图前提、已知关联回忆任务与扩大门槛。')
    readme=readme.replace('| `tests/` | 23项CPU检查','| `tests/` | 29项CPU检查')
    readme=readme.replace('两个窄方法候选已经关闭。现已完成2K云端稠密控制；后续先审查任务对远端信息是否敏感，再决定稀疏比较。已有云运行不代表方法新颖性已确认，当前没有后台训练作业。','两个窄方法候选已经关闭。2K云端稠密控制完成后，已建立远端信息隔离测试；小模型尚未可靠学会按键查找，先在本机排查监督与任务设置。已有云运行不代表方法新颖性已确认，本轮训练已结束。')
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    p=ROOT/'logs/control-state.json'; control=json.loads(p.read_text())
    control.update(updated_utc=now,status='far_recall_controls_complete_dense_learnability_failed',
        budget_status='500_usd_total_ceiling_30_usd_first_cloud_stage_actual_invoice_unknown',
        cloud_billing_status='not_rechecked_this_turn_manual_console_stop_if_running',
        current_decision_report='docs/far-recall-results-2026-09-14.md',audit_report='logs/far-recall-final-audit.json',
        result_scope='Historical 8 scientific 70M runs; additional 330k CPU recall diagnostics recorded separately.',
        language_model_training_runs_scope='Historical 70M real-text training only; excludes toy recall and frozen indexer diagnostics.',
        correcteness_tests_note='29 CPU tests; 4 CUDA tests belong to prior cloud stage',
        correctness_tests_passed=29,active_training_jobs=0,paid_experiment_gate_passed=False,
        science_method_expansion_gate_passed=False,far_recall_data='data/far-recall-v1',far_recall_2k_examples=5632,
        far_recall_toy_diagnostics=audit['totals'],far_recall_learnability_gate_passed=False,
        far_recall_toy_test_accuracy=0.380859375,far_recall_cuda_runner_ready=False,
        current_turn_gpu_jobs_started=0,current_turn_cloud_invoice_delta_usd=None,
        next_action='Audit established associative-recall implementation and supervision; bounded local key-value binding control before any GPU scaling. Keep 80% final dev/test gate. Old Pod stop not checked.')
    save(p,control)
    entry=['','## '+now+' 远距离测试与本机可学性阶段','',
        '本轮未启动新的GPU作业，旧Pod停止/计费状态未重新查询。以下均为CPU；与历史70M训练单独计数。','']
    events=[]
    for version in ['far-recall-v0','far-recall-v1']:
        d=json.loads((ROOT/'data'/version/'audit.json').read_text())
        events.append((d['started_utc'],d['finished_utc'],version+'：5632条2K符号题及反事实/路径审计通过；程序查表非神经准确率。'))
    summary=json.loads((ROOT/'results/far-recall-summary.json').read_text())
    for r in [*summary['runs'],summary['failed_run']]:
        detail=f"{r['name']}：{r['updates']}更新，纯训练{r['training_seconds']:.2f}秒。"
        detail+=f"测试{r['dense_test']['accuracy']:.2%}，80%门槛未通过。" if r['status']=='complete' else '生成器重复家族断言失败，保留checkpoint，无最终测试。'
        events.append((r['started_utc'],r['finished_utc'],detail))
    for key,text in [('probe','70M只读CPU前向10例：静态局部远端输出不变，接通远端路径后响应；非答题准确率。'),('query_probe','最新小模型只读CPU开发集查询干预1024例：答案不同的查询对中仅16.14%改变预测。')]:
        r=summary[key]; events.append((r['started_utc'],r['finished_utc'],text))
    for start,end,detail in sorted(events): entry.append(f'- {start} 至 {end}：{detail}')
    entry.extend(['', '补记：70M敏感性v0在batch=2时因浮点舍入导致严格相等检查失败；事件/日志保留，v1采用同形状batch=1顺序前向通过。流式v1低32位种子碰撞已复现，PCG64回归检查及新64000题族去重通过。',
        '', '29项CPU正确性测试通过；清单SHA、更新顺序、时间顺序、有限损失/梯度和流式家族唯一性审计通过。新小模型累计11142更新、22818816输入tokens、178272受监督答案。当前不扩大，先排查键值绑定。完整报告：docs/far-recall-results-2026-09-14.md。'])
    with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n'.join(entry)+'\n')
    for name in ['STATE.md','README.md','TIMELINE.md','logs/control-state.json','logs/far-recall-final-audit.json','docs/far-recall-results-2026-09-14.md','docs/far-recall-validation-plan-2026-09-14.md','scripts/report-far-recall.py','scripts/probe-recall-query-use.py','scripts/save-recall-handoff.py']:
        dest=backup/'final-state'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    save(backup/'manifest.json',[dict(path=str(p.relative_to(backup)),sha256=sha(p)) for p in sorted(backup.rglob('*')) if p.is_file()])
    print(json.dumps(dict(utc=now,totals=audit['totals'],handoff=str(backup),active_training_jobs=0)))

if __name__=='__main__':main()

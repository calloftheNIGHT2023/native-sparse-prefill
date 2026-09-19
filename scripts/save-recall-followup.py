"""Write bounded follow-up findings and handoff, preserving previous state."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result=json.loads((ROOT/'results/recall-followup-v0/result.json').read_text(encoding='utf-8'));runs=result['runs']
    labels=['普通注意力，只教最后一题','普通注意力，四题都教','第一层固定读前一个token','普通注意力，学习率1e-3']
    now=datetime.now(timezone.utc).isoformat();archive=ROOT/'provenance/recall-followup-handoff-2026-09-14';archive.mkdir(exist_ok=False)
    for name in ['STATE.md','README.md','logs/control-state.json']:
        d=archive/'prior-state'/name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,d)
    lines=['# 本机四组定位实验：尚未通过基础可学性门槛','',
        '本轮完成4次CPU训练和额外新题检查，没有启动GPU作业。没有任何一组达到预先固定的开发/测试均80%正确率。当前决定是停止在这套自制题上继续扫参数，先建立作者公开最小例子的可复现基线，再逐项加回远距离约束。','',
        '## 用人话说','',
        '模型能看见前面的几个数值，但还不能可靠地按问题中的编号选对那个数值。增加问题监督、强制传递相邻信息、核对公开默认学习率，都没解决当前设置的问题。这说明本轮几种调整不足以建立可靠对照，不证明任务无解、算力无用或原生稀疏训练不可行。','',
        '## 结果','',
        '| 设置 | 最终开发 | 原测试 | 另外128个新题族 | 新题族95%区间 | 门槛 |','|---|---:|---:|---:|---:|---|']
    for r,label in zip(runs,labels):
        c=r['fresh_ci'];lines.append(f"| {label} | {r['development'][-1]['accuracy']:.2%} | {r['dense_test']['accuracy']:.2%} | {c['accuracy']:.2%} | {c['low']:.2%}—{c['high']:.2%} | 未通过 |")
    lines.extend(['','新题族种子2026091420，在所有训练完成后生成，未进入训练和原先开发/测试集合；128族×8变体共1024条。置信区间按题族重采样，不能把一族中的8个变体当作独立样本；它不反映训练种子之间的不确定性。原留出集已参与探索，新增对照不称预注册独立确认。','',
        '## 对照是否公平','',
        '- 全部330752名义参数、128长度、2层、模型seed41，逐次从相同初始化重新开始。每组4000更新、每批16条、8192000输入tokens；所有题族SHA和答案变体逐步完全一致。',
        '- 前两组输入完全相同，只改变监督位置。四题组受监督答案256000个，其余各64000个，损失均取平均；不是等标签量、等训练计算。前三个额外查询按非目标记录位置排序，最终查询仍指向目标；这是本项目控制格式，不是原版MQAR。',
        '- 前驱组第一层只读取前一位置，第二层仍全局；不读取答案元数据。第一层单一选择使Q/K无须学习路由，名义参数相同也不表示有效自由度相同。它不是普通完整注意力基线或新算法。',
        '- 学习率组恢复普通全局注意力，仅把峰值3e-4改为公开默认1e-3，保留当前warmup与余弦比例。学习率与前驱组是在此前诊断后追加，结论仅限当前任务和训练配置。','',
        '## 有没有学会按编号改变答案','',
        '| 设置 | 换编号且正确答案不同时，预测改变比例 | 两个查询都答对 | 保留四题输入时最后一题开发准确率 |','|---|---:|---:|---:|'] )
    for r,label in zip(runs,labels):
        q=r['query_probe'];lines.append(f"| {label} | {q['prediction_changed_fraction']:.2%} | {q['both_correct_fraction']:.2%} | {q['multiquery_input_last_answer_accuracy']:.2%} |")
    lines.extend(['','干预只在开发集进行，记录不变、仅替换查询键。这些成组查询对互相关联，不能当成2672次独立实验。模型在训练格式下的最后一题也不足，因此单题格式变化不能解释全部失败。没有据此声称已确定唯一根因。','',
        '## 公开实现核对','',
        '[Zoology生成器](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/data/multiquery_ar.py)使用多个查询位置监督；[默认配置](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/config.py)学习率1e-3。位置嵌入、记录布局、词表、值是否重复等仍与本项目不同，所以本轮不算论文复现。',
        '', '[MAD生成器](https://github.com/athms/mad-lab/blob/0f49a452b84ca0d13f8eb9c1ffa649032376fb1b/mad/data/instances.py)的训练分支监督完整下一token序列，评测另有检索标签；也不同于我们的最终答案监督。[Induction-head前作](https://www.transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)已有前驱信息传递和后续匹配的机制分析，前驱对照不构成原创。以上均为定向实现/机制核对，非全文精读或穷尽查重。','',
        '## 时间轴','', '| 运行 | 开始UTC | 结束UTC | 更新数 | 纯训练秒 |','|---|---|---|---:|---:|'])
    for r in runs:lines.append(f"| {r['name']} | {r['started_utc']} | {r['finished_utc']} | {r['updates']} | {r['training_seconds']:.2f} |")
    lines.extend(['',f"本轮总计16000更新、{result['input_tokens']:,}输入tokens、{result['supervised_answers']:,}个受监督答案。先前2次完成/1次中止的小模型诊断仍完整保留；累计6次完成/1次中止、27142更新、55586816输入tokens、626272受监督答案。历史8次70M训练另计，未混入本轮。",'',
        '32项正确性测试通过。四次训练的文件SHA、初始化、逐步输入、损失/梯度有限性和UTC顺序检查通过；新题族与训练及旧留出家族SHA均无重叠。全部最终权重、优化器、随机状态、源码快照和逐步日志留存。','',
        '- 汇总与逐样本结果：`results/recall-followup-v0/`；审计：`logs/recall-followup-audit.json`。',
        '- 四次训练目录与日志均按上表run ID保存；监督密度配对审计另见`results/recall-supervision-pair-v0/`。',
        '- 冻结计划：`docs/recall-supervision-plan-2026-09-14.md`、`docs/recall-binding-plan-2026-09-14.md`、`docs/recall-learning-rate-plan-2026-09-14.md`。',
        '- 实现证据：`docs/recall-implementation-audit-2026-09-14.md`及`literature/recall-implementation-audit-2026-09-14/`。','',
        '## 接下来怎么改','',
        '本轮停止训练。下一阶段先按Zoology公开的64长度、4键值对最小例子核对数据、模型和监督，保留全部配置差异，建立已知任务的学习曲线；不继续给当前失败任务零散加参数。若已知例子能跑通，再逐项恢复噪声、距离和反事实成组控制，观察哪一步破坏可学性。这个已知基线尚未执行，不能写成已复现。',
        '', '只有基础基线可复现、远端任务本身可学且独立新题检查通过，才进入70M/2K和稀疏方法比较。本轮不支持论文贡献、全程稀疏收益或Qwen混合架构结论。未启动GPU不等于旧Pod停止计费；本轮未查旧Pod或实际账单。'])
    report=ROOT/'docs/recall-followup-results-2026-09-14.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state=f'''# 当前状态：四组本地定位完成，暂停自制任务上的调参

更新：{now}；项目D:/ChatGPT/projects/native-sparse-prefill。本轮全部CPU训练及只读评测已结束，新增GPU作业0，旧Pod停止和实际账单未核实。首阶段30美元/总500美元不变。

## 最新结论

四组330752参数、128长度、2层、seed41的4000步训练均未达到固定80%开发/测试门槛。新题族准确率：只教最后一题40.23%，四题都教35.84%，第一层前驱传递41.60%，普通注意力学习率1e-3为34.38%。全部输入题族与初始化配对核验通过。只换查询编号时仍大多保持同一预测，尚未可靠学会键值查找。

多查询监督、相邻传递及学习率默认值的核对不足以解决当前可学性。后两项是自适应追加诊断；只覆盖一个种子和一种小模型，不能据此否定任务可解性或稀疏训练方向。前驱传递是已知机制且改变结构，不是新贡献。

## 下一步

停止在当前自制任务上零散加参数。先完整核对并跑通Zoology的64长度/4键值对公开最小例子，记录数据、模型、监督和CPU实现差异；之后逐项加回噪声、距离、反事实家族。该公开例子尚未训练，不得声称已复现。当前没有付费扩大依据，不自动追加租卡。

## 结果与日志

- 最新报告：docs/recall-followup-results-2026-09-14.md。
- 本轮汇总/图/新题：results/recall-followup-v0/；审计：logs/recall-followup-audit.json；32项CPU测试通过。
- 逐步日志、权重、配置、源码、优化器见4个recall-*运行目录；开始结束见TIMELINE.md。
- 本轮4次完成/16000更新/32768000输入tokens/448000监督答案。
- 小模型诊断累计6次完成、1次中止，27142更新/55586816输入tokens/626272监督答案；中止运行是此前流式种子碰撞，失败证据保留。
- 历史70M训练仍8次（6CPU+2GPU），6930更新/11440128预测tokens，另计；模型和日志已备份。
- 旧2K路径审计与70M敏感性仍有效，但没有证明神经模型会答题：docs/far-recall-results-2026-09-14.md。静态局部支持界不适用于动态选块或Qwen Gated DeltaNet。

均匀集合外抽查和头间normalizer两条候选继续关闭。没有确认原创贡献；关联回忆和induction机制已有前作，不换名包装。当前新任务运行器仅CPU，未执行Qwen或70M/2K新任务训练。
'''
    (ROOT/'STATE.md').write_text(state,encoding='utf-8')
    p=ROOT/'README.md';parts=p.read_text(encoding='utf-8').split('\n\n')
    parts[3]='最新完成4组本机定位实验，增加监督、相邻传递和公开默认学习率均未让小模型可靠学会查找；新题准确率34.4%—41.6%，未过80%门槛。32项CPU检查通过。暂停当前自制任务上的调参，下一步跑通公开最小基线后逐项加回远距离约束；本轮未启动GPU。详见STATE.md。'
    text='\n\n'.join(parts).replace('- [当前状态](STATE.md)：已完成事项、下一步和实验启动条件。','- [当前状态](STATE.md)：已完成事项、下一步和实验启动条件。\n- [最新四组定位报告](docs/recall-followup-results-2026-09-14.md)：配对训练、新题复核、查询干预和时间轴。')
    text=text.replace('| `tests/` | 29项CPU检查','| `tests/` | 32项CPU检查');p.write_text(text,encoding='utf-8')
    p=ROOT/'logs/control-state.json';c=json.loads(p.read_text(encoding='utf-8'))
    c.update(updated_utc=now,status='recall_followup_complete_all_learnability_gates_failed',active_training_jobs=0,
        correctness_tests_passed=32,current_decision_report='docs/recall-followup-results-2026-09-14.md',audit_report='logs/recall-followup-audit.json',
        paid_experiment_gate_passed=False,far_recall_learnability_gate_passed=False,
        far_recall_toy_diagnostics=dict(completed_runs=6,failed_runs=1,optimizer_updates=27142,input_tokens=55586816,supervised_answers=626272),
        recall_followup_runs=4,recall_followup_updates=16000,recall_followup_fresh_accuracies=[r['fresh_evaluation']['accuracy'] for r in runs],
        current_turn_gpu_jobs_started=0,next_action='Stop adaptive tuning of bespoke recall; establish pinned public Zoology minimal example, then add distance/noise/counterfactual constraints separately. No GPU expansion yet.')
    save(p,c)
    with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as f:
        f.write('\n## '+now+' 四组本机定位完成\n\n')
        for r in runs:f.write(f"- {r['started_utc']} 至 {r['finished_utc']}：{r['name']}完成4000更新，测试{r['dense_test']['accuracy']:.2%}，固定门槛未通过，纯训练{r['training_seconds']:.2f}秒。\n")
        f.write(f"- {result['started_utc']} 至 {result['finished_utc']}：校验四组SHA/配对输入和初始化，另128个新题族只读复核完成，无参数更新。32项CPU正确性检查通过。\n\n暂停当前自制任务调参，下一步先建立公开最小基线；未启动GPU，旧Pod状态未知。报告docs/recall-followup-results-2026-09-14.md。\n")
    for name in ['STATE.md','README.md','TIMELINE.md','logs/control-state.json','logs/recall-followup-audit.json','docs/recall-followup-results-2026-09-14.md','docs/recall-implementation-audit-2026-09-14.md','docs/recall-supervision-plan-2026-09-14.md','docs/recall-binding-plan-2026-09-14.md','docs/recall-learning-rate-plan-2026-09-14.md','scripts/save-recall-followup.py']:
        d=archive/'final-state'/name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,d)
    save(archive/'manifest.json',[dict(path=str(p.relative_to(archive)),sha256=sha(p)) for p in sorted(archive.rglob('*')) if p.is_file()])
    print(json.dumps(dict(report=str(report),active_training_jobs=0,utc=now)))

if __name__=='__main__':main()

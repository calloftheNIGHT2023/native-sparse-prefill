"""Save the completed CPU baseline and read-only checks without altering run evidence."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    run = read("results/zoology-basic-cpu-v1/result.json")
    query = read("results/zoology-query-intervention-v0/result.json")
    audit = read("logs/zoology-baseline-final-audit.json")
    distance = read("logs/zoology-fresh-distance-audit.json")
    assert run["status"] == "complete"
    assert run["upstream_stop_threshold_reached"] and run["independent_fresh_at_least_99"]
    assert query["optimizer_updates"] == 0
    verified = {}
    for folder in ["results/zoology-basic-cpu-v1", "results/zoology-query-intervention-v0"]:
        manifest = read(folder + "/manifest.json")
        for item in manifest:
            assert sha(ROOT / folder / item["path"]) == item["sha256"], item["path"]
        verified[folder] = len(manifest)
    assert sha(ROOT / "results/zoology-basic-cpu-v1/checkpoint.pt") == query["checkpoint_sha256"]
    assert sum(x["count"] for x in distance["bins"].values()) == 4000
    assert sum(x["correct"] for x in distance["bins"].values()) == 3964
    now = datetime.now(timezone.utc).isoformat()
    handoff = ROOT / "provenance/zoology-baseline-handoff-2026-09-14"
    handoff.mkdir(exist_ok=False)
    previous = handoff / "before"
    previous.mkdir()
    changed = ["STATE.md", "README.md", "TIMELINE.md", "logs/control-state.json", "docs/zoology-baseline-results-2026-09-14.md"]
    for path in changed:
        target = previous / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)

    report_path = ROOT / "docs/zoology-baseline-results-2026-09-14.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace("最终验证正确率99.12%", "最终验证正确率99.125%")
    rows = [f"| {gap} | {b['count']} | {b['correct']} | {b['accuracy']:.2%} |" for gap, b in distance["bins"].items()]
    report += """
## 查询干预与距离复核（训练后补充）

在同一批独立新序列上，每条选最后一个查询，只把查询键换成前方记录中另一个键，其他输入和权重不变；两种查询的正确答案必定不同。1000对输入中，原最后一题正确率99.2%，改查询后98.6%，99.5%的预测随查询改变，两题均正确的比例97.8%。这支持模型学会了随键查找，而非总是输出相同答案。

这项检查使用同一个训练种子和检查点，且换键可能造成查询键重复，属于探索性输入干预，不是新的独立训练重复。原始输入变更、预测和标签保存在results/zoology-query-intervention-v0/，未进行优化更新，检查点SHA保持一致。

距离定义为查询位置减去相应源值位置。直接按已保存的新题预测分组，没有新增前向或训练：

| 距离（token位置差） | 答案数 | 答对数 | 正确率 |
|---|---:|---:|---:|
""" + "\n".join(rows) + """

各距离组均约99%，但最长上下文只有64，不能推广到2K长上下文或全程稀疏训练。距离核验见logs/zoology-fresh-distance-audit.json。

学习曲线在第21轮为32.55%，第22轮为96.375%，第23轮为99.125%。这是单次运行的观察；不能据此证明新的学习相变，也不能把旧运行失败只归因于训练时间不足。

下一轮先在同一作者配置上将上下文从64增加到128，重新训练而非直接外推位置嵌入；数据数量、4组键值、优化规则等保持一致。跑通后再逐项增加距离约束，最后才开展稠密与稀疏的公平比较。本次收尾没有启动这些后续训练。
"""
    report_path.write_text(report, encoding="utf-8")

    state = f"""# 当前状态：公开最小基线跑通，可以继续检查远距离条件

更新：{now}；项目D:/ChatGPT/projects/native-sparse-prefill。本轮CPU训练和只读评测已结束，活动训练0，新增GPU作业0。首阶段30美元/总500美元不变，旧Pod停止与实际账单仍未核实。

## 最新结论

固定版本Zoology公开basic例子在CPU完成23轮、7199次更新，验证正确率99.125%，独立1000条新序列/4000个答案正确率99.1%（序列bootstrap 95%区间98.8%—99.375%）。模型437760参数、64长度、2层、1头、seed123；训练与评测约345秒。原训练器保留，只修改CPU默认设备和Windows种子整数类型。

只改变最后一题查询键的1000对输入中，改后正确率98.6%，99.5%的预测随查询改变。距离分组正确率均约99%，最长上下文仍只有64。单训练种子、公开已知任务，不是新方法贡献或长上下文结果，也不证明全程稀疏训练有效。

## 下一步

已具备一个可靠的稠密基线。保持作者模型及优化流程，先单独把上下文64增加到128，再逐项加入远距离约束；每轮保留独立新题与查询干预。之后再安排稠密/稀疏比较。下一轮未启动，没有新的付费扩大依据；不自动追加租卡。

## 结果与日志

- 最新报告：docs/zoology-baseline-results-2026-09-14.md；固定方案：docs/zoology-baseline-plan-2026-09-14.md。
- 原始训练/数据/权重/代码：results/zoology-basic-cpu-v1/；逐步events.jsonl和logs/zoology-basic-cpu-v1.log；时间轴TIMELINE.md。
- 基线49个清单文件和查询干预3个清单文件SHA核验通过；36项CPU正确性检查通过。日志包装和原训练器单步权重/随机状态逐位一致。
- 本轮公开基线1次完成、7199更新、14720000输入tokens、920000监督答案；固定10000序列重复23轮。另1次准备失败0更新；另2次技术单步更新，不计科研训练。
- 历史自制小模型诊断仍6次完成、1次中止，27142更新/55586816输入tokens/626272监督答案。最新四组未过门槛见docs/recall-followup-results-2026-09-14.md。不能把与本轮差异归因于单一配置因素。
- 历史70M训练仍8次（6CPU+2GPU），6930更新/11440128预测tokens，另计；模型和日志已备份。
- 旧2K路径审计与70M输入敏感性不等于答题准确率，静态局部支持界不适用于动态选块或Qwen Gated DeltaNet。

均匀集合外抽查和头间normalizer两条候选继续关闭；没有确认原创贡献。已知关联回忆和induction机制不换名包装。当前未执行Qwen或70M/2K新任务训练。
"""
    (ROOT / "STATE.md").write_text(state, encoding="utf-8")
    control = read("logs/control-state.json")
    control.update(updated_utc=now, status="pinned_zoology_minimal_baseline_complete_cpu",
        active_training_jobs=0, current_turn_gpu_jobs_started=0,
        next_action="Keep pinned Zoology baseline; next change context length 64 to 128 alone, then distance constraints. No further training started in this turn.",
        current_decision_report="docs/zoology-baseline-results-2026-09-14.md",
        audit_report="logs/zoology-baseline-final-audit.json",
        correctness_tests_passed=36,
        result_scope="Historical 70M runs, handcrafted 330752-parameter diagnostics, and public 437760-parameter Zoology baseline counted separately.",
        correcteness_tests_note="36 CPU tests passed; 4 CUDA tests belong to prior cloud stage",
        zoology_minimal_baseline_passed=True,
        zoology_baseline=dict(completed_runs=1,setup_failures=1,failed_training_runs=0,
            optimizer_updates=run["updates"],input_tokens=run["input_tokens"],
            supervised_answers=run["supervised_answers"],epochs=run["epochs"],
            validation_accuracy=0.99125,fresh_accuracy=run["fresh_evaluation"]["accuracy"],
            query_intervention_accuracy=query["changed_query_accuracy"],
            query_prediction_changed_fraction=query["prediction_changed_fraction"],
            context_length=64,parameters=437760,training_seeds=[123],
            source_commit="1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb",
            technical_instrumentation_updates=2))
    save(ROOT / "logs/control-state.json", control)
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    old = next(x for x in readme.splitlines() if x.startswith("最新完成4组"))
    readme = readme.replace(old, "最新已在CPU跑通Zoology公开最小基线：23轮、7199次更新，独立新题正确率99.1%，换查询后98.6%。36项CPU检查通过。下一步保持作者配置逐项增加上下文和距离约束；本轮训练已结束、未启动GPU。这是已知基线验证，尚无新的稀疏方法贡献。详见STATE.md。")
    readme = readme.replace("研究问题：QSA 提前用于训练时", "原始研究问题（下述具体候选现已关闭，保留作历史记录）：QSA 提前用于训练时")
    readme = readme.replace("- [最新四组定位报告]", "- [公开最小基线结果](docs/zoology-baseline-results-2026-09-14.md)：训练、新题、查询干预、距离分组与证据边界。\n- [公开基线固定方案](docs/zoology-baseline-plan-2026-09-14.md)：作者配置与平台适配。\n- [历史四组定位报告]")
    readme = readme.replace("32项CPU检查", "36项CPU检查")
    readme = readme.replace("小模型尚未可靠学会按键查找，先在本机排查监督与任务设置。", "公开64长度基线已可靠学会按键查找，下一步逐项加回远距离约束。")
    readme += "\n公开Zoology基线依赖锁定于`requirements-zoology-cpu-lock.txt`，上游源文件和两处平台补丁在`third_party/zoology-1ad20d1/`；本地日志关闭W&B联网。最新完整结果和接续状态见STATE.md。\n"
    readme_path.write_text(readme, encoding="utf-8")
    timeline = f"""
## {now} 公开Zoology基线完成与收尾

- {run['started_utc']} 至 {run['finished_utc']}：zoology-basic-cpu-v1在CPU完成23轮、7199次更新；纯训练325.57秒、训练评测墙钟约345秒，14720000输入tokens、920000监督答案。验证99.125%，触发原作者超过99%的提前停止规则；独立1000条新序列/4000个答案99.1%。
- {audit['utc']}：49个基线清单文件SHA、连续更新、UTC顺序、有限损失/梯度、轮末余弦学习率及首次停止阈值全部核验通过；36项CPU检查通过。原训练器/日志包装同批次权重与随机状态逐位一致，2次技术更新另计。
- {query['started_utc']} 至 {query['finished_utc']}：1000对查询键干预完成，改键后98.6%正确、99.5%预测改变，0更新，检查点SHA不变。保存预测的距离分组均约99%，最大上下文仍64。
- {now}：保存报告、状态与交接快照；公开基线1次完成，准备失败v0为0更新。历史自制小模型6完成/1中止和70M的8次训练不改写。活动训练0、新增GPU作业0；旧Pod账单状态未知。下一步上下文128的单因素检查尚未启动。

补记：源文件换行SHA检查失败、Windows NumPy种子范围准备失败及配置序列化缺失字段的原记录均保留；完整配置另存resolved-config.json。报告生成脚本首次语法错误日志也保留，该错误不属于训练失败。本次成功是64长度、单种子的已知公开基线；没有完成长上下文或稀疏方法验证。
"""
    with (ROOT / "TIMELINE.md").open("a", encoding="utf-8") as f:
        f.write(timeline)
    save(ROOT / "logs/zoology-handoff-audit.json", dict(utc=now, verified=verified,
        checkpoint_sha256=query["checkpoint_sha256"], distance_answers=4000,
        distance_correct=3964, active_training_jobs=0,current_turn_gpu_jobs_started=0,
        scope="Final artifact verification only; no optimizer steps or forward passes"))
    after = handoff / "after"
    artifacts = changed + ["logs/zoology-baseline-final-audit.json", "logs/zoology-handoff-audit.json",
        "logs/zoology-fresh-distance-audit.json", "logs/zoology-correctness-v2.log",
        "requirements-zoology-cpu-lock.txt", "docs/zoology-baseline-plan-2026-09-14.md",
        "results/zoology-basic-cpu-v1/manifest.json", "results/zoology-basic-cpu-v1/result.json",
        "results/zoology-basic-cpu-v1/resolved-config.json", "results/zoology-query-intervention-v0/manifest.json",
        "results/zoology-query-intervention-v0/result.json", "scripts/finalize-zoology-baseline.py"]
    for path in artifacts:
        target = after / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    files = [dict(path=p.relative_to(handoff).as_posix(),sha256=sha(p)) for p in sorted(handoff.rglob("*")) if p.is_file()]
    save(handoff / "manifest.json", files)
    for item in files:
        assert sha(handoff / item["path"]) == item["sha256"]
    print(json.dumps(dict(status="complete", verified=verified, handoff_files=len(files),
        active_training_jobs=0, report=str(report_path)), ensure_ascii=False))


if __name__ == "__main__":
    main()

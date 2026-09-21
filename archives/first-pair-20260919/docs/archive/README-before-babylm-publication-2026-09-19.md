# 原生稀疏训练与 Prefill

独立项目目录：`D:\ChatGPT\projects\native-sparse-prefill`。

原始研究问题（下述具体候选现已关闭，保留作历史记录）：QSA 提前用于训练时，只监督已选中的上下文块是否限制了索引器发现遗漏信息？少量抽查未选块，能否以更低的总训练成本减少稠密预热，同时保持语言建模和长上下文能力？

最新完成Zoology长度128的CPU检查：40轮、独立新题99.15%，换查询后99.00%。38项检查通过，全部训练结束。本轮没有GPU作业。固定128长度，将查询集中在后半段，明确其对前方记录的依赖；之后再设计稠密、局部与候选稀疏比较。尚无新的稀疏方法贡献，详见STATE.md。

## 从这里开始

- [当前状态](STATE.md)：已完成事项、下一步和实验启动条件。
- [最新128长度结果](docs/zoology-length128-results-2026-09-14.md)：与64基线对照、新题、距离分组与查询干预。
- [128长度预定计划](docs/zoology-length128-plan-2026-09-14.md)。
- [公开最小基线结果](docs/zoology-baseline-results-2026-09-14.md)：训练、新题、查询干预、距离分组与证据边界。
- [公开基线固定方案](docs/zoology-baseline-plan-2026-09-14.md)：作者配置与平台适配。
- [历史四组定位报告](docs/recall-followup-results-2026-09-14.md)：配对训练、新题复核、查询干预和时间轴。
- [最新远距离测试报告](docs/far-recall-results-2026-09-14.md)：两次完整训练、一次失败、查询干预及全部时间轴。
- [远距离测试设计与前作](docs/far-recall-validation-plan-2026-09-14.md)：计算图前提、已知关联回忆任务与扩大门槛。
- [首轮云训练报告](docs/cloud-first-results-2026-09-14.md)：2K、1M/10M控制，逐次结果与时间轴。
- [当前扩大节点与具体安排](docs/scale-decision-2026-09-14.md)：数据、训练量、上下文与费用边界。
- [6组实际主干训练结果](docs/joint-pilot-results-2026-09-14.md)和[逐次时间轴](docs/joint-pilot-run-index-2026-09-14.md)。
- [头间校正结果](docs/head-mixture-results-2026-09-14.md)：60次对照仍未通过门槛，关闭该候选。
- [联合训练固定计划](docs/joint-pilot-plan-2026-09-14.md)：首次真正更新语言模型主干，非新算法或官方Qwen复现。
- [当前主报告：分数尺度定位与新文本确认](docs/calibration-followup-results-2026-09-13.md)：先定位旧参照失效，再用不重合的新文本比较；未支持付费扩训。
- [公开实现核对](docs/indexer-fidelity-audit-2026-09-13.md)：RMS增益、分数尺度、位置编码与训练/推理证据边界。
- [本轮96次索引器运行时间轴](docs/calibration-run-index-2026-09-13.md)：72次因素诊断和24次新文本对照。
- [前轮三检查点验证](docs/realtext-all-checkpoints-decision-2026-09-13.md)：历史报告，评分失效原因由当前报告补充。
- [逐次运行时间轴](docs/run-index-2026-09-13.md)：120 次真实文本索引器运行的开始、结束、单调计时与原始日志。
- [首轮实测结果](docs/mechanism-v0-results-2026-09-13.md)：24 次 CPU 微型诊断，没有稳定收益，暂不启动租卡。
- [具体机制查重](docs/mechanism-overlap-screen-2026-09-13.md)：定向核对与证据边界，未证明不存在重复。
- [当前低预算路线](docs/low-budget-plan-2026-09-13.md)：本地诊断、少数小模型对照和必要重复；总上限 500 美元，首笔最多 30 美元，其余按信号和实测速度安排。
- [原方法草案](docs/native-sparse-training-plan-2026-09-13.md)：机制与候选对照，实际规模按当前低预算路线收缩。
- [前作与可行性](docs/qsa-prefill-feasibility-2026-09-13.md)：QSA、NSA、MoBA、MSA、HiLS 等的证据范围。
- [历史成本情景](docs/native-sparse-training-cost-2026-09-13.md)：原多阶段预算保留供对照，不作为当前默认执行计划。
- [时间轴](TIMELINE.md)：保留准备、训练、评测和失败记录的起止时间。

## 目录

| 位置 | 用途 |
|---|---|
| `src/` | CPU 稀疏注意力参考、索引器诊断运行器、结果核验与历史预算工具 |
| `configs/` | 当前预算配置和各轮冻结实验配置 |
| `literature/` | 各轮原文/实现快照、提取文本、来源与哈希；阅读深度见相应报告 |
| `data/` | 112段冻结诊断文本及208个Q/K/V轨迹，另304段联合训练token数据、来源与哈希 |
| `tests/` | 38项CPU检查，scripts/test-cloud-cuda.py另有4项CUDA检查 |
| `logs/` | 当前状态、环境、预算、数据/运行审计与失效诊断 |
| `results/` | 合成与真实文本索引器诊断、NLL干预、逐次日志、权重和源码快照 |
| `provenance/` | 从旧项目复制文件的清单与校验记录 |

## 运行现有工具

预算计算器仅使用 Python 标准库，可使用 Python 3.10 或更高版本；不需要 GPU。它不会连接云账户或创建实例。其默认配置仍为历史多阶段情景；当前 500 美元上限另存于 `configs/research-spending-cap-v1.json`，该文件为支出约束记录，不能作为旧计算器的 `--config`。训练工作量仍须由实测速度确定。

```powershell
cd D:\ChatGPT\projects\native-sparse-prefill
python src/estimate_native_sparse_cost.py
```

需要保存新预算时使用 `--output logs/<新的文件名>.json`，保留历史版本。每个方法、GPU、模型和上下文长度的吞吐都需要实测，不能把一个配置的速度套用到所有阶段。

已建立独立 `.venv`，使用 Python 3.12.14、PyTorch 2.10.0+cpu。重建 CPU 环境可在 Python 3.12 下运行 `scripts/setup-cpu.ps1`。云端独立环境位于 `/workspace/native-sparse-prefill/.venv-cloud`，CUDA12.4已核验；本机环境仍为CPU。

真实文本环境额外安装 Transformers 4.57.6、PyArrow 22.0.0 和 Matplotlib 3.10.8；完整版本见 `requirements-realtext-lock.txt`，重建脚本为 `scripts/setup-realtext.ps1`。已有数据和结果无需重新下载或训练。以下示例用于需要复现时，输出目录必须另取新名。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
# 重跑时必须用新的输出目录；以下命令会在已有同名结果时拒绝覆盖。
.\.venv\Scripts\python.exe src/run_mechanism.py --output results/mechanism-v0
```

参考实现使用稠密 score/mask 张量做正确性验证；它不代表高效稀疏 kernel。完整记录与已知限制见首轮报告。

真实文本运行器：

```powershell
.\.venv\Scripts\python.exe src/run_realtext.py --config configs/realtext-v1-optimization-check.json --data data/realtext-v0-r1 --output results/realtext-v1-reproduction-NEW
.\.venv\Scripts\python.exe src/verify_realtext.py --data data/realtext-v0-r1 --runs results/realtext-v1-optimization-check --nll results/realtext-v1-nll-intervention --output logs/recheck-NEW.json
```

## 当前接续边界

当前同时保留旧参考和加入RMS增益/分数缩放的诊断实现。新配置通过 `indexer_calibration` 字段显式启用，未改变旧配置的含义。运行器、NLL干预和健康核查均按保存的配置重建索引器。

两个窄方法候选已经关闭。2K云端稠密控制完成后，已建立远端信息隔离测试；64长度基线已跑通，128长度检查完成，最新判断见STATE.md。已有云运行不代表方法新颖性已确认，本轮训练已结束。

每次实验使用独立 run ID，保存 UTC 起止时间、配置、随机种子、代码快照、数据版本、设备、tokens、指标、失败原因与实际机时，并更新 `TIMELINE.md`。预算工具执行记录不能计作训练实验。

本目录于 2026-09-13 从 `structured-teacher-learning` 中单独整理建立。旧项目保留；其中旧选题的实验和结论不属于本项目证据。

公开Zoology基线依赖锁定于`requirements-zoology-cpu-lock.txt`，上游源文件和两处平台补丁在`third_party/zoology-1ad20d1/`；本地日志关闭W&B联网。最新完整结果和接续状态见STATE.md。

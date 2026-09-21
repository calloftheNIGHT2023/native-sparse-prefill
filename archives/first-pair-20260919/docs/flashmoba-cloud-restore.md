# FlashMoBA 云端恢复说明

适用于本阶段固定的 Linux x86_64 / Python 3.11 / torch 2.8.0+cu128 环境。原 `.venv-cloud` 保留，不向其中覆盖安装。Windows 本机只保存与核验归档，不能直接安装这个 Linux wheel。

## 保留内容

- `exports/flashmoba-wheels-v0/`：已打包并在本机核验的 Linux wheel、官方源码 tar 和 manifest；wheel 内扩展与实际 GPU 测试的扩展 SHA 相同。尚未在新机器重新安装验证。
- `results/flashmoba-environment-v0/`、`flashmoba-environment-resume-v1/`：最初安装与续编的未完成记录，保留工具链来源、错误及日志。
- `results/flashmoba-environment-headers-v2/`：最终成功环境记录与 `pip-freeze.txt`。运行环境仍名为 `.venv-flashmoba-v0`。
- `scripts/bootstrap_flashmoba_official.py`：原始安装全过程，初次安装专用，不要在已有目录盲目重跑。
- `src/flashmoba_fixed_metadata.py`：只缓存固定长度信息的调用包装器；不是官方新算法。
- 每项测试都有自己的 `results/flashmoba-*/` 和 `logs/flashmoba-*.log`。历史失败目录不能覆盖。

## 换机器时

先按顺序恢复 `exports/topk-efficiency-cloud-stage-v0.tar.gz`、`exports/block-candidate-stage-overlay-v0.tar.gz`、`exports/flashmoba-stage-overlay-v0.tar.gz` 到新目录，后一个增量覆盖前一个快照的同名文件，并按最后一层 manifest 核验。底包包含本工程测试所依赖的旧模块，不能只复制一个 `.py` 文件。科学训练权重的历史原档仍独立保留。单独的 `flashmoba-cloud-results-v0.tar.gz` 是本轮云端原始证据备份，不包含收尾时写入的所有本地报告。

建立独立 Python 3.11 venv，先从 PyTorch cu128 官方源安装 torch 2.8.0，再安装已记录的基础依赖与校验过的 wheel。完整依赖版本以 `results/flashmoba-environment-headers-v2/pip-freeze.txt` 为准；其中官方模块的本地源路径要替换为迁移后的 wheel，不把旧机器的绝对路径照抄。

当前 wheel 只编译 sm80 cubin，目标是 Ampere/Ada 同主版本架构；不能承诺在 H100、Blackwell 或另一 torch/Python ABI 上直接可用。新卡先运行实际 gate，再开展实验。若需要重新编译，使用官方 commit `39d9ac043b271d046a2181a9991e99a26b67bca1` 和 CUTLASS `a2439551c765c5393aebe557ee75d3a0412d2211`，重新选定目标架构并保留新编译日志。

官方源码 tar 不包含 CUTLASS 子模块，重新编译时须取得该固定版本。Python wheel 不包含主机驱动。此前 toolkit/driver 小版本组合的成功只适用于已验证的机器，不自动推及新主机。若重编译，还需复用 `scripts/repair_flashmoba_headers.py` 中从 NVIDIA Python 包 include 目录构造 CPATH 的做法，避免再次缺少 cusparse.h。

## 恢复后最短检查

使用新结果目录运行 `scripts/verify_flashmoba_official_v1.py`，具体参数先查脚本 `--help`。核对 GPU 型号、torch/CUDA、官方 commit、扩展 SHA 和未修改的源码。原始 `verify_flashmoba_official.py` 对池化精度的假设不符实际实现，其失败目录保留作诊断证据，不能用它替代 v1。

`scripts/verify_flashmoba_metadata.py` 是可选 CUDA Graph 包装器检查，本机型测试在第3项失败并定位到排序组件；它不是普通官方 API 使用的前置条件。没有完整 graph 性能结果。需要研究该包装器时应使用新目录独立修复与验证，不把当前包装器标记为已通过。

正确性门槛通过只代表算子可以用于后续实验，不代表已验证真实文本质量、完整模型吞吐或论文原创性。

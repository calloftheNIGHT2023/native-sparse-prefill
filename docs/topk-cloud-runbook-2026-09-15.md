# 当前云环境与重跑说明

主报告：`docs/topk-cloud-efficiency-results-2026-09-15.md`。本阶段所有实验已结束，没有后台科学训练。

当前 Pod `xe89d7w13itaek`，RTX 6000 Ada 48GB。直连命令：

```powershell
ssh -i REDACTED_CONNECTION_METADATA -p 22100 root@REDACTED_IPV4
```

远端目录 `/workspace/native-sparse-prefill/efficiency-20260915-v1`。本轮环境为 Python 3.11.10，torch2.4.1+cu124，torchvision0.19.1+cu124，Triton3.0.0；运行模型使用 `.venv-cloud/bin/python`。完整包清单见 `logs/cloud-environment-freeze-v0.txt`。虚拟环境采用系统 PyTorch，新容器如更换模板，需重建环境，不能直接假设旧虚拟环境有效。

在 Linux Python 3.11 新机器上可用下列方式恢复独立环境。下载与编译时间也会占用云实例计费时间。

```bash
python3 -m venv .venv-cloud
.venv-cloud/bin/python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
.venv-cloud/bin/python -m pip install -r requirements-router-portable.txt
```

恢复代码、两份冻结模型和已暴露的评测数据后，先进行算子验证。以下输出名仅作示例；每次必须换新名字，不覆盖已有目录。

```bash
cd /workspace/native-sparse-prefill/efficiency-20260915-v1
.venv-cloud/bin/python scripts/verify_triton_selected_attention.py --output results/NEW-aggregation-gate
.venv-cloud/bin/python scripts/verify_triton_topk_selector.py --output results/NEW-ranking-gate
.venv-cloud/bin/python scripts/benchmark_topk_cuda_graphs.py --lengths 4096 8192 16384 --chunk 1024 --include-triton --triton-gate results/NEW-aggregation-gate/verification.json --include-triton-rank --selector-gate results/NEW-ranking-gate/verification.json --output results/NEW-core-benchmark
.venv-cloud/bin/python scripts/evaluate_topk_precision.py --aggregation-gate results/NEW-aggregation-gate/verification.json --selector-gate results/NEW-ranking-gate/verification.json --output results/NEW-precision-replay
```

默认值得保留的基线是 `triton_rank_topk`；显式启用 streaming 才会使用实验性的流式内核。其当前源码配置 BM64 未通过效率目标，不应启动长训练。BM16 历史源码在对应结果目录 `source/src/triton_streaming_selector.py` 中，不能把 BM16 的 gate 拿来验证当前 BM64 源码。

独立归档 `exports/topk-efficiency-cloud-stage-v0.tar.gz` 包含本阶段代码、结果、日志、报告、两份小模型检查点、评测集和 pinned Zoology 文件；不包含 Python/Torch 二进制、SSH 私钥、全部旧项目实验或新的训练语料。原始云端结果另存 `exports/topk-cloud-results-v2.tar.gz`，旧科学总归档仍原样保留。

Pod 尚未停止。SSH 退出不等于停止租卡；本阶段结果均已回传本机。若在控制台暂停，选择 Stop 可以保留该 Pod 的卷盘；Terminate 的卷盘删除语义不同，不能当成暂停使用。实例实际价和旧余额需在 RunPod 账单确认，不能由 GPU 运行秒数代替。

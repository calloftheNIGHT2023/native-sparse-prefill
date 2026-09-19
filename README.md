# Native sparse pretraining: BabyLM D/E pilot

从随机初始化、全参数训练的小型 GDN/QSA 混合语言模型研究。当前比较密集全局层 D 与从第一步使用学习式稀疏全局层 E；其余 GDN 层保持一致。采用 BabyLM 2026 Strict-small，模型约 95M / 97M 参数、上下文最多 2048 token。

## 最新结果：2026-09-19

首对单种子训练均完成，各 **14,122 更新、163,224,857 输入 token、99,998,882 累计词暴露**。这些词暴露来自固定 10M 词语料的重复读取，不是 100M 独立新词。

| 组 | 训练墙钟 | 最终固定 48 窗面板 NLL |
|---|---:|---:|
| D：密集全局层 | 33.024 小时 | 5.033447 |
| E：step 0 起稀疏全局层 | 31.804 小时 | 4.984512 |

**两组开发面板损失在中期约 3.4–3.5 后明显上升，终点接近 5，存在共同过拟合迹象。** E 的终点面板均值略低；部分来源更差，完整结果保留。完整开发集与正式语言任务评测尚未执行。

上述时间来自两张独立 A6000、不同驱动；当前实现仍计算完整分数，**不能作为稀疏加速证据**。这些是单种子先导结果，不是质量等价、完整 Qwen 架构效果或论文贡献已成立的结论。

- [首对训练结果和边界](docs/babylm-first-pair-results-2026-09-19.md)
- [基线优先的研究顺序](docs/babylm-baseline-first-priority-2026-09-17.md)
- [审稿自评及条件式后续](docs/babylm-reviewer-verdict-2026-09-17.md)
- [结构与梯度实现审计](docs/babylm-qsa-implementation-audit-2026-09-17.md)
- [数据边界审计](docs/babylm-document-boundary-audit-2026-09-17.md)
- [完整时间轴](TIMELINE.md)

## 仓库内容

- `src/babylm_hybrid/`：当前模型、训练器、评测和计数。
- `scripts/*babylm*`、`tests/test_babylm*`：准备、运行、数值与工程检查。
- `configs/`：实验配置、参考软件版本；云部署配置是历史记录，不能直接作为新作业授权。
- `evidence/babylm-first-pair/`：GitHub 快照中的每更新压缩日志、全部 58 次面板曲线、终局摘要和计数审计。
- `evidence/data-manifests/`：数据来源、固定版本和原始 SHA；原始语料不在仓库中。
- `docs/`：计划、报告和失败记录。旧 LoRA/CPT、人工检索与早期小实验为历史档案，不能回答本轮从零全参数预训练问题。
- `EXPORT_MANIFEST.json`：GitHub 导出清单、原始与导出 SHA、连接信息脱敏记录。

GitHub 导出目录与本机运行目录分离。训练源码保持原字节；连接元数据只在导出副本中脱敏。脱敏配置内嵌的历史 SHA 指向原始实验文件，不指向修改后的副本。完整模型、优化器状态、环境目录、原始语料及本机控制/连接记录没有上传。第三方目录保留各自许可；本仓库未额外授予原创代码的开源许可。

## 无需 GPU 的证据复核

在 GitHub checkout 根目录运行（Python 3.10+，仅标准库）：

```bash
python scripts/verify_github_evidence_v0.py
```

检查导出 SHA、两组 14,122 次更新的数据顺序/计数/主干学习率、共同随机初始化、科学源码 SHA，以及 58 个配对面板。不会下载数据、加载模型或训练。

## 环境与后续复现

实际科学作业使用 Python 3.11.10、PyTorch 2.10.0+cu128、FP32、TF32 关闭。其余固定版本见 `configs/babylm-reference-runtime-v0.requirements.txt`。CPU 工程环境与 GPU 科学运行分开记录。

原始数据按 `evidence/data-manifests/` 的固定 URL/commit 获取并核验，随后按现有准备脚本重建训练/开发窗口。tokenizer 仅使用训练集；不能让 dev/test 参与训练。历史 frozen manifest 不应被新机器生成的元数据静默替换。新训练应使用单独冻结的新协议和新输出目录，先完成 GPU 数值核验。

下一科学步骤是完整开发集与固定语言任务评测。只有结果提供依据时才追加局部窗口、晚转换、独立种子或效率实验，不默认扩展全部组合。

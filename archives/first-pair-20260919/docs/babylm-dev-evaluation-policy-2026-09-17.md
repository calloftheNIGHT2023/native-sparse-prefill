# BabyLM 开发集窗口与 NLL 入口 v0

本轮只接通开发集评测，未执行科学 D/E 训练，也未运行完整候选模型 forward。数据生成复用固定 tokenizer 与已有 dev token ledger；没有读取最终 test、重训 tokenizer 或改写训练流。

## 完整开发集入口

生成器：`scripts/prepare_babylm_dev_windows_v0.py`。

Manifest：`data/babylm-dev-windows-v0/manifest.json`，SHA256：

`baa53c08d1c26e6dda2f238d2221a1d2cfef4c754ec46327765987f423519619`

窗口与 train 采用同一基础函数/规则：文件及显式 `= = = ... = = =` 标记重置；标记保留在后段，原顺序切为最多 2048 token 的不重叠窗口；尾窗和空白全部保留，不插 BOS/EOS。每个窗口独立重置位置、attention、GDN 递归及短卷积状态。隐藏的上游拼接断点仍未知。

|完整 dev 量|值|
|---|---:|
|原始空白分隔词|10,418,979|
|窗口|18,792|
|输入 token|17,437,534|
|有效 next-token loss 位置|17,418,742|
|保守窗口词暴露|10,420,962|
|切词造成的额外窗口暴露|1,983|
|原样保留的零词/空白记录|12,087|
|丢弃/复制/插入 token|0/0/0|

词计数继续按 tokenizer 字符 offsets 与原始空白词区间相交计；只对切开记录补取 offsets 并逐 token ID 核对，不将原词数硬设为 train 的 10M。开发集这些计数属于评测输入，不能写成训练 token/优化器更新。原先 train/dev 重复审计的限制继续保留，本文不宣称两者文档完全独立。

## 固定的小型监测 panel

文件：`data/babylm-dev-windows-v0/fast-panel.json`，SHA256：

`7eaa40e393a8ab261c224ebce041d775cc044717ba97064f9811833fa3118623`

每源按照 `SHA256('babylm-dev-panel-v0|source|start|end')` 的最小值取最多 8 窗；其中 `source` 使用 manifest 中的完整源文件名（例如 `bnc_spoken.dev`），start/end 为源 token 的左闭右开绝对坐标，UTF-8 编码。按源顺序、源内 hash 排序输出 index；没有用模型结果或窗口长度筛选。

六源各 8 窗，共 **48 窗、77,707 loss tokens**。所有完整开发集窗口仍保留。本 panel 用于成本受控的过程监测，**不是官方 BabyLM fast benchmark**，也不能冒充完整 dev NLL 或最终测试。由于按源各抽相同窗数，它不构成原始语料比例的无偏估计；报告需注明 panel NLL，并列每源结果。

## NLL API

`src.babylm_hybrid.evaluation.evaluate_windows(model, manifest_path, window_indices, device='cpu', max_windows=None)`。

- 模型由调用方提前放到指定 device，函数不搬动参数，不创建优化器。
- 使用公共 `iter_windows`，每窗一个 forward，无跨窗状态；输入复制为 long tensor，避免修改只读 memmap。
- 调用 `model.eval()` 并进入 `torch.no_grad()`；正常返回和异常退出都恢复所有 module 原始 training 标志，包括根 train、子模块 eval 的混合状态。
- 使用模型的 **lm_loss**，不使用包含辅助项的 total loss；显式 `aux_weight=0.0`。
- 每窗验证模型 `token_loss_count` 恰好等于 ledger 的有效目标数，再累计 `lm_loss × loss_tokens`。总 NLL 为累计负对数似然除总 loss token 数，单位 natural log/token；不能把各窗口 loss 直接平均。
- 没有目标的窗口仍 forward 并计入输入/词/forward 账目，但 NLL 贡献为零；整体没有目标时返回 `nll=None`，避免将 `NaN×0` 混入统计。
- 返回 `total` 和 `per_source`：NLL、NLL sum、词暴露、输入/loss tokens、window/forward 数及可用 attention 计数。attention 计数对所有 global 层求和，不能把 reference 分配量当实测加速。
- 返回评测 manifest 哈希、实际窗口索引和零优化器更新；调用方仍要区分工程检查与科学开发集评测。

调用方读取 panel 的 `window_indices`；`max_windows` 只能取该确定顺序的前缀，并会在返回值保留实际索引。重复/越界/非整数 index 和负 cap 均直接拒绝。

## 本轮验证、命令与计数

实际运行过的主体命令：

```powershell
.venv\Scripts\python.exe scripts/prepare_babylm_dev_windows_v0.py
.venv\Scripts\python.exe -m unittest tests.test_babylm_evaluation_v0 -v
```

七项 fake 模型单元测试全部通过，覆盖 token 加权（1 个 loss token 的 NLL=2，3 个的 NLL=4，结果应为 3.5）、空目标、计数、max cap、混合模式恢复、异常恢复、非有限 loss 拒绝和无模型信息的 panel 选取。共 **7 次 fake forward**，没有真实模型训练。

另做读取器首/中/末窗口检查和拒绝 allowlist 外源路径的检查：即使关闭 hash 验证，train/dev 两个允许 token 根目录以外的路径仍拒绝。结果：`results/babylm-evaluation-v0/reader-audit.json`。

唯一额外真实模型检查：固定 tiny 随机初始化 dense 模型、16384 词表，选择“第一个长度为 2..32 的 canonical dev 窗口”而非按分数选择。实际为 **index 1004，7 输入 tokens、6 loss tokens、1 次 CPU forward**；NLL 有限、无参数梯度、train 状态恢复。结果：`results/babylm-evaluation-v0/tiny-dev-integration.json`。该随机 tiny 值只是调用链检查，**不是 D/E 差距或可发表性能结果**。

本轮合计：7 fake forwards + 1 tiny real-model engineering forward；**0 backward、0 optimizer updates、0 GPU 小时、0 科学训练或科学模型评测 forward**。没有多模型对比或完整候选模型实测。

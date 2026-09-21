# BabyLM 小型 GDN/QSA 实现与梯度审计（2026-09-17）

状态：本机组件审计通过部分门槛，完整 BabyLM 模型、GPU 稀疏训练内核与费用门槛尚未通过。本文不修改 STATE/control，不运行科学训练，不下载模型权重。科学优化器更新为 0。

## 1. 可执行结论

能在现有 CPU 环境实现可微的小型 GDN/QSA 数值参考，不需要先迁移 GPU。不能直接启动现有训练脚本：`JointAttention` 是固定单文档 GPTNeoX 代理；旧 `CalibratedIndexer` 的位置编码和损失归约不同于本次目标；NeMo 的公开 QSA 索引器明确被冻结且 forward 带 no_grad。FlashMoBA 不能替代独立 QSA 索引器。

建议先新建一个独立的 `babylm_hybrid` 模块，复用通过校验的低层数学规则，不修改历史代码。第一步产物是 4 层微型全参数测试模型，第二步才构建约 97M 参数候选。CPU reference 通过不等于 GPU 省时实现通过。

## 2. 来源和当前环境

已读 STATE.md、新版 BabyLM 方案、9/13 indexer fidelity 报告及固定快照。2026-09-17 再次打开了 Qwen 原报告，相关公式仍为 §2.1.1/§2.1.2。

- Qwen 技术报告：https://arxiv.org/html/2608.30320v1 ，本地快照 `literature/qsa-prefill-screen-2026-09-13/qsa.txt` 281–346 行。
- NeMo QSA 固定提交：https://github.com/NVIDIA-NeMo/Automodel/blob/f7ccd6f7902634af34c2f31b3294ac250dc97670/nemo_automodel/components/models/qwen3_8_flash_next/qsa.py 。本地 `literature/indexer-fidelity-2026-09-13/nemo-qsa.py`。
- NeMo 层实现：同一提交的 `layers.py`，本地同名快照。
- vLLM 固定提交：319cc5ef19946d34c2e66cbbec5bda29d0bfa328；本地 `vllm-pre-indexer.py`、`vllm-indexer-kernel.py`；来源、SHA256 见原目录 manifest。
- 本机 `torch=2.10.0+cpu`，`transformers=4.57.6`，无 FLA，无 qwen4_exp 模块；有 Qwen3NextGatedDeltaNet 及 PyTorch chunk/recurrent 参考。
- 本次证据：`literature/babylm-qsa-audit-2026-09-17/cpu-component-audit.json`、`existing-cpu-tests.log`、`audit_cpu_components.py`。

## 3. QSA 对应与不能静默继承的差异

|部分|本次应实现的契约|可用来源/现有差异|
|---|---|---|
|独立索引器|Q 为多头，K 为单头，独立于主注意力 Q/K；4×128 index Q、1×128 index K 为可行候选|论文式12；现有参考缩为2头等，不是生产尺寸|
|压缩顺序|先线性投影 K，连续4 token用FP32平均，再K RMSNorm，再按块起始位置RoPE；不能先RoPE再池化|NeMo qsa.py 552–562|
|可学习norm|Q/K各有128维可学习增益，初始化有效增益1|旧QSAIndexer没有；CalibratedIndexer有，但默认开关不可隐式继承|
|RoPE|token query位置、微块起点K位置；NeoX半区配对；theta=1e7；索引器旋转64/128维|旧partial_rope用相邻配对、theta1e4，必须新实现并测试|
|打分|各头内ReLU，再求和；掩码仅允许完整可见同文档块|式15没有1/sqrt(index_dim)；NeMo推理参考有统一缩放；两者top-k等价但训练KL不等价|
|训练温度|新基线明确 `index_score_scale=1.0` 以对齐式15；若以后温度调参须当配方因素，不能称修复官方错误|不得自动沿用旧CalibratedIndexer(scaled=True)|
|选择|完整块结束位置≤当前query，最多64块；展开256token，再加0–3个不完整尾token；当前完整块也可能未入top-k|预算不是“64token”；不额外强制当前块，否则改变基线|
|短前缀|可见完整块≤预算时全选，加尾部；最初0–3token仅尾部|不得对空块集合softmax；top-k ties固定规则并记录|
|teacher|主注意力softmax按头求和并L1归一（等价等权mean），每块max pool，再对所监督完整块归一|尾token可参与主softmax，但不形成KL的完整块标签|
|KL|Early在selected support上；Late预热在全可见support上；空support贡献0；选中1块时KL和梯度恒0|旧subset_kl只平均有块query；式18/20分母是N个query，需新实现“有效loss query总数”分母|
|detach|teacher断梯度；索引器输入hidden断梯度；LM只通过硬路由后的主Q/K/V学习，indexer只由辅助KL学习|这是明确选择的训练契约；公开NeMo是冻结推理indexer，不能据其no_grad声称恢复了官方完整训练detach配方|

必须保留两个独立名字：`full_teacher_restrict_then_renorm` 与 `sparse_core_teacher`。先逐head在全支持softmax再限制，与先在选中支持softmax再跨head平均，一般不等价，因为每头丢弃的概率质量不同。本次Early的可执行参考采用实际稀疏core概率产生的teacher；Late预热采用实际dense core概率。需写入协议，而不能把两者交替使用仍声称完全相同的KL。论文公开描述不足以单独恢复所有训练图和实现细节。

建议契约：`loss = token_CE + lambda_aux * mean_over_global_layers(KL_sum / valid_query_count)`，不是对层求和后让辅助系数随模型深度改变；lambda数值尚未冻结。loss mask是否将文末无next-token位置计入辅助分母须与数据协议统一。

## 4. GDN 接口可复用，但两处必须修正

本机 Qwen3NextGatedDeltaNet 已有 q/k/v/z、beta/decay投影，短因果深度卷积、L2归一、gated-delta recurrence以及输出投影。其输入输出为 `[batch, length, hidden]`。CPU fallback可用于数学参考。

**输出门必须从 Qwen3Next 默认 SiLU 改为本研究要保留的 sigmoid。** 只替换最终RMSNorm×gate，不把 q/k/v 前的SiLU也改成sigmoid。本次审计在内存里用独立SigmoidNorm验证了该接口，未修改site-packages。

**GDN与卷积都必须在文档边界重置。** 本地4.57.6接口没有NeMo的cu_seqlens打包契约；简单传0/1 padding mask不会在中间文档处同时重置conv和recurrent state。最小正确CPU实现可按已确定的真实segment分别调用模块再拼回；GPU必须使用支持varlen/segmented conv和delta kernel的固定实现。仅给QSA加block-diagonal mask不足以避免GDN跨文档泄漏。

另需记录：本机chunk/recurrent实现对L2-normalized Q再乘1/sqrt(d_k)，状态FP32。公式核对和独立oracle须按同一代码约定；不把FP32参考误写成任意精度的数学精确实现。

## 5. 约97M参数候选确实可构造，但并非完整Qwen4

这是尺寸审计候选，不是已批准的最终模型。选择如下：

- vocab=16384、hidden=768、12层：GDN/GDN/GDN/global重复3次。
- SwiGLU intermediate=2048，普通pre-norm残差，输入输出embedding绑定。
- GDN：4个key heads×64，12个value heads×64，conv4；保持value/key head分组比3，缩小value expansion。
- global core：6个Q heads×128，2个KV heads×128，Q/K norm，sigmoid query output gate；RoPE64维、theta1e7。该缩小head维度使旋转比例变为1/2，必须披露，不能称完整尺寸复现。
- indexer：4个Q heads×128、1个K head×128、RoPE64、4-token块、64块预算。
- 不使用MoE、PLE、GR多分支、MTP；这些差异显式列入架构表。

按模块参数算术：dense约 **95,391,000**，Early含3个独立indexer约 **96,866,328**。GDN单层已真实实例化计数；整体数字仍须新模型构建后逐tensor重新核验，不能拿算式代替最终manifest。相同主干的dense对照无indexer，在总参数上略小，应同时报告共享主干与额外索引器参数，不为凑相同总参数偷偷增加dense容量。

在单个真实2048-token段，64完整块+尾部保留494,336个query-key对，占dense causal 2,098,176对的 **23.5603%**。这不是12.5%：因果前缀短时不能按最终query预算除序列长来估计全序列工作量。分段短文本下比例会更高，可能接近100%；必须按真实layout统计。

有微块索引器与KL开销、只有3/12层是global、FFN仍密集，因此上述pair比例不能证明总训练更快。使用显式gather产生的复制、dense route mask或FlexAttention大块覆盖也可能抹掉收益。没有可微的精确路由GPU实现通过计时前，不启动完整收费训练。

## 6. 本次实际完成的CPU验证

执行时间 2026-09-17T20:28:07Z，脚本与JSON在独占来源目录。

1. GDN chunk与recurrent：长度1、7、8、9、17、65，chunk8，检查输出、最终state、q/k/v/g/beta梯度。固定容差rtol2e-4/atol2e-5；全部通过。最大输出绝对误差7.45e-8，最大梯度绝对误差1.91e-6。
2. 独立GDN模块换成sigmoid输出门，输入 `[2,11,32]`，完整反传。dt_bias、A_log、conv、qkvz投影、beta/decay投影、norm、out全部有有限且非零梯度。未做optimizer.step。
3. 重跑已有 `test_sparse_reference`、`test_gathered_core`、`test_joint_attention`，15项全部通过，证明历史数值oracle仍能运行。**这些是旧GPTNeoX/参考算子测试，不是新BabyLM混合模型通过测试。**

复现命令：`.venv/Scripts/python.exe literature/babylm-qsa-audit-2026-09-17/audit_cpu_components.py`。源哈希在JSON。脚本有一次requires_grad张量转scalar的记录警告，不影响数值检查；未产生CUDA或科学训练。

## 7. 新模型上线前的最小CPU测试清单（尚待实现）

|ID|输入与断言|通过标准|
|---|---|---|
|C01|4层tiny混合模型，vocab97/hidden32；D/E/W同一shared state_dict|共享tensor及初始RNG来源完全匹配；indexer RNG独立|
|C02|单文档长度1/3/4/5/255/256/257/259/260；另有非4整除多文档|合法块/尾部/无重复/无未来/无跨文档；第一个真正稀疏query位置零基259|
|C03|显式相位标量oracle检查Q token RoPE、K block-start RoPE以及池化先于RoPE|FP64 rtol/atol1e-10；FP32单独门槛，不以浮点平局验证排序|
|C04|路由固定，GQA头映射独立展开；gather与dense-masked比较前向及q/k/v/主投影梯度|纯FP64参考1e-9；启用FP32softmax分支采用预先固定1e-4级容差|
|C05|选中所有可见块时D/E的LM logits/loss/shared梯度相同|辅助loss关闭；tiny FP32 rtol2e-4/atol2e-5；不能比较路由ties的梯度|
|C06|CE-only与KL-only分开backward|CE：所有主干参数组可达，indexer grad=None；KL：indexer可达，主干及teacher无梯度|
|C07|selected KL手算，2块以上；empty/one-block query；多文档大量短前缀|空/单块loss0；非选score直接导数0；按全部有效query归约，标签和为1|
|C08|全负ReLU点、零scores ties、超大score、nonfinite检测|负区局部导数应为0且记录dead-score比例；不能暗中LeakyReLU或动态温度来使测试通过|
|C09|teacher先全softmax再restrict与稀疏core teacher构造反例|显式确认一般不等价；每个模式调用正确teacher名称，避免隐藏全dense教师费用|
|C10|扰动某文档和所有未来tokens，观察其他文档和更早logits|QSA与GDN状态/conv双重隔离；pad loss无梯度贡献|
|C11|GDN独立逐步oracle与chunk的前后向；拆段/重置等价|长度含chunk边界；状态FP32且finite；不能只测输出|
|C12|完整tiny CE+KL一更新，保存weights/optimizer/RNG/data cursor，再重放第二更新|预期参数组有更新；indexer不漏出optimizer；连续与恢复轨迹一致；完整测试后另记工程更新|
|C13|local-window W使用匹配真实visible预算与相同边界|与手写可见set一致；不加载indexer或KL、不暗加全局锚点|
|C14|L切换前后同主干状态；预热indexer-only，随后解冻全部|冻结只在协议规定预热；所有曝光、前向、时长计账|

梯度“非零”应按参数组/多样输入检查，不要求一次minibatch里每个embedding行或每个ReLU单元都非零。hard top-k的离散选集不可用常规全路径finite differences检验；固定路由oracle和避开边界的score/KL检查分开做。

## 8. 下一步模块顺序及付费门槛

1. 新的config/model模块：不要包裹预训练基座；D/E/L/W共享初始化构造；先tiny4层。
2. 新indexer与selected-teacher模块：FP32池化、正确RoPE、显式score温度、detach与KL归约；逐项对应C02–C09。
3. GDN段边界wrapper和完整模型梯度：C01/C05/C06/C10–C12。
4. 约97M候选构造计数，真实tokenizer/segments批次演练；记录实际稀疏率和每组trainable参数。
5. 再设计GPU精确QSA forward/backward与dense强基线，完成同卡端到端测速。可接受先做正确reference用于筛查，但reference速度不能当稀疏方法潜力的正反证据。

本审计没有解决最终全模型实现、训练超参数、GR是否必须保留、数据边界、可微GPU索引/注意力效率和成本预算。最紧迫的风险是**短文档下实际路由几乎全密集**与**把冻结推理indexer当成可训练indexer**；这两项必须在下一批GPU付款之前排除。

## 9. 独立 QSA 梯度契约检查补充（20:32:37Z）

按主任务要求新增独占脚本 `scripts/check_babylm_qsa_gradient_contract_v0.py`，结果 `results/babylm-qsa-gradient-contract-v0.json`。该脚本独立实现小张量数学契约，不导入旧QSAIndexer，不修改主实现。4×128索引器、64维NeoX式RoPE、theta1e7、FP32微块池化、可学习norm、selected-support teacher和N-query归约均显式。

五组检查通过：

1. selected KL为3.307207；索引器Q/K投影及两norm均有有限非零梯度；hidden、主QKV和teacher无KL梯度，未选score直接梯度为0。
2. 单次随机标签CE反传：主QKV与hidden有梯度，所有indexer参数grad=None。无optimizer.step、无训练循环。
3. 21/19/3 token的多文档（含重复label）未来扰动与跨文档扰动均不改变不相关输出/路由。
4. 单独构造260-token边界：t=255、256、258仍全可见；**t=259首次65完整块竞争64预算，当前完整块可被丢弃，尾部为0**。这排除了“总是强制保留当前完整块”的不同算法。
5. 空support与单块support的KL及score梯度恰为0。

第一次调用因argsort dim需用关键字的API错误在计算前失败，修复并保留 `gradient-contract-first-invocation-failure.txt`；未掩盖失败。科学训练计数仍为0。脚本只验证上述数学契约；使用dense小张量oracle和MHA主fixture，不宣称Qwen官方训练图忠实复现，不替代GQA/GDN/全模型/恢复/性能测试。

复现：`.venv/Scripts/python.exe scripts/check_babylm_qsa_gradient_contract_v0.py`。

# W：原定义固定局部注意力的一轮从头训练

用户在完整C2报告后再次要求“继续”，本批执行此前建议的唯一纯local W。新批次与旧C2有限队列分开；旧数据、代码、结果与负结果全部保留。自动监控仍暂停，不扩种子、候选、sink变体或学习率搜索。

科学条件：W_fixed_local。随机初始化、全参数、第一步即固定局部连接。共享主干参数逐tensor初始SHA与原D完全相同；12层中9层GDN、3层global及其他模型结构不变。每query保留最近至多64个完整块，每块4tokens，再保留原当前不完整因果尾部，无sink。每个连续文档段重置位置、GDN卷积和递推状态。保留连接数精确为4×min(floor((q+1)/4),64)+(q+1)%4，和F的每query预算相同。

W移除索引器及其辅助目标，aux_weight=0，不创建indexer参数、optimizer组或scores。F多出的参数/辅助更新/路由开销如实记账。固定原主干LR3e-4、AdamW betas(0.9,0.95)、eps1e-8、weight decay0.1、主干clip1，原1M真实词warmup与完整词数cosine到0.1倍。seed20260917、data-order-seed20260919、窗口顺序、16窗累计（末步6窗）不变。

完整一轮1413updates、22598训练F/B、10001709词、16325414输入token、16302816监督token。初始与250/500/750/1000/1250/1413共7次48窗面板，336evalF。末100固定1314–1413，纯LM token加权NLL、exp(NLL)和mean(stepPPL)分别报告。唯一final1413模型再评原18792窗完整dev、17437534input、17418742targets、10420962词，18792F/0B/0updates；不使用中途最好点。全队列科学LM F总数22598+336+18792=41726，B22598、updates1413。

旧训练引擎只认识dense/sparse，故兼容调用mode=dense仅用于无indexer优化器分支。新模型内部mode=local，协议scientific_condition=W_fixed_local、actual_policy=local，所有主attention统计显式标local。模型含非参数持久buffer local_attention_contract_v0=[4,64]；旧Dense严格加载会失败。由新factory实现局部训练，禁止把旧日志mode=dense当作密集结果或临时评估mask当作从头W。

正确性门槛：新tiny CPU与CUDA局部mask/oracle、前后向、初始SHA/RNG、因果/文档隔离、全参数梯度、marker与optimizer重放通过；旧D/E原始代码SHA不改。硬件fresh admission与共享flock避免同卡重复工作。CUDA tiny仍用原CPU-CUDA atol1e-4/rtol1e-3、full-support与checkpoint atol1e-6/rtol1e-5，不放宽。随后唯一6update真实模型工程短测，保留原完整词调度但仅96F/B、6工程更新、0过程面板；科学训练重新随机初始化，不复用短测权重。门槛失败保留日志并停止，不自动重跑。

在W看分前采用工程筛查门槛NLL_W−NLL_F≥log(1.01)（约0.00995033），即W PPL比F至少高1%，才考虑扩大当前设置中学习路由有价值的论点。它不是显著性/等价/非劣界限：W更好或差距不够，不宣布统计等价，而是停止扩大该论点；若F通过，只进入更强简单位置对照与独立确认，不能直接出论文。纯local W不是sink+local；后一变体当前不执行。当前dev已用于探索，不能当独立盲测。

运行边界：当前REDACTED_CONNECTION_METADATA的2g.48gb MIG，GPU按已有本Pod租价$1.09/h、含存储保守$1.40/h。预飞6update合作上限600秒，外层615秒TERM并15秒KILL；科学train最大6900秒、完整dev3300秒、整队列10800秒硬截止，执行最多$4.20；连准备本批最多$6，沿用优化周期$20。派发前按互不重叠UTC区间复算已观测开销、闲置和余量，未知历史费用不补零。所有父子同独立PGID，整个科学队列持共享MIG锁，失败停止并清理其进程组，禁止自动resume/延长/重试。

完整分数reference仍计算完整QK；本轮测质量与对照必要性，不能用该W或跨GPU耗时宣称真实稀疏加速。完成后本机逐SHA备份、独立复算并报告；不自动启后续实验、不操作Pod电源或对外发布。

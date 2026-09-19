import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
a=json.loads((ROOT/'results/router-necessity-audit-v0/summary.json').read_text());b=json.loads((ROOT/'results/router-layer-necessity-audit-v0/summary.json').read_text())
for folder in ['router-necessity-audit-v0','router-layer-necessity-audit-v0']:
    src=ROOT/'results'/folder
    for r in json.loads((src/'manifest.json').read_text()):assert hashlib.sha256((src/r['path']).read_bytes()).hexdigest()==r['sha256']
lookup={(r['method'],r['policy']):r for r in a['runs']};single={(r['method'],r['layer'],r['policy']):r for r in b['rows']}
doc='''# 80轮后选路必要性诊断

关键结果：保留第一层原路由，只让第二层固定保留当前任务的4个源值位置，三组都仍超过99%。因此原4对KV/top8测试的高分，不足以单独证明第二层需要按问题动态挑记录。整个模型的主QK仍在做内容选择，不能把这个结果说成模型根本不读内容。

最初同时替换两层时固定位置只有约78%。随后单独改层检查表明，那个降分不能全归因于第二层选择。后一项是看到首项结果后的事后追加，计划和两次结果目录分开保存。

## 同一批题的对照

| 80轮主干 | 原始 | 两层都固定值位置 | 只固定第二层值位置 | 只固定第二层：交换源值 | 第二层用其他题选路 |
|---|---:|---:|---:|---:|---:|
'''
for name in ['exact_r16_shadow','learned_r16','learned_r64']:
    original=lookup[name,'original'];both=lookup[name,'fixed_values'];fixed=single[name,1,'fixed_values'];donor=single[name,1,'other_row_scores']
    doc+=f"| {name} | {original['fresh']:.4%} | {both['fresh']:.4%} | {fixed['fresh']:.4%} | {fixed['swapped']:.4%} | {donor['fresh']:.4%} |\n"
doc+='''
所有干预都因果、最多8条注意力边并含自身/前一位置；值位置固定为0起算的1/3/5/7，剩余远端槽位按位置补齐，不查答案构造掩码。其他题选路为同批索引分数循环错配，可能同时改变token种类和中间激活，下降本身不是某一个机制的因果证明。原模型第二层并非每个问题都覆盖全部4个值，但允许它覆盖全部4个值的替代策略在这个测试上足够。

固定位置策略明确知道公开生成器的布局，是诊断而非通用算法。还不能断言这种固定策略从零训练就能达到同分。下一步应让待检索记录数超过窗口可容纳数量，再检查内容选路是否必要；只加长度但仍保留4对KV不能解决这个问题。

## 已有研究与当前动作

[Zoology](https://arxiv.org/pdf/2312.04927)算法1已公开KV放在开头；[Mamba位置捷径论文](https://aclanthology.org/2025.findings-acl.629/)已有位置模式变化。因此不是新基准/新捷径发现的原创声明。本项目正在补[MSA预热后选中集合KL](https://arxiv.org/html/2606.13392v1)和[KSA加性评分](https://github.com/awni/k_sparse_attention)两种已有机制的token级适配对照，各40轮；不改变本轮已固定的训练数据和预算。

## 证据与计数

第一项18条件/92160答案，第二项12条件/61440答案，合计153600答案、0主干/索引器更新；原路由15360预测与已归档GPU结果完全一致。所有父checkpoint及模型/索引器权重前后哈希一致，数据仍为seed2026091481。没有产生新的独立训练种子或真实文本证据。

原始预测/掩码统计：results/router-necessity-audit-v0和results/router-layer-necessity-audit-v0。各有计划、源码快照、UTC、manifest与逐条件预测；初始源码和后续支持单层干预的源码分别存档，未覆盖先前结果。
'''
(ROOT/'docs/router-necessity-results-2026-09-14.md').write_text(doc,encoding='utf-8');print(doc)

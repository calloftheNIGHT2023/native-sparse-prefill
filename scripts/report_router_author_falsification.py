"""Write an evidence-based closure only after the independent audit passes."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_frozen_router import now
R=Path(__file__).resolve().parents[1]
def main():
    audit=json.loads((R/'results/router-author-falsification-analysis-v0/audit.json').read_text(encoding='utf-8'));assert audit['status']=='passed'
    ev=json.loads((R/'results/router-author-falsification-evaluation-v0/evaluation.json').read_text(encoding='utf-8'));rows=ev['conditions'];native=[x for x in rows if x['name'].startswith('router-author-falsification-') and x['clean_and_swap_gate'] and json.loads(((R/x['checkpoint']).parent/'fit-result.json').read_text(encoding='utf-8'))['curves'][-1]['development_accuracy']>=.99]
    assert native,'No closure by counterexample without a passing native final checkpoint'
    fits=[json.loads((R/x['directory']/'fit-result.json').read_text(encoding='utf-8')) for x in audit['fits']]
    lines=['# 当前“原生稀疏训练障碍”候选：学习率反证','',f'更新：{now()}。本轮拟合、统一新题评测和 CPU 复核已完成。','',
    '**结论：当前解释应撤回，当前论文候选结题。** 原来在学习率0.01下从零训练原生top8表现很差；只采用另一档作者学习率、保持模型/数据/稀疏算子不变，就出现了通过新题与换答案测试的反例。因此，现有结果不能支撑“必须引入新的稀疏训练修复方法”。19轮时记录的分数没有错，错误风险在于把未经分别调优的差距解释为方法障碍。','',
    '这里的原生top8指从第一步限制注意力聚合到最多8个位置（2个局部位置+6个内容选择位置）。参考实现仍计算完整QK以选位置，不是高效稀疏kernel，也没有实际预训练加速或Qwen4验证。','',
    '## 统一新测试结果','',
    '新题seed2026091701，1024条×16个答案；每条另做一次源值交换（1024个答案）；填充噪声seed2026091702。与训练、开发及此前三个已暴露测试集合检查无重复。所有模型使用已冻结的最终检查点，不依新测试选择轮次。','',
    '| 条件 | 最终轮次 | 新题准确率 | 交换后准确率 | 填充噪声准确率 |','|---|---:|---:|---:|---:|']
    labels={'dense_lr01_epoch19':'完整注意力，lr0.01，seed123','dense_lowlr_epoch64':'完整注意力，lr0.002154，seed123','native_lr01_epoch19':'原生top8，lr0.01，原19轮','router-author-falsification-resume-v0':'原生top8，lr0.01，续训64轮','router-author-falsification-lowlr-v0':'原生top8，lr0.002154，seed123','router-author-falsification-confirm-v0':'原生top8，lr0.002154，seed124'}
    for x in rows:lines.append(f"| {labels.get(x['name'],x['name'])} | {x['epoch']} | {100*x['test']['accuracy']:.4f}% | {100*x['swapped']['accuracy']:.4f}% | {100*x['noisy']['accuracy']:.4f}% |")
    lines+=['','通过新题与交换测试的本轮原生条件数：'+str(len(native))+'。不同初始化使用相同训练数据，不等于不同数据集重复。噪声是额外分布变化诊断，单列展示，预定主要门槛是开发/新题/交换准确率至少99%。','',
    '## 这轮实际做了什么','',
    '| 训练任务 | 学习率 | 初始化种子 | 范围 | 新增更新 | 首次开发≥99% | 最终开发准确率 | 任务墙钟秒 |','|---|---:|---:|---|---:|---:|---:|---:|']
    for f in fits:lines.append(f"| {f['method']} | {f['lr']:.9g} | {f['seed']} | {f['starting_epoch']}→64 | {f['new_main_updates']} | {f['first_observed_development_crossing'] if f['first_observed_development_crossing'] is not None else '未达标'} | {100*f['curves'][-1]['development_accuracy']:.4f}% | {f['wall_seconds']:.2f} |")
    lines+=['',f"本轮共新增 {audit['main_updates']:,} 次正式主模型更新，索引器更新0，{audit['input_tokens']:,} 个输入token暴露、{audit['supervised_answers']:,} 个监督答案暴露；是重复训练数据的暴露量，不是独立语料量。旧19轮前缀没有重复计数。CPU/CUDA续训等价测试各3次独立技术更新，均不进入科学检查点。",'',
    '前一轮已有6次正式拟合；本轮包含1次继续拟合和2次从初始化拟合，不应称为3个独立初始化重复。完整UTC起止、每步损失/学习率/时间见各结果目录events.jsonl及fit-result.json。','',
    f"三项任务记录墙钟总计 {sum(f['wall_seconds'] for f in fits)/60:.2f} 分钟。该数字不等于Pod账单；没有租新卡，当前账单仍未核对。",'',
    '![实际开发曲线](../results/router-author-falsification-analysis-v0/curves.png)','',
    '## 为什么撤回当前角度','',
    '1. 第19轮和第64轮在lr0.01下的失败是真的；但原生方法在另一档普通学习率成功，说明这些失败不是该任务必须新增机制才能解决的证据。','2. 低学习率反例与原失败训练的初始化哈希、数据哈希及第一步损失和梯度范数完全一致；相同模型辅助代码SHA保持。差异是学习率及其同形余弦轨迹。','3. 新题测试排除了单纯记住训练样本；交换测试检查它会随当前序列中的对应值改变答案。有限实验不能证明所有稀疏配置稳定、所有任务适用或实际训练更便宜。','4. 这也不构成一篇“新方法论文”：目前只有普通优化设置的反例。已有SSA讨论梯度缺失与完整/稀疏对齐，ZETA使用历史摘要补充信息，另有集中注意力的形成时间与收敛理论。','',
    '相关原文及具体章节在 [查重证据](router-author-falsification-literature-2026-09-14.md)。不能把本任务的现象称为这些论文的完整复现，亦不声称它们的条件定理被本实验推翻。','',
    '## 验证与保留材料','',
    f"CPU逐答案复现 {audit['cpu_replayed_answers']:,} 个GPU测试预测；全部一致。日志计数、每步有限梯度、最终优化器/scheduler、计划和源码SHA、原始父文件SHA均检查通过。审核结果：results/router-author-falsification-analysis-v0/audit.json。",'',
    'config.json保留作者lr0.01/seed123的模板；本次实际覆盖参数在每个provenance.json、fit-result.json、每步optimizer日志及effective-config.json中明确记录。有效配置补充文件不改动原模板。','',
    '方案：docs/router-author-falsification-plan-2026-09-14.md；事前补充：docs/router-author-falsification-addendum-2026-09-14.md；同一变化审计：provenance/router-author-falsification-single-change-audit-v0.json；新题原始预测：results/router-author-falsification-evaluation-v0/evaluation.json；CPU复核：同目录cpu-replay.json。','',
    '当前候选到此停止追加训练，不将普通调参的成功包装成新贡献。更广泛的原生稀疏预训练可行性没有被否定；需要另一个经查重且可检验的新命题，才能开启下一候选。','']
    (R/'docs/router-author-falsification-results-2026-09-14.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()

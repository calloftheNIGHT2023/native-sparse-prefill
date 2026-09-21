"""Render only an independently verified complete W union; never run a model."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
audit_path = HERE / 'metrics-audit.json'
audit = json.loads(audit_path.read_text(encoding='utf-8'))
assert audit['status'] == 'passed_complete_two_segment_W_union_audit'
assert audit['complete_quality_evidence'] and not audit['failures']
metrics = audit['full_dev']
labels = {'D': '密集注意力', 'E': '原学习式稀疏路由', 'F': '分数尺度修正后的稀疏路由', 'W': '固定局部窗口，无学习式路由'}
lines = ['# W单轮训练：完整开发集评测结果', '',
         f"核验时间：{datetime.now(timezone.utc).isoformat()}。", '',
         '剩余6,390窗已补齐，与原12,402窗组成固定完整开发集：18,792窗、17,418,742个监督token。两段无重复或遗漏，原超时记录完整保留。', '',
         '## 完整开发集结果', '', '| 组别 | 设置 | NLL | PPL（越低越好） |', '|---|---|---:|---:|']
for tag in ['D', 'E', 'F', 'W']:
    m = metrics[tag]['total']
    assert m['records'] == 18792 and m['loss_tokens'] == 17418742
    lines.append(f"| {tag} | {labels[tag]} | {m['nll']:.9f} | {m['ppl']:.6f} |")
if all(metrics['W']['total']['nll'] < metrics[tag]['total']['nll'] for tag in ['D','E','F']):
    lines += ['', '本次完整开发集上，固定局部窗口W的PPL低于密集D、原稀疏E和修正稀疏F。这个对照没有显示学习式选块相对于固定局部窗口的额外语言收益；继续增加路由改动缺少当前证据支持。它不证明局部注意力在其他数据、规模或种子上普遍更好。']
lines += ['', 'W相对其他组的完整开发集差异（负值表示W的PPL更低）：', '', '| 对照 | W减对照的NLL | W相对PPL变化 |', '|---|---:|---:|']
for tag in ['D', 'E', 'F']:
    d = audit['W_minus_baseline'][tag]['total']
    lines.append(f"| {tag} | {d['nll_difference']:+.9f} | {100*d['ppl_relative_difference']:+.3f}% |")
screen = metrics['W']['total']['nll'] - metrics['F']['total']['nll'] >= math.log(1.01)
lines += ['', f"原先冻结的学习式路由扩展筛查条件（PPL_W / PPL_F ≥ 1.01，即W的PPL至少比F高1%）在本种子点估计上{'满足' if screen else '不满足'}。这个筛查结果不是显著性或非劣性检验。", '',
          '## 各来源', '', '| 来源 | 监督token | D PPL | E PPL | F PPL | W PPL |', '|---|---:|---:|---:|---:|---:|']
for src, entry in metrics['W']['per_source'].items():
    values = ' | '.join(f"{metrics[tag]['per_source'][src]['ppl']:.6f}" for tag in ['D','E','F','W'])
    lines.append(f"| {src} | {entry['loss_tokens']:,} | {values} |")
lines += ['', '## 查询历史长度（窗口内）', '', '| 查询历史长度 | 监督token | D PPL | E PPL | F PPL | W PPL |', '|---|---:|---:|---:|---:|---:|']
for pos in ['1-256','257-512','513-1024','1025-2048']:
    values = ' | '.join(f"{metrics[tag]['position'][pos]['ppl']:.6f}" for tag in ['D','E','F','W'])
    lines.append(f"| {pos} | {metrics['W']['position'][pos]['loss_tokens']:,} | {values} |")
lines += ['', '位置桶是描述统计；较后位置不自动意味着任务依赖远距离信息。', '',
          '## 训练、评分与费用账目', '',
          '原W仍为同一次随机初始化、全参数1413步训练，16,325,414输入token。补评没有执行任何反向传播或优化器更新，没有改变模型、样本、评分器、精度或检查点选择。', '',
          '完整分数用两段逐窗NLL总和除以全部监督token，再取exp；没有直接平均两段PPL。原超时前缀12,402次前向，加补评6,390次前向，共18,792次。迁移tiny的9次模型前向、7次反向、3次工程更新以及48窗工程重放另计，不冒充科学训练。', '',
          '| 评分段 | 起始UTC | 结束UTC | 评分秒数 |', '|---|---|---|---:|']
for key, label in [('original_prefix','原超时前缀'),('new_remainder','补齐剩余窗口')]:
    t = audit['timing_segments'][key]
    lines.append(f"| {label} | {t['started_utc']} | {t['completed_utc']} | {t['elapsed_wall_seconds']:.3f} |")
lines += ['', f"两段评分器活动时间合计 {audit['timing_segments']['summed_scorer_active_wall_seconds']:.3f} 秒，不包括迁移、部署、工程检查、读回和间隔空闲。新卡小时租价及完整发票未核实，费用不填0、不套旧卡单价。", '',
          '## 能支持的结论与边界', '',
          '这里能比较同一个小型GDN/注意力混合模型、同一BabyLM训练数据和一轮训练下的完整开发集点估计。该开发集已参与开发判断，不是新的独立盲测；只有一个配对种子，不能据此宣称统计优势、非劣、完整Qwen架构有效或论文证据已经充分。', '',
          '当前参考实现仍计算完整注意力分数。不同GPU、共享负载、两段评分或逻辑连接数都不能代替真正稀疏内核的端到端训练成本证据。', '',
          '原超时文件与新增补评文件分别保留，原训练检查点与全部评分日志在本机备份并逐SHA核验。没有自动追加新训练、候选、种子或监控任务。', '',
          f"独立审计：{audit['checks']:,}项通过，0失败。原始文件及SHA见 `metrics-audit.json`、`execution-audit.json` 和 `evidence-index.json`。", '']
target = HERE / 'REPORT.md'
assert not target.exists()
target.write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'report': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'model_calls': 0}))

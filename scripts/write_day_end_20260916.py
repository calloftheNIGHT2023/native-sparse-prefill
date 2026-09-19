"""Summarize audited evidence after the final bounded stage; never starts GPU work."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, tarfile, io

R = Path(__file__).resolve().parents[1]
def load(name):
    return json.loads((R/name).read_text(encoding='utf-8'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    now = datetime.now(timezone.utc).isoformat()
    gentle = load('results/gentle32k-audit-v0/result.json')
    old = load('results/expanded76-audit-v0/result.json')
    fresh = load('results/fresh256-audit-v0/result.json')
    assert gentle['status'] == old['status'] == fresh['status'] == 'verified'
    t = gentle['tradeoff']
    lines = ['# 今日收尾：稀疏训练成本与质量', '', f'生成时间 UTC：{now}。用户要求当前轮次完成后结束今天的工作，不启动后续实验。', '',
             '当前结论以已审计结果为准。模型是 Qwen2.5-0.5B，LoRA 调优、32K 上下文；尚不是 Qwen4 全参数原生稀疏预训练的证明。', '',
             '## 同76窗口、同128步的开发集比较', '',
             '|配置|种子数|训练秒/种子|PPL|短题|无原文|32K长题|',
             '|---|---:|---:|---:|---:|---:|---:|']
    rows = [x for x in old['summary'] if x['step']==128]
    rows += [x for x in gentle['summary'] if x['k']==48 and x['step']==128]
    for x in rows:
        ac=x['accuracy'];name='密集' if x['k']==0 else f"K{x['k']}"
        lines.append(f"|{name}|{x['seeds']}|{x['train_seconds']:.2f}|{x['ppl']:.4f}|{ac['short']:.2%}|{ac['no_context']:.2%}|{ac['long32768']:.2%}|")
    lines += ['',f"K48 相对复用密集128步对照：训练循环节时 {t['training_time_saving_percent']:.2f}%，PPL 变化 {t['ppl_cost_percent']:+.2f}%，长题准确率变化 {t['task_changes_pp']['long32768']:+.2f} 个百分点。", '',
              '密集控制在旧阶段训练，本轮重放两份密集校准值、核验环境和逐字节初始化一致后复用。训练时间跨阶段比较是成本估计；此前同轮交错短测速中，M64/K48三轮节时1.45%、3.22%、3.51%。其他温和档位慢于密集，均淘汰。这里的时间包含前反向与更新，不含加载、评测、编译、传输或闲置。', '',
              '## 不把不利证据藏起来', '',
              '- K32 曾在旧64篇文章上接近密集，但更换256篇项目内未使用文章后，长题落后13.48个百分点，文章配对95%区间为[-18.75,-8.01]个百分点。该配置的训练省时约17%，不等于其长文本质量已接近。',
              '- K48本轮仍是原64篇开发文章和原两颗种子，不能当作独立确认；不能以区间覆盖零来宣称等价。',
              '- 零更新密集模型仍需列为基线，不能只拿更贵但未改善任务成绩的密集训练模型比较。', '',
              '## K48逐文章差距', '', '|指标|平均差值pp|文章配对95%区间pp|','|---|---:|---|']
    u=gentle['paired_article_uncertainty']
    for name,x in u['sparse_minus_dense'].items():
        lines.append(f"|稀疏减密集：{name}|{x['mean_pp']:+.2f}|[{x['ci95_pp'][0]:+.2f}, {x['ci95_pp'][1]:+.2f}]|")
    for name,x in u['long_minus_no_context'].items():
        lines.append(f"|{name}：长题减无原文|{x['mean_pp']:+.2f}|[{x['ci95_pp'][0]:+.2f}, {x['ci95_pp'][1]:+.2f}]|")
    lines += ['', '上述区间条件于两颗固定种子，复用开发集，不表示跨训练种子、任务或模型规模的一般结论。', '',
              '## 今天的收尾与下次入口', '',
              '- K48阶段新增256科学更新、6诊断更新、576任务计分前向，已收集全量检查点和原始预测。累计5760科学更新、235诊断更新、22112任务计分前向；历史不同规模实验分别计账，不能把所有更新当成同一模型的训练量。',
              '- 原始证据包已下载并逐文件验证；优化器状态、学习率、数据顺序、初始化和校准重放均通过审计。',
              '- Python及49个依赖包版本、原始CUDA扩展安装包、新M64/M128N32模块与源代码均已保存；本轮没有执行空环境恢复测试。',
              '- 本轮没有配置自动追加训练。停卡是否成功以 logs/day-end-stop-status.json 为准；没有停止本机或关闭用户程序。',
              '- 下次先读取本报告和K48原始审计结果，判断有限省时是否足以抵偿质量损失，再决定新的实验设计。今天不扩大模型、不租新卡。', '',
              '相关文件：docs/gentle32k-results-2026-09-16.md、docs/fresh256-results-2026-09-16.md、docs/m64-density-cost-results-2026-09-16.md、docs/m128n32-density-cost-results-2026-09-16.md。', '',
              f"K48原始包SHA256：{gentle['archive']['sha256']}。"]
    doc=R/'docs/day-end-report-2026-09-16.md';doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    names=['docs/day-end-report-2026-09-16.md','docs/gentle32k-results-2026-09-16.md','docs/fresh256-results-2026-09-16.md','docs/m64-density-cost-results-2026-09-16.md','docs/m128n32-density-cost-results-2026-09-16.md','results/gentle32k-audit-v0/result.json','results/expanded76-audit-v0/result.json','results/fresh256-audit-v0/result.json','provenance/gentle32k-protocol.json','provenance/gentle32k-analysis-plan-v0.json','scripts/report_gentle32k.py','scripts/paired_article_uncertainty.py','scripts/write_day_end_20260916.py']
    names += [f.relative_to(R).as_posix() for f in (R/'provenance/day-end-20260916-v0').glob('*') if f.is_file()]
    archive=R/'exports/day-end-report-20260916-v0.tar.gz';assert not archive.exists()
    manifest=[]
    with tarfile.open(archive,'w:gz') as tf:
        for name in sorted(set(names)):
            p=R/name;raw=p.read_bytes();manifest.append(dict(path=name,sha256=sha(p),bytes=len(raw)))
            item=tarfile.TarInfo(name);item.size=len(raw);tf.addfile(item,io.BytesIO(raw))
        raw=json.dumps(manifest,indent=2).encode();item=tarfile.TarInfo('manifest.json');item.size=len(raw);tf.addfile(item,io.BytesIO(raw))
    proof=dict(sha256=sha(archive),bytes=archive.stat().st_size,files=len(manifest),utc=now)
    archive.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(report=str(doc),tradeoff=t,archive=proof)))

if __name__=='__main__':
    main()

import json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def save(p,obj): p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
now=datetime.now(timezone.utc).isoformat()
summary=json.loads((ROOT/'results/cloud-controls-summary.json').read_text()); runs=summary['runs']
assert len(runs)==2 and all(r['status']=='complete' for r in runs)
audit=json.loads((ROOT/'logs/cloud-controls-audit.json').read_text()); assert len(audit['runs'])==2
cloud_updates=sum(r['updates'] for r in runs); cloud_tokens=sum(r['tokens'] for r in runs)
last=runs[-1]; gate=last['fixed_context_gate']; dg=gate['dev_gap_nats']; tg=gate['test_gap_nats']
p=ROOT/'logs/control-state.json'; state=json.loads(p.read_text())
state.update(updated_utc=now,status='cloud_controls_complete_backed_up_pod_stop_needed',active_training_jobs=0,
    gpu_jobs_started=2,gpu_scientific_training_runs_completed=2,cloud_backbone_updates=cloud_updates,
    cloud_training_tokens=cloud_tokens,language_model_training_runs=8,training_runs=8,
    language_model_optimizer_updates=1536+cloud_updates,language_model_training_tokens=393216+cloud_tokens,
    measured_gpu_throughput_available=True,current_topic_new_cloud_spend_usd=None,
    cloud_context_gate_passed=gate['passed'],cloud_billing_status='pod_running_manual_stop_required',
    current_decision_report='docs/cloud-first-results-2026-09-14.md',audit_report='logs/cloud-controls-audit.json',
    formal_cuda_training_runner_scope='Dense 2K control only; sparse joint wrapper CUDA numerics verified, no formal cloud sparse joint run yet.',
    next_action='Stop Pod from RunPod console after verified local backups. Inspect fixed context-control result and design stronger held-out far-context readout before any further expansion; no new algorithm contribution established.')
save(p,state)
p=ROOT/'configs/next-scale-stage-v0.json'; c=json.loads(p.read_text())
c.update(status='dense_controls_completed_local_backups_verified',corpus_status='Ready: data/cloud-articles-v2,384train windows,16dev articles,13test articles,2048context',
    runtime_status='A40 CUDA verified; 2K gather short test15.14GiB, dense formal training4.04GiB; formal runs complete',
    confirmed_new_cloud_spend_usd=None,context_gate_passed=gate['passed'],results='results/cloud-controls-summary.json')
save(p,c)
p=ROOT/'configs/research-spending-cap-v1.json'; c=json.loads(p.read_text())
c.update(cloud_jobs_started_by_this_project=2,cloud_bill_reconciled=False,automatic_budget_enforcement_implemented=False,
    current_pod_stop_status='manual_stop_needed_after_verified_backup',updated_utc=now)
save(p,c)
lines=['# 当前状态：A40首轮云控制实验完成','',f'更新：{now}。项目：D:/ChatGPT/projects/native-sparse-prefill。','',
    '**没有运行中的训练；Pod计费尚未停止。** 当前只有SSH权限，RunPod CLI缺少API凭证，可控内置浏览器未登录。两个运行的完整模型、优化器、日志已回传并逐文件SHA256校验；可以在控制台点Stop，不必删除Pod。首阶段30美元、总预算500美元不变，实际账单未核实。','',
    '## 本次完成','',
    '- A40 46068MiB，Python3.11.10、PyTorch2.4.1+cu124、Transformers4.57.6；23项CPU检查和4项GPU数值/梯度检查通过。',
    '- 256、512、1024、2048各4次技术更新；2K参考gather峰值15.14GiB，后3步约0.249秒。实际dense训练峰值4.04GiB，每步约0.11秒。',
    '- 2K文章数据：384训练窗口来自230篇文章（单遍786,432个预测tokens）、16验证文章、13测试文章；排除旧416段落所在整篇文章。长跑重复同一小语料，不能称为10M独立tokens。',
    f'- 两次真实GPU主干训练分别为1,048,576与9,998,336 tokens，共{cloud_updates:,}更新/{cloud_tokens:,}tokens。同一随机初始化种子41，衰减计划不同，不能充当种子重复。',
    '- checkpoint包含模型、索引器、优化器、CPU/CUDA随机状态和阶段信息；原始事件、配置、源码快照、逐文章NLL、时间轴齐全。','',
    '## 本次结论','',
    f'10M控制的“开头+附近 NLL − 完整上下文 NLL”为开发{dg:.6f}、测试{tg:.6f} nats。预先固定的双集合0.02门槛'+('已通过。' if gate['passed'] else '未通过。'),
    '', '这是同一稠密权重上更换可见上下文的干预结果，还没有新的稀疏训练方法比较或原创贡献。模型能使用部分历史；局部注意力通过多层传播信息，差距小不能证明稀疏训练无损。当前任务是否需要更强远端依赖读出见最新报告。','',
    '## 仍需遵守的边界','',
    '- 未通过的均匀集合外抽查、头间normalizer两条变体保持关闭，不换名反复扫参。',
    '- QSA/MSA原生训练、预热和辅助监督已有前作；没有已确认原创贡献。纯GPTNeoX缺少Qwen Gated DeltaNet混合路径。',
    '- 不自动追加模型规模、种子或超过10M的训练。下一步要先确定更有区分度且新的验证任务。',
    '- 退出Python/SSH不等于停Pod计费；未配置自动停卡，不把预算配置文件当作自动预算控制。','',
    '## 历史证据与入口','',
    '- 最新报告：docs/cloud-first-results-2026-09-14.md；汇总results/cloud-controls-summary.json；审计logs/cloud-controls-audit.json。',
    '- 本轮原始数据与模型：results/cloud-dense-context-v0/、results/cloud-dense-context-10m-v0/。连接和环境provenance/cloud-connect-2026-09-14/、logs/cloud-connection-2026-09-14.json。',
    '- 此前6次CPU主干联合训练，1536次更新/393216tokens，见docs/joint-pilot-results-2026-09-14.md。',
    '- 冻结主干真实文本主实验204次/221994索引器更新；另72次尺度诊断/28800更新，合成24次/2880更新，不能算主干训练。',
    '- 本轮两次数据准备因文章数量不足失败，保留v0/v1进度与v1失败日志。历史v1 UTC约1.34秒回拨保留原记录；新GPU运行UTC按顺序。',
    '- 用户授权、预算、进展：TIMELINE.md、logs/control-state.json、configs/research-spending-cap-v1.json。']
(ROOT/'STATE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
p=ROOT/'README.md'; s=p.read_text(encoding='utf-8'); start=s.index('均匀抽查和头间归一化校正'); end=s.index('\n\n## 从这里开始',start)
s=s[:start]+'均匀抽查和头间归一化校正已关闭。最新完成 A40 上2K上下文的1M/10M稠密控制训练，结果、模型、日志已回传并校验；23项CPU和4项CUDA检查通过。当前仍无确认原创贡献。训练已结束，Pod仍需在控制台停止以结束GPU计费。此前全部实验保留，当前决定见 STATE.md。'+s[end:]
s=s.replace('- [当前状态](STATE.md)：已完成事项、下一步和实验启动条件。','- [当前状态](STATE.md)：已完成事项、下一步和实验启动条件。\n- [首轮云训练报告](docs/cloud-first-results-2026-09-14.md)：2K、1M/10M控制，逐次结果与时间轴。')
s=s.replace('CUDA 训练环境尚未安装。','云端独立环境位于 `/workspace/native-sparse-prefill/.venv-cloud`，CUDA12.4已核验；本机环境仍为CPU。')
s=s.replace('| `tests/` | 已通过的20项因果/文档/梯度、路由、gather一致性与数值检查 |','| `tests/` | 23项CPU检查，scripts/test-cloud-cuda.py另有4项CUDA检查 |')
st=s.index('均匀抽查没有通过新文本验证门槛'); en=s.index('\n\n每次实验使用',st)
s=s[:st]+'两个窄方法候选已经关闭。现已完成2K云端稠密控制；后续先审查任务对远端信息是否敏感，再决定稀疏比较。已有云运行不代表方法新颖性已确认，当前没有后台训练作业。'+s[en:]
p.write_text(s,encoding='utf-8')
with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as f:
    f.write(f'\n## {now} 云端控制完成与回传审计\n\n')
    for r in runs:
        f.write(f"- {r['name']}：{r['started_utc']} 至 {r['finished_utc']}，{r['updates']}更新/{r['tokens']}tokens，训练{r['training_seconds']:.2f}秒；完整备份已校验。\n")
    f.write(f'\n10M上下文门槛：dev={dg:.6f},test={tg:.6f},pass={gate["passed"]}。训练全部完成；未确认新贡献。尚需用户控制台Stop，不能把当前实际账单登记为0或把进程退出当作停卡。\n')
print(json.dumps(dict(status=state['status'],gpu_runs=2,updates=cloud_updates,tokens=cloud_tokens)))

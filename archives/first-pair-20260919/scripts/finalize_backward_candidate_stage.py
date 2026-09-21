"""Write the evidence-backed report, run index, current state, and small overlay."""
import hashlib
import json
import statistics
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(p): return json.loads((ROOT/p).read_text(encoding='utf-8'))
now=datetime.now(timezone.utc).isoformat()
old=read('results/flashmoba-long-backward-original-v0/result.json')
new=read('results/flashmoba-long-backward-barrier-v0/result.json')
train=read('results/flashmoba-backward-training-v0/result.json')
controller=read('results/flashmoba-backward-stage-controller-v0/result.json')
backup=read('provenance/flashmoba-backward-cloud-local-verification-v0.json')
gpu=read('provenance/flashmoba-backward-final-gpu-v0.json')
assert controller['status']=='complete' and controller['integration_repeat_pass']
assert new['deterministic_repeat_pass'] and new['reference_tolerance_pass']
assert backup['all_members_sha256_verified']
timings=[]
for mode in [False,True]:
    medians={}
    for label in ['original','barrier']:
        samples=[]
        for p in (ROOT/'results').glob(f'flashmoba-backward-timing-*-{label}-v0/result.json'):
            row=json.loads(p.read_text(encoding='utf-8'))
            samples.extend(next(x['milliseconds_per_backward'] for x in row['rows'] if x['deterministic']==mode))
        medians[label]=statistics.median(samples)
    timings.append(dict(deterministic=mode,**medians,change_percent=100*(medians['barrier']/medians['original']-1)))
(ROOT/'results/flashmoba-backward-package-v0/timing-summary-local.json').write_text(json.dumps(timings,indent=2)+'\n',encoding='utf-8')
report=f'''# 8K反向同步修复与完整模型短训练：本轮验证通过

更新 {now}。新卡为RTX 6000 Ada（48GB）。本轮把8K反向不稳定问题缩小到一个共享索引同步补丁，并完成内核和完整模型的验证。

直观地说：以前同一份输入重复求梯度会明显变动；增加一次必要的线程块同步后，在当前测试中，“确定性”模式每次给出的梯度完全一样，完整模型的短训练也能完全重复。此结果支持共享索引读写竞争的解释，尚不是跨硬件的普遍结论，也没有单独做逐条指令的racecheck或同编译环境的补丁回退对照。

## 固定输入反向对照

输入是Qwen2.5-0.5B保存的真实8K QKV，Q为[8192,14,64]，K/V为[8192,2,64]，BF16，块长128、K4含当前块。使用同一份已保存的FP32池化结果，每个模式只建立一次前向图，再用相同上游梯度反向16次，没有优化器更新。

| 检查 | 原版 | 同步补丁 |
|---|---:|---:|
| 确定性模式，全Q梯度后15次相对首轮L2差 | 2.18%–6.92% | 0，逐位一致 |
| 确定性模式，128行相对FP64参考L2差 | 2.58%–34.69% | 每次约2.58% |
| 默认模式，全Q重复差最大值 | 1.20e-6 | 1.31e-6 |
| 两种模式中K/V重复差 | 0 | 0 |
| 独立短序列前向、反向、因果性检查 | 12/12通过 | 12/12通过 |

候选通过预先写下的门槛：128行实际掩码匹配、确定性16次Q/K/V逐位相同、所有选中参考相对L2不超过5%、12项标准检查通过。默认模式仍有普通浮点累加顺序造成的小量差异，没有把它宣称为逐位确定。

128个参考位置一半来自旧重复差异较大的位置，一半随机抽取；它们是有偏诊断集合，不代表全模型错误率。实际K4路由与离线重建在114688行中有2行不同，但这2行不在参考集合中；128行全部匹配，因此此处梯度参考比较成立。2个差异位置的候选分数非常接近，具体值保存在k4-full-mask-differences-local.json。

![反向重复差与参考差](figures/flashmoba-backward-barrier-v0.png)

图中零线附近的默认模式曲线与补丁曲线存在重叠。右图是选中128行的梯度相对差，不是模型预测错误率。

## 同步开销

原版—补丁—补丁—原版的ABBA顺序，各自独立进程；同一份固定路由。每模式每进程预热50次，再做9批、每批50次反向，用CUDA事件计时。

| 模式 | 原版中位数 | 补丁中位数 | 相对变化 |
|---|---:|---:|---:|
| 默认 | {timings[0]['original']:.4f} ms | {timings[0]['barrier']:.4f} ms | {timings[0]['change_percent']:+.2f}% |
| 确定性 | {timings[1]['original']:.4f} ms | {timings[1]['barrier']:.4f} ms | {timings[1]['change_percent']:+.2f}% |

这组微基准没有显示新增同步的明显中位数开销；默认模式有波动，不能据此宣称补丁加速。它只计冻结单层图的eager反向，含启动间隙和分配，不含路由、完整模型和优化器。确定性模式自身在此测试中约为默认模式的6倍耗时；“新增同步开销很小”不代表开启确定性没有成本。

## 8K完整模型短训练

Qwen2.5-0.5B底座冻结，FP32主干、BF16注意力；24层Q/K/V/O投影各加rank8 LoRA，1,081,344个可训练参数。AdamW学习率1e-4，固定初始化、种子和数据，dropout=0，不使用梯度检查点。池化用FP32，注意力反向用同步补丁及确定性模式。

四条轨迹：池化BN32、BN32重复、BN64、BN128；每条4步，共16个优化更新。它们不是4个独立随机种子。WikiText训练集取4个8K窗口，验证集取2个8K窗口，未在测试集上训练。

结果：四条轨迹的首次路由、首次梯度和最终LoRA参数更新逐位一致；每条训练后的三种池化配置验证值也完全一致。首次路由对比覆盖24层，共2,752,512个query-head行。所有3对比较的梯度和更新相对L2差均为0。

平均验证NLL从3.86416降到3.62866。训练只有4步、验证只有2个窗口，且没有本轮原版或dense质量对照；只能说明这次整合能稳定学习，不能证明补丁改善质量或泛化。

完整诊断脚本用时44.74秒，16个更新已全部结束；峰值张量显存约32.19GB（约29.98GiB）。本轮编译用时约14.23分钟，环境修复和等待也占用了租用时间，不能用训练44.74秒代替总计费时长。实际账单未知。

## 补丁、失败和可复现性

原版固定commit：39d9ac043b271d046a2181a9991e99a26b67bca1；CUTLASS固定commit：a2439551c765c5393aebe557ee75d3a0412d2211。原版源码恢复缺失文件后保持未修改。

隔离副本只在load_row_indices_to_smem_bwd写共享索引之前新增一次__syncthreads()，保留写后同步。原版与候选分别在独立进程加载，并记录实际扩展路径和SHA。

- 原版扩展SHA：b114a7755aad6556bc72eac1f8de0fcd0d5bd1e78d0dc1b1e521ac8621ce4d53。
- 候选扩展SHA：72c3fda9e7d4da701bfc76b0e32f4d80fdbeec70d462c83b87b97f51d887ed1c。
- 候选wheel：exports/flashmoba-barrier-wheels-v0/，包内.so与实际测试.so的SHA一致，原版wheel保留。环境是Python3.11、torch2.8.0+cu128；不是任意新卡通用安装包。

本次迁移留下不完整venv和源码。恢复过程包含磁盘配额失败、源目录完整性失败、残缺egg-info导致的setup失败、以及主动调整编译并行数的中断；都保留在environment-v1至v5结果和日志中，不能算作训练失败。运行环境改到/opt容器盘，数据和结果仍在/workspace持久化盘。详见恢复说明和时间轴。

云端原始归档flashmoba-backward-cloud-stage-v0.tar.gz为87,607,372字节，SHA256 ac1481b0378dfa135c8a4d98e08c68979caae5a363bbd23c11da3f88a8928096。已经下载并逐一核验149个清单文件；150个归档成员包含清单自身。报告/图/最终状态另存收尾overlay并同步云端。

## 对研究方向意味着什么

现在有一条经过长序列验证、可重复的训练基线，可以继续做稀疏训练的质量和成本对照。同步修复是基础设施修复；通用FP32路由和数值一致性已有相关工作，不能把本轮直接包装成新的机器学习方法或已经足够投稿的论文。

下一步应单独冻结dense/原版默认模式/修复后稀疏模型的训练协议，在相同token预算下检验收敛与质量，在相同质量下比较端到端时间。先做有对照的小规模试验，再决定是否扩大；不要把确定性复现实验的耗时直接当作通常训练的效率。

最后GPU快照：{gpu['recorded_utc']}，2MiB、0%利用率、没有GPU计算进程。Pod没有停止，实际账单未知。
'''
(ROOT/'docs/flashmoba-backward-newpod-results-2026-09-15.md').write_text(report,encoding='utf-8')
events=[]
for p in sorted((ROOT/'results').glob('flashmoba-backward-environment-v*/events.jsonl')):
    for line in p.read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row['event'] in ['command_start','command_end','failed','complete']:
            events.append((row['utc'],p.parent.name,row['event'],row.get('label','')))
index=['# 本轮日志与时间轴','', '所有时间均为UTC；每步训练另保留train_nll、梯度范数、耗时。','',
       '| 阶段 | 开始 | 结束 | 结果 |','|---|---|---|---|']
for name,x in [('原版8K反向',old),('候选8K反向',new),('8K整合训练',train)]:
    index.append(f"| {name} | {x['started_utc']} | {x['finished_utc']} | {x['status']} |")
index += ['', '| 轨迹 | 开始 | 结束 | 更新数 |','|---|---|---|---:|']
for x in train['trajectories']:
    index.append(f"| {x['name']} | {x['started_utc']} | {x['finished_utc']} | {x['updates']} |")
index += ['', '完整日志目录：results/flashmoba-backward-environment-v1至v5、flashmoba-backward-stage-controller-v0、flashmoba-long-backward-original-v0、flashmoba-long-backward-barrier-v0、flashmoba-newpod-standard-original-v0、flashmoba-newpod-standard-barrier-v0、flashmoba-backward-timing-*、flashmoba-backward-training-v0。', '', '## 环境事件', '']
index += [f'- {t}：{folder} / {event} {label}' for t,folder,event,label in sorted(events)]
(ROOT/'docs/flashmoba-backward-run-index-2026-09-15.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
state=f'''# 当前状态：8K同步补丁通过，16步完整模型整合通过

更新 {now}。报告docs/flashmoba-backward-newpod-results-2026-09-15.md；日志时间轴docs/flashmoba-backward-run-index-2026-09-15.md。

新Pod 4rvqz1eqrmdl0w，直连195.26.233.77:27747，RTX 6000 Ada。有效环境/opt/native-sparse-flashmoba-env-v2；迁移旧venv不再是已验证环境。

原版确定性8K反向重复Q梯度相对L2差2.18%–6.92%；隔离共享索引同步补丁后16次Q/K/V逐位一致，128个有效参考位置相对FP64差约2.58%，原版/补丁各12项标准检查通过。实际K4全体有2行与CPU重建不同，128行参考全部匹配，不能声称全体路由一致。

4条4步8K LoRA轨迹，共16优化更新；同配置与FP32池化跨配置的首次路由、首次梯度和最终参数更新完全一致。平均验证NLL3.86416→3.62866，仅2个验证窗口、无本轮质量基线，不能作为质量提升或论文结论。

候选wheel已保存，包内扩展与测试SHA一致。云端原始包87.6MB已下载本机，149个清单文件逐一核验通过。收尾报告/状态/图另有overlay。迁移恢复失败与编译中断日志均保留。

当前0训练作业；{gpu['recorded_utc']} GPU快照2MiB、0%，Pod仍在运行，实际账单未知。不能宣称已经停卡。

下一步：冻结有dense/原版默认/修复后稀疏对照的质量与端到端成本试验；先小试再扩大。同步修复只恢复可信基线；已确认原创ML贡献仍为0，不把工程修复包装成已足够投稿。
'''
(ROOT/'STATE.md').write_text(state,encoding='utf-8')
control=read('logs/control-state.json')
control.update(updated_utc=now,status='backward_barrier_kernel_and_8k_training_integration_passed',
    current_cloud_connection_available=True,active_training_jobs=0,current_pod_billing_state='running_actual_invoice_unknown',
    cloud_billing_status='running_actual_invoice_unknown',long_context_backward_gate_passed=True,
    current_decision_report='docs/flashmoba-backward-newpod-results-2026-09-15.md',
    run_index='docs/flashmoba-backward-run-index-2026-09-15.md',
    next_action='Predefine dense/original-default/patched-sparse quality and end-to-end cost controls before further scale-up.',
    current_turn_gpu_jobs_started=9,current_turn_optimizer_updates=16,
    backward_candidate_stage=dict(kernel_gate_passed=True,integration_repeat_passed=True,trajectories=4,optimizer_updates=16,
        confirmed_original_ml_contributions=0,candidate_wheel='exports/flashmoba-barrier-wheels-v0',raw_archive_local_sha256_verified=True,
        confirmed_gpu_idle_at=gpu['recorded_utc'],pod_stopped=False,reference_sample_rows=128,all_k4_reconstruction_differing_rows=2))
(ROOT/'logs/control-state.json').write_text(json.dumps(control,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as f:
    f.write('''
- 16:00:01–16:00:04 UTC：新卡原版8K反向复现；确定性Q差2.18%–6.92%，128参考掩码全部匹配。
- 16:08:27–16:22:41 UTC：12并行作业完成隔离同步补丁编译，原版源码未改。
- 16:22:49–16:22:50 UTC：候选确定性16次Q/K/V逐位一致，参考门槛通过；随后12项标准检查通过。
- 16:23:04–16:23:23 UTC：ABBA冻结图反向测速，同模式中位数没有明显新增同步开销。
- 16:23:27–16:24:12 UTC：4条4步8K LoRA整合，共16更新；首次路由、首次梯度和最终参数更新全部逐位一致。
- 16:25:56 UTC：候选wheel与原始归档完成；16:26:53 UTC本机149个清单文件SHA核验通过。
- 16:27:50 UTC：GPU空闲、无GPU计算进程，Pod未停止。详细时间轴见docs/flashmoba-backward-run-index-2026-09-15.md。
''')
paths=[ROOT/p for p in ['STATE.md','TIMELINE.md','logs/control-state.json',
    'docs/flashmoba-backward-newpod-results-2026-09-15.md','docs/flashmoba-backward-run-index-2026-09-15.md',
    'docs/flashmoba-backward-cloud-restore-2026-09-15.md','docs/figures/flashmoba-backward-barrier-v0.png','docs/figures/flashmoba-backward-barrier-v0.pdf',
    'scripts/finalize_backward_candidate_stage.py','scripts/plot_backward_barrier_results.py',
    'provenance/flashmoba-backward-cloud-local-verification-v0.json','provenance/flashmoba-backward-final-gpu-v0.json',
    'provenance/flashmoba-newpod-4rvqz1-v0.json','results/flashmoba-backward-package-v0/timing-summary-local.json',
    'results/flashmoba-long-backward-original-v0/k4-full-mask-differences-local.json']]
rows=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
mf=ROOT/'provenance/flashmoba-backward-final-overlay-v0.json';mf.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
archive=ROOT/'exports/flashmoba-backward-final-overlay-v0.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
    for p in [*paths,mf]:t.add(p,arcname=p.relative_to(ROOT).as_posix())
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=len(paths)+1)))

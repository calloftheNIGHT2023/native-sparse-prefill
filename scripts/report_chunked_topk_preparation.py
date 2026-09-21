"""Summarize completed local evidence without modifying historical raw results."""
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    out=ROOT/'results/chunked-topk-preparation-audit-v0';out.mkdir(parents=True,exist_ok=False)
    p32=read('results/chunked-topk-checkpoint-replay-v1/verification.json')
    p64=read('results/chunked-topk-checkpoint-replay-fp64-v0/verification.json')
    fullgrad=read('results/chunked-topk-fullmodel-gradients-v0/verification.json')
    cuda=read('results/chunked-topk-local-cuda-preflight-v0/preflight.json')
    cpu=read('results/chunked-topk-cpu-benchmark-v1/benchmark.json')
    names=['chunked-topk-local-cuda-fp32-v0','chunked-topk-local-cuda-bf16-q256-v0',
           'chunked-topk-local-cuda-bf16-q1024-v0','chunked-topk-local-cuda-b16-q256-v0']
    gpu={name:read('results/'+name+'/benchmark.json') for name in names}
    assert p32['status']=='predictions_match_but_logits_differ' and p32['all_predictions_match']
    assert p64['status']==fullgrad['status']==cuda['status']=='passed'
    assert p32['total_answers']==p64['total_answers']==67584
    assert cpu['status']=='complete' and all(d['status']=='complete' for d in gpu.values())
    assert all(not d['scientific_optimizer_updates'] for d in [cpu,*gpu.values()])
    assert all(not d['optimizer_updates'] for d in [p32,p64,fullgrad,cuda])
    archive=ROOT/'exports/router-author-falsification-final-v0.tar.gz'
    assert sha(archive)=='f156526d396102236e2a48613d72b3bc02421e85b554ef27712cf14e46dea25f'
    references=read('results/chunked-topk-checkpoint-replay-v1/frozen-inputs.json')['files']
    protected={p:h for p,h in references.items() if p.startswith('results')}
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    table=[]
    for name,d in gpu.items():
        for row in d['rows']:
            if row['status']=='passed':
                table.append(dict(run=name,method=row['method'],length=row['length'],batch=row['batch'],
                    dtype=row['dtype'],operation=row['operation'],median_ms=row['median_seconds']*1000,
                    peak_increment_mib=row['cuda_peak_increment_bytes']/2**20))
    saved={r['method']:r['autograd_storage']['unique_saved_storage_bytes']
           for r in cpu['rows'] if r['length']==1024 and r['operation']=='attention_forward_backward' and r['status']=='passed'}
    timestamp=now()
    audit=dict(utc=timestamp,status='local_engineering_stage_complete',scientific_optimizer_updates=0,
        new_cloud_resources=0,new_cloud_rental_spend_usd=0,historical_cloud_bill_verified=False,
        distinct_replayed_answers=67584,precision_replays=['float32','float64'],
        fp32_all_answer_predictions_equal=True,fp32_all_logits_close=False,fp64_all_logits_close=True,
        cpu_saved_storage_bytes_at_1024=saved,gpu_valid_timing_conditions=len(table),
        gpu_unsupported_flash_conditions=sum(r['method']=='dense_sdpa_flash' and r['status']!='passed' for d in gpu.values() for r in d['rows']),
        gpu_timing_rows=table,old_final_archive_sha256=sha(archive),old_scientific_files_verified=protected,
        confirmed_new_original_contributions=0,active_project_training_jobs=0,
        source_sha256={p:sha(ROOT/p) for p in ['src/chunked_topk_attention.py','scripts/benchmark_chunked_topk.py',
            'scripts/preflight_chunked_topk_cuda.py','scripts/verify_chunked_topk_checkpoints.py',
            'scripts/verify_chunked_topk_fullmodel_gradients.py','tests/test_chunked_topk_attention.py']})
    save(out/'audit.json',audit)
    title='# 同等效果下的效率验证：本地阶段结果\n\n'
    text=title+f'记录于 {timestamp}。本轮新增云租卡支出0美元、科学训练更新0；未核对旧 Pod 是否仍在计费。\n\n'
    text+='## 当前结论\n\n实现迁移后，两个冻结模型的67,584个答案全部不变；在本机RTX 5070完成CUDA算子梯度验证和四组效率短测。此分块PyTorch版本相对旧掩码代码节省部分存储，但已测配置里未表现出优于优化完整注意力的实用效率组合。目前不需要为它扩大训练或租更贵的卡。\n\n'
    text+='这不要求准确率超越最新方法，也不否定全部稀疏训练方向。被否定的旧解释仍是“这个任务必须新增机制才能学会”。同等效果下的成本贡献依然可以研究，但此已知分块工程基线尚不构成新贡献。\n\n'
    text+='## 实现与验证\n\n新增查询分块精确选点、紧凑索引、选中边汇总及重计算反向；候选评分仍为二次复杂度。8项CPU单元测试通过，覆盖有限差分、Q/K/V梯度、因果/局部边、重复索引累加和固定dropout；两个完整模型的FP64参数梯度最大误差分别为 '
    text+=', '.join(f"{c['maximum_absolute_gradient_error']:.3g}" for c in fullgrad['checks'])+'。均未做优化器更新。\n\n'
    text+=f"GPU preflight通过FP64、FP32和固定选点的FP16/BF16梯度检查；该检查峰值分配 {cuda['cuda_peak_allocated_bytes']/2**20:.2f} MiB。半精度检查不能代表整模型半精度质量。\n\n"
    text+='### 必须保留的数值失败\n\nFP32严格logit容差在一个噪声条件未通过，最大差1.43；预测答案仍全部一致。定位到第一层约1e-6量级误差在第二层近并列选点处被放大，第6/7名远程候选差约1.19e-6。两实现同时使用FP64后，67,584个答案一致，最大logit差约9.77e-14。原FP32失败未被放宽容差改成成功；不能承诺逐步训练轨迹完全一致。详细原始结果见各replay目录和数值附录。\n\n'
    text+='## CPU存储与GPU实测不能混为一谈\n\n'
    text+=f"长度1024时，反向保存的独立张量存储：旧掩码 {saved['legacy_masked_topk']/2**20:.2f} MiB，分块 {saved['chunked_topk']/2**20:.2f} MiB，CPU SDPA {saved['dense_sdpa_auto']/2**20:.2f} MiB。这不是GPU峰值显存。\n\n"
    text+='下面为本机FP32、batch=1、head=1、dim=128、长度4096、query_chunk=64。时间取7次中位数，显存为计时阶段相对基线的峰值增量：\n\n| 实现 | prefill核心 ms | 前向+反向 ms | prefill显存增量 MiB | 前向+反向显存增量 MiB |\n|---|---:|---:|---:|---:|\n'
    d=gpu[names[0]]
    for method,label in [('legacy_masked_topk','旧完整矩阵掩码top-k'),('chunked_topk','分块top-k'),('dense_sdpa_auto','优化完整注意力SDPA')]:
        rows={r['operation']:r for r in d['rows'] if r['length']==4096 and r['method']==method and r['status']=='passed'}
        f,b=rows['prefill_core'],rows['attention_forward_backward']
        text+=f"| {label} | {f['median_seconds']*1000:.3f} | {b['median_seconds']*1000:.3f} | {f['cuda_peak_increment_bytes']/2**20:.3f} | {b['cuda_peak_increment_bytes']/2**20:.3f} |\n"
    text+='\n另做了BF16分块256/1024，以及FP32 batch16分块256来排查过小分块和小批量开销；完整逐次计时均保存，不能只挑对稀疏有利的一项。BF16长度4096从分块256改成1024，分块prefill从15.277ms降至5.846ms，但对应SDPA仍约0.593ms；峰值增量从6.42升至21.94MiB，对应SDPA约1MiB。优化分块存在时间与存储取舍，未形成最终优势。\n\n'
    text+='## 性能结论的限制\n\n- 所有稀疏主计时包括选点，已缓存索引的组件耗时另列，不能替换总耗时。\n- CUDA后端为实际观测的memory-efficient attention；当前Windows PyTorch没有编译FlashAttention，强制Flash的18项条件均明确记录失败。没有伪造FlashAttention成绩。\n- 本机并非独占GPU，运行前其他负载利用率约21%–35%；只有短测、固定执行顺序，数字用于工程排障，尚不能作为论文性能表，也不外推A40。\n- 这些是注意力核心测量，不含整个模型、优化器、数据管道和完整prompt预填充。没有实测完整训练加速。\n\n'
    text+='## 已知相关工作与下一步\n\n[Gupta等2021](https://aclanthology.org/2021.sustainlp-1.5/) §2.2与局限部分已经讨论精确top-k分块、重计算及二次打分成本。本轮把其已有思路适配到当前因果规则，没有原创性声明。[SAS 2026](https://arxiv.org/abs/2609.13141)本轮仅核查摘要，端到端排序方向仍须全文排查。\n\n下一步优先在本机处理选点和分散GPU调用的成本，并先明确相对既有方法的具体贡献。暂不为当前工程版本租卡或扩大科学训练。运行命令和限制见 `docs/chunked-topk-runbook-2026-09-15.md`。\n'
    (ROOT/'docs/chunked-topk-preparation-results-2026-09-15.md').write_text(text,encoding='utf-8')
    state='''# 当前状态：同等效果下的效率验证，本地阶段完成

更新 {utc}。活动科学训练0。本轮云租卡新增支出0美元，旧账单与Pod状态仍未核对。总预算上限仍为500美元，不等于尚余500美元。

用户明确接受准确率相近、但有其他实用价值的贡献。之前只撤回“当前MQAR差距必须用新机制修复”的解释；不再把它写成整个原生稀疏方向结束。已知小模型精确top8通过普通学习率调整，两个种子达到新题99.9939%。真实文本、Qwen4、原创方法和整体训练效率仍未证明。

本轮：分块精确top-k、紧凑索引和一阶反向已实现。两个冻结模型67584个答案保持一致，FP64输出及完整模型梯度检查通过；FP32有一个近并列选点导致logit差异的条件，保留失败记录，不声称逐步训练轨迹等价。

本机发现可用CUDA环境：C:/Users/callofthenight/AppData/Local/Programs/Python/Python314/python.exe，torch2.10.0+cu128，RTX5070。无需新装环境，未使用云机。4组本地GPU基准/54个有效计时条件完成；18个强制Flash条件不支持，实际完整注意力对照为CUDA memory-efficient SDPA。此工程版本目前不比该完整对照更划算，不启动扩大训练。硬件非独占，短测不能直接用作论文性能表。

当前报告 docs/chunked-topk-preparation-results-2026-09-15.md；审计 results/chunked-topk-preparation-audit-v0/audit.json；运行说明 docs/chunked-topk-runbook-2026-09-15.md。下一步先处理选点及GPU调用开销，并精确筛查贡献重合，暂不租卡。

旧科学结果和最终归档SHA未变；旧报告 docs/router-author-falsification-results-2026-09-14.md 保留为历史记录。旧348文件归档中的STATE和控制文件是当时快照，不能用旧快照哈希要求当前状态文档不更新。
'''.format(utc=timestamp)
    (ROOT/'STATE.md').write_text(state,encoding='utf-8')
    control=read('logs/control-state.json')
    control.update(updated_utc=timestamp,event='user_requested_continue_same_quality_efficiency_preparation',
        status='sparse_efficiency_local_gpu_baseline_complete',active_training_jobs=0,
        current_decision_report='docs/chunked-topk-preparation-results-2026-09-15.md',
        audit_report='results/chunked-topk-preparation-audit-v0/audit.json',
        next_action='本地降低候选打分和GPU调用成本，先筛具体贡献；当前工程版本不扩大训练、不新增租卡。',
        confirmed_original_contributions=0)
    control['chunked_topk_efficiency_v0']=dict(status='complete_local_diagnostic',report=control['current_decision_report'],
        optimizer_updates=0,cloud_spend_this_stage_usd=0,old_bill_verified=False,
        predictions_replayed=67584,fp32_logits_allclose=False,fp64_logits_allclose=True,
        gpu='local RTX5070',cuda_benchmark_runs=4,valid_gpu_timing_conditions=len(table),
        original_method_claim=False,full_model_speedup_proven=False,local_gpu_environment_available=True)
    save(ROOT/'logs/control-state.json',control)
    timeline=[]
    for label,d in [('FP32冻结检查点重放',p32),('FP64冻结检查点重放',p64),('完整模型参数梯度核对',fullgrad),('本机CUDA正确性',cuda),('CPU基准v1',cpu),*gpu.items()]:
        timeline.append((d['started_utc'],f"- {d['started_utc']} → {d['finished_utc']}：{label}，状态 {d['status']}；0优化器更新。"))
    with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 同等效果下效率验证（后补起止记录，按UTC排序）\n\n')
        f.write('- 2026-09-15T00:09:43.773457+00:00：首次FP32迁移检查因噪声logit容差失败而结束，已保留原始记录；后续完整诊断没有提高原容差。\n')
        f.write('- 2026-09-15T00:11:57.937489+00:00：CPU基准v0因组件函数调用错误提前结束；修复后另存v1，原日志保留。\n')
        f.write('\n'.join(x[1] for x in sorted(timeline))+'\n')
        f.write(f'- {timestamp}：更新方向解释和当前状态；已核对旧最终归档SHA及原科学检查点，保留67584答案一致、FP32数值差异和GPU效率负结果。新云支出0，未终止或修改远程Pod。\n')
    print(json.dumps(dict(report='docs/chunked-topk-preparation-results-2026-09-15.md',audit=str(out/'audit.json'),gpu_valid_conditions=len(table))))

if __name__=='__main__':main()

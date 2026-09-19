import hashlib,json,statistics
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj): p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def main():
    runs=[]; audit=[]
    for name in ['cloud-dense-context-v0','cloud-dense-context-10m-v0']:
        out=ROOT/'results'/name
        if not (out/'result.json').exists(): continue
        result=json.loads((out/'result.json').read_text())
        manifest=json.loads((out/'manifest.json').read_text())
        for rec in manifest: assert sha(out/rec['path'])==rec['sha256'],rec['path']
        events=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
        updates=[x for x in events if x['event']=='optimizer_step']
        assert [x['step'] for x in updates]==list(range(1,result['updates']+1))
        assert all(np.isfinite(x['lm_loss']) and x['gradient_norm']>0 for x in updates)
        assert all(a['utc']<=b['utc'] for a,b in zip(events,events[1:]))
        cfg=json.loads((out/'frozen-config.json').read_text())
        assert result['tokens']==result['updates']*cfg['sequence_length']
        assert result['initial_hash']!=result['final_hash']
        distributions={}
        for split,controls in [('development',result['development'][-1]['controls']),('test',result['test'])]:
            a=np.array([x['late_nll'] for x in controls['dense']['per_article']])
            b=np.array([x['late_nll'] for x in controls['sink_recent']['per_article']])
            g=np.random.default_rng(14403); delta=b-a
            samples=delta[g.integers(0,len(a),(10000,len(a)))].mean(1)
            distributions[split]=dict(articles=len(a),mean_gap_nats=float(delta.mean()),
                conditional_article_bootstrap_95ci=np.quantile(samples,[.025,.975]).tolist(),
                scope='Conditional on this single trained checkpoint; not training seed uncertainty')
        result['name']=name; result['started_utc']=events[0]['utc']; result['bootstrap']=distributions
        result['peak_cuda_gib']=max(x['peak_cuda_bytes'] for x in updates)/1024**3
        runs.append(result); audit.append(dict(name=name,verified_files=len(manifest),updates=len(updates),
            tokens=result['tokens'],backbone_changed=True,utc_ordered=True,checkpoint_sha256=sha(out/'checkpoint.pt')))
    assert runs
    folder=ROOT/'results/cloud-control-figures'; folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,len(runs),figsize=(6*len(runs),4),squeeze=False)
    for ax,r in zip(axes[0],runs):
        for mode,label in [('dense','Full context'),('sink_recent','Sink + recent'),('self_only','Current token only')]:
            ax.plot([x['step']*2048/1e6 for x in r['development']],
                [x['controls'][mode]['late_nll'] for x in r['development']],marker='o',markersize=3,label=label)
        ax.set(xlabel='Training tokens (millions; repeated corpus)',ylabel='Development late-token NLL',
            title=f"{r['tokens']/1e6:.2f}M-token dense control")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(folder/'learning-curves.png',dpi=180); plt.close(fig)
    now=datetime.now(timezone.utc).isoformat()
    save(ROOT/'logs/cloud-controls-audit.json',dict(updated_utc=now,runs=audit,invoice_spend_verified=False))
    save(ROOT/'results/cloud-controls-summary.json',dict(updated_utc=now,runs=runs))
    lines=['# 首轮 A40 云训练结果','', '已完成环境接入、2K 显存短测、模型真实训练和本地校验。以下是稠密模型的上下文使用控制实验，不是新稀疏方法的胜出结果。','',
        '| 训练量 | 开发集：完整上下文 | 开发集：开头+附近 | 测试集：完整上下文 | 测试集：开头+附近 |',
        '|---|---:|---:|---:|---:|']
    for r in runs:
        d=r['development'][-1]['controls']; t=r['test']
        lines.append(f"| {r['tokens']:,} | {d['dense']['late_nll']:.5f} | {d['sink_recent']['late_nll']:.5f} | {t['dense']['late_nll']:.5f} | {t['sink_recent']['late_nll']:.5f} |")
    lines.extend(['','NLL 越低越好。完整上下文对比开头及附近共64个完整块（含开头块，共256 tokens），另外保留未满块的尾部。该限制用于判断当前任务是否对全局路由敏感。对照是在同一个稠密模型权重上换注意力支持集，不是两种训练方法之间的比较。',
        '', '## 实际运行与记录','', '| 运行 | 开始 UTC | 结束 UTC | 更新 | 纯训练秒数 | 显存峰值 GiB |','|---|---|---|---:|---:|---:|'])
    for r in runs:
        lines.append(f"| {r['name']} | {r['started_utc']} | {r['finished_utc']} | {r['updates']} | {r['training_seconds']:.2f} | {r['peak_cuda_gib']:.2f} |")
    lines.extend(['','每次运行均保存原始逐步日志、训练顺序、固定配置、代码快照、模型与优化器、CPU/CUDA随机状态、评测逐文章数据和SHA256清单。远端路径为 `/workspace/native-sparse-prefill/results/`，同名目录已回传至本项目 `results/` 并逐文件核对。','',
        '## 数据与结论范围','',
        '- 70,426,624参数，随机初始化，纯GPTNeoX，全部6层更新；尚未验证Qwen混合Gated DeltaNet架构。',
        '- 每段2048输入tokens，384段训练窗口来自230篇文章（单遍786,432个预测tokens）、16篇开发文章、13篇测试文章。排除先前416段落所在整篇文章；原始WikiText划分不变。10M阶段反复经过该小语料，不能称为10M独立新tokens。',
        '- 1M与10M为相同初始化种子的两次独立时长配置，学习率衰减不同，10M从头重跑，不能当作两个独立随机种子或完全相同的前1M轨迹。',
        '- 两阶段读取同一留出集，10M结果不是新测试集上的独立确认；若形成候选方法，需另设最终确认集。',
        '- 预先固定门槛：完整上下文在开发与测试NLL均比开头+附近好至少0.02。文章bootstrap只描述该模型条件下文章抽样的不确定性，不代表跨训练种子稳定性。',
        '- 附近注意力可以通过多层间接传播远处信息；对照无差异不能证明模型不使用历史，或稀疏训练必然无损。','', '## 判读',''])
    for r in runs:
        gate=r['fixed_context_gate']; d=r['development'][-1]['controls']; t=r['test']
        lines.append(f"- {r['name']}：开发差距{gate['dev_gap_nats']:.6f}，测试差距{gate['test_gap_nats']:.6f} nats，门槛{'通过' if gate['passed'] else '未通过'}；只看当前token的测试NLL为{t['self_only']['late_nll']:.5f}。")
    lines.extend(['','## 环境与费用','',
        'A40显存46068MiB，Python3.11.10、PyTorch2.4.1+cu124、Transformers4.57.6。23项CPU测试和4项CUDA测试通过；256、512、1024、2048各完成4个合成数据技术更新，2K gather参考峰值15.14GiB，后三步约0.249秒。它会复制KV，不能作为高效稀疏内核加速证据。',
        '', '镜像原有PyGObject缺少PyCairo依赖，pip check原样记录；该GUI依赖不参与训练，核心训练检查通过。未修改用户镜像中的其他软件。',
        '', '首轮30美元、总500美元上限保持。实际账单与实际租用时价无法通过当前SSH核实；GPU进程限时不等于停止Pod计费。当前无Pod API凭证，内置浏览器也未登录，备份结束后需从RunPod控制台点Stop，不能把退出训练或SSH误记为停卡。'])
    (ROOT/'docs/cloud-first-results-2026-09-14.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(runs=len(runs),audit=audit)))
if __name__=='__main__': main()

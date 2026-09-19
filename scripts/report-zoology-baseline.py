"""Summarize the upstream minimal run after it really finishes."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
    folder=ROOT/'results/zoology-basic-cpu-v1';r=json.loads((folder/'result.json').read_text(encoding='utf-8'));assert r['status']=='complete'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for f in manifest:assert sha(folder/f['path'])==f['sha256'],f['path']
    e=[json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()];s=[x for x in e if x['event']=='optimizer_step'];v=[x for x in e if x['event']=='validation']
    assert [x['step'] for x in s]==list(range(1,r['updates']+1))
    assert len(v)==r['epochs'] and r['updates']==313*r['epochs']
    assert all(a['utc']<=b['utc'] for a,b in zip(e,e[1:]));assert all(np.isfinite(x['loss']) and np.isfinite(x['grad_norm']) and x['grad_norm']>0 for x in s)
    assert all(x['step']==313*(x['epoch']+1) for x in v)
    assert not any(x['valid/accuracy']>.99 for x in v[:-1])
    assert r['epochs']==100 or v[-1]['valid/accuracy']>.99
    assert r['supervised_answers']==40000*r['epochs'] and r['examples_seen']==10000*r['epochs']
    for x in s:
        expected=.001*(1+np.cos(np.pi*x['epoch']/100))/2
        assert np.isclose(x['learning_rate'],expected,rtol=1e-12,atol=1e-14)
    instrument=json.loads((ROOT/'results/zoology-instrumentation-check-v0/result.json').read_text())
    assert instrument['upstream_and_logged_weights_bitwise_equal'] and instrument['torch_rng_after_step_equal']
    scores=np.asarray(r['fresh_evaluation']['sequence_scores']);rng=np.random.default_rng(2026091426)
    means=scores[rng.integers(0,len(scores),(4000,len(scores)))].mean(1);ci=[float(np.quantile(means,.025)),float(np.quantile(means,.975))]
    now=datetime.now(timezone.utc).isoformat();audit=dict(utc=now,run=folder.name,files_sha_verified=len(manifest),ordered_steps=len(s),epochs=r['epochs'],
        stopping_rule_verified=True,epoch_learning_rate_schedule_verified=True,logging_one_step_parity=instrument,fresh_accuracy=r['fresh_evaluation']['accuracy'],fresh_sequence_bootstrap_95=ci)
    save(ROOT/'logs/zoology-baseline-final-audit.json',audit)
    figures=ROOT/'results/zoology-basic-figures';figures.mkdir(exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([x['epoch']+1 for x in v],[100*x['valid/accuracy'] for x in v],marker='o',markersize=3)
    axes[0].axhline(99,ls='--',color='grey',label='Upstream >99% stopping threshold');axes[0].set(xlabel='Epoch',ylabel='Validation accuracy (%)',ylim=(0,102));axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    train=[np.mean([x['loss'] for x in s if x['epoch']==i]) for i in range(r['epochs'])]
    axes[1].plot(range(1,r['epochs']+1),train,label='Training batch mean');axes[1].plot(range(1,r['epochs']+1),[x['valid/loss'] for x in v],label='Validation batch mean');axes[1].set(xlabel='Epoch',ylabel='Cross-entropy loss');axes[1].legend();axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(figures/'learning.png',dpi=170);plt.close(fig)
    passed=r['upstream_stop_threshold_reached'] and r['independent_fresh_at_least_99']
    title='公开最小例子已在CPU跑通' if passed else '公开最小例子执行完成，但未达到成功门槛'
    lines=['# Zoology基线：'+title,'',
        f"固定提交1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb的官方basic例子，实际完成{r['epochs']}轮、{r['updates']:,}次更新，最终验证正确率{v[-1]['valid/accuracy']:.2%}，独立新题正确率{r['fresh_evaluation']['accuracy']:.2%}。本轮没有GPU作业。",'',
        '## 做了什么','',
        '直接使用作者的数据生成、LanguageModel和Trainer.fit/train_epoch/test，模型不是之前的GPTNeoX替代品。保留原优化器、每轮余弦调度、数据顺序及超过99%验证准确率的提前停止规则。最小例子为词表256、长度64、4组键值对、训练10000序列、原验证1000序列、batch32、seed123。',
        '', f"实际模型{r['parameters']:,}参数，128宽、2层、1头、绝对位置嵌入，输入输出权重共享。AdamW学习率1e-3、weight decay0.1、最多100epochs，无额外warmup或梯度裁剪。模型初始化先于随机填充数据生成，保持上游调用顺序。",'',
        '这轮同时恢复了数据、模型与优化的完整配置，不能据此将与旧实验的差异归因于某一个因素。属于固定版本公开最小例子的Windows CPU验证，不是原论文完整结果复现、新方法或Qwen实现。','',
        '## 平台适配与失败记录','',
        '- TokenEmbeddings默认分配设备cuda改为cpu。',
        '- NumPy验证种子范围超过Windows默认32位C-long，显式使用np.int64；v0在数据准备阶段因此失败，更新数0，日志和原源码快照保留。',
        '- 早期源文件复制因Windows换行转换造成SHA不同，源一致性测试拦截后按原始字节恢复；该问题发生在训练前，失败测试日志保留。',
        '- Pydantic默认序列化会省略MQAR子类字段，运行时对象仍含正确参数。config.json保留，另存resolved-config.json完整配置和说明；实际每序列4个标签已从保存数据核验。',
        '- 依赖为本机版本，见requirements-zoology-cpu-lock.txt。不能声称与作者GPU随机初始化和数值逐位一致。','',
        '36项正确性检查通过，原训练器与日志包装在一个相同CPU训练批次上的最终参数和随机状态逐位一致。该技术检查各做一次更新，共2次，未计入科研训练量。','',
        '## 独立检查与证据边界','',
        f"原代码名为test的1000序列参与每轮停止判断，因此这里称为验证集。训练结束后另用固定种子2026091425生成1000条新序列、4000个答案，模型不更新；准确率{r['fresh_evaluation']['accuracy']:.2%}，按序列bootstrap的95%区间为{ci[0]:.2%}—{ci[1]:.2%}。该区间不代表不同训练种子的波动。",'',
        '保存的训练、验证与新题输入逐行SHA没有交叉；标签均核对为前方四条记录的键值对应，查询位置不含答案。因随机填充允许符号重复，任务答案以生成器指定的前方记录为准。', '',
        '## 时间与日志','',
        f"- 开始UTC：{r['started_utc']}；结束UTC：{r['finished_utc']}。",
        f"- 更新{r['updates']:,}次；纯训练记录时间{r['training_seconds']:.2f}秒（含少量日志计时钩子开销）；输入tokens {r['input_tokens']:,}，受监督答案{r['supervised_answers']:,}。固定10000序列重复使用，不能称为这些数量的独立样本。",
        '- 原始数据、模型、优化器、调度器、随机状态、配置和代码：results/zoology-basic-cpu-v1/。checkpoint保存点位于轮末验证后，不是已经实现的恢复CLI。',
        '- 逐步UTC/损失/梯度/学习率：results/zoology-basic-cpu-v1/events.jsonl；控制台日志：logs/zoology-basic-cpu-v1.log。',
        '- 核验：logs/zoology-baseline-final-audit.json；曲线：results/zoology-basic-figures/learning.png。','',
        '## 来源与下一步','',
        '[官方最小配置](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/experiments/basic_examples/basic.py)、[原训练器](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/train.py)。所有源文件、原始SHA和平台补丁保存在third_party/zoology-1ad20d1/。']
    if passed:lines.extend(['','基础可学性现在有一个已通过的参照。下一步保留作者模型与优化流程，先只增加上下文长度，再检查远端信息隔离，逐项记录差异；尚不能据此租大卡或宣称全程稀疏方法有效。'])
    else:lines.extend(['','当前最小配置在这台机器的这一实现/种子下也未通过；不能把它称为成功复现。下一步应核对论文实验配置、版本与训练环境差异，并保留这次失败，不继续从这一未通过基线推断稀疏方法优劣。'])
    (ROOT/'docs/zoology-baseline-results-2026-09-14.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=passed,epochs=r['epochs'],updates=r['updates'],validation=v[-1]['valid/accuracy'],fresh=r['fresh_evaluation']['accuracy'],ci=ci)))

if __name__=='__main__':main()

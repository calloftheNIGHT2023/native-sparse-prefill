"""Verify every checkpoint evaluation and report the full empirical cost-quality curve."""
import hashlib,json,tarfile,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 archive=R/'exports/length-density-cost-evidence-v0.tar.gz';proof=read(archive.with_suffix('.json'))
 assert sha(archive)==proof['sha256']
 dest=R/'results/cloud-length-density-cost-evidence-v0';dest.mkdir(exist_ok=False)
 with tarfile.open(archive) as tar:
  members=tar.getmembers();names=[x.name for x in members];assert len(names)==len(set(names))
  manifest=json.load(tar.extractfile('amp-recovery-results-manifest.json'));expected={x['path']:x for x in manifest['files']}
  assert set(names)==set(expected)|{'amp-recovery-results-manifest.json'}
  for m in members:
   assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   p=(dest/m.name).resolve();assert dest.resolve() in p.parents;raw=tar.extractfile(m).read()
   if m.name in expected:
    e=expected[m.name];assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256']
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 verification=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),sha256=proof['sha256'],files=len(expected))
 dump(dest/'LOCAL-VERIFICATION.json',verification)
 pp=dest/'provenance/length-density-cost-protocol.json';protocol=read(pp);assert sha(pp)==sha(R/'provenance/length-density-cost-protocol.json')
 for name,h in protocol['sources'].items():assert sha(dest/name)==h
 stage=dest/'results/length-density-cost-stage-v0';control=read(stage/'result.json')
 assert control['status']=='complete' and control['returncode']==0 and control['optimizer_updates']==0
 d=read(stage/'benchmark/result.json');assert d['status']=='complete' and d['optimizer_updates']==0 and d['weights_unchanged']
 assert d['protocol_sha256']==sha(pp) and d['calibration_replay_max_abs_error']<=1e-6
 original=R/'results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k0-lr0.0003/checkpoint-256.pt'
 assert sha(original)==d['checkpoint_sha256']
 rows=d['rows'];assert len(rows)==12
 assert rows==[json.loads(s) for s in (stage/'benchmark/rows.jsonl').read_text().splitlines()]
 train=np.load(R/'data/flashmoba-amp-recovery-v0/train-calibration.npz')['train'];metrics=[]
 expected_order=[(rnd,length,k) for rnd in [0,1] for length in protocol['length_orders'][rnd] for k in protocol['orders'][rnd]]
 assert [(x['round'],x['length'],x['k']) for x in rows]==expected_order
 for x in rows:
  pair=protocol['window_pairs'][x['round']];assert x['window_pair']==pair
  full=np.concatenate([train[pair[0]],train[pair[1]][1:]])[:x['length']+1]
  assert hashlib.sha256(full.tobytes()).hexdigest()==x['input_sha256']
  assert len(x['times'])==len(x['losses'])==len(x['gradient_norms'])==3
  assert np.isfinite(x['times']+x['losses']+x['gradient_norms']).all() and min(x['times'])>0
  assert abs(np.median(x['times'])-x['median_seconds'])<1e-10
  assert x['peak_bytes']>0
 for length in [16384,32768]:
  for k in [0,16,32]:
   xs=[next(x for x in rows if x['length']==length and x['k']==k and x['round']==rnd) for rnd in [0,1]]
   base=[next(x for x in rows if x['length']==length and x['k']==0 and x['round']==rnd) for rnd in [0,1]]
   ratios=[x['median_seconds']/b['median_seconds'] for x,b in zip(xs,base)]
   if k:
    prior=next(x for x in d['candidates'] if x['length']==length and x['k']==k);assert prior['ratios']==ratios
   metrics.append(dict(length=length,k=k,seconds=[x['median_seconds'] for x in xs],ratios=ratios,time_saved_pct=[100*(1-v) for v in ratios],peak_gib=[x['peak_bytes']/1024**3 for x in xs]))
 out=R/'results/length-density-cost-audit-v0';out.mkdir(exist_ok=False)
 audit=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),archive=verification,optimizer_updates=0,measurements=36,warmups=24,seconds=control['seconds'],metrics=metrics)
 dump(out/'result.json',audit);(out/'source.py').write_bytes(Path(__file__).read_bytes())
 lines=['# 更长输入能否让更多保留块仍省计算？','','同一已有密集适配权重，在16K/32K输入下对比密集、K16、K32的前向加反向计算成本；两个输入、两轮反转顺序。0优化更新，不产生新质量成绩。','', '|长度|K（0=密集）|两轮中位秒数|两轮节时|两轮峰值GiB|','|---:|---:|---|---|---|']
 for x in metrics:
  lines.append(f"|{x['length']}|{x['k']}|{x['seconds'][0]:.3f} / {x['seconds'][1]:.3f}|{x['time_saved_pct'][0]:+.2f}% / {x['time_saved_pct'][1]:+.2f}%|{x['peak_gib'][0]:.2f} / {x['peak_gib'][1]:.2f}|")
 lines+=['','计入输入搬运、梯度清零、全模型前向、全部token损失反向及梯度裁剪；不含optimizer.step、评测、保存。因此是训练主要计算部分的成本，不等于完整训练节时、RunPod账单或推理速度。','32K输入拼接已有训练窗口，16K为嵌套前缀；人工边界只用于计时，不能据此说32K阅读能力已经验证。K是最多保留块数，不是统一稀疏百分比。',f"全部12格完成，共36次计时、24次热身；4个原校准窗口重放误差≤1e-6，断点哈希核对，权重摘要前后不变，全部损失/梯度范数有限。控制器用时{control['seconds']:.2f}秒。只有两个输入的描述性计时，无跨设备置信结论。",f"证据包SHA256：{proof['sha256']}，{len(expected)}文件逐项核验。冻结配置：provenance/length-density-cost-protocol.json；逐格UTC与输入哈希保存在原始rows.jsonl。",'']
 content='\n'.join(lines).replace('同一已有密集适配权重','## 直观结果与下一步\n\n32K时，K16两轮节时30.61%/31.25%，K32两轮节时16.59%/17.29%；K32峰值约40.01GiB，可在当前48GB卡运行。相同K32在16K反而慢8.29%–9.56%。这给下一轮训练提供了一个明确可行点：32K/K32，而不是要求16K/K32在无成本收益时继续硬做。\n\n这个结果支持“更长输入能让当前稀疏实现获得更大的训练计算节省”，并允许在同样32K长度下把保留块数从16提高到32而仍然省时。是否因此保住接近密集的质量，尚需32K密集/K32的适配实验；本轮没有测过这项质量，不能把旧16K的PPL代价直接挪到32K。\n\n后续应冻结32K的训练、校准与新评测切分；密集与K32使用相同token曝光和匹配的学习率搜索预算，先做短训练双种子筛查，保存完整optimizer/RNG/数据游标。比较固定token预算的质量与总训练时间，也画两边的早停曲线。目标是质量足够接近且更省，不要求超过密集；PPL相对差与任务百分点差分别报告。当前只完成成本测试，尚未启动该32K训练。\n\n'+'同一已有密集适配权重',1)
 (R/'docs/length-density-cost-results-2026-09-16.md').write_text(content,encoding='utf-8')
 print(json.dumps(audit))
if __name__=='__main__':main()

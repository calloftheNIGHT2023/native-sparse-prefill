"""Audit capability gates and task-level paired accuracy/prefill outcomes."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1]
E=R/'results/cloud-task-quality-evidence-v0';S=E/'results/task-quality-stage-v0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert read(E/'LOCAL-VERIFICATION.json')['status']=='verified'
 control=read(S/'result.json');assert control['status'] in ['complete','capability_gate_failed'] and control['optimizer_updates']==0
 p=read(E/'data/task-quality-v0/protocol.json');gate=read(S/'pilot-dense0/gate.json');pilot=read(S/'pilot-dense0/result.json')
 assert gate['protocol_sha256']==sha(E/'data/task-quality-v0/protocol.json') and gate['pilot_result_sha256']==sha(S/'pilot-dense0/result.json')
 for task in p['tasks']:
  xs=[x for x in pilot['predictions'] if x['task']==task];short=sum(x['correct'] for x in xs if x['variant']=='short');absent=sum(x['correct'] for x in xs if x['variant']=='no_context');rule=p['ability_gate'][task]
  assert len(xs)==64 and gate['decisions'][task]['short_correct']==short and gate['decisions'][task]['no_context_correct']==absent
  assert gate['decisions'][task]['passed']==(short>=rule['short_min_correct'] and short-absent>=rule['context_gain_min_correct'])
 lines=['# 实际任务正确率与prefill结果','','全部只评测已有四个断点，0个训练更新。RACE-M扩充上下文和自建检索诊断，不是官方RACE/LongBench/RULER成绩。','','## 能力检查（独立32题）','','|任务|有原文正确|无原文正确|进入正式比较|','|---|---:|---:|---|']
 for task,d in gate['decisions'].items():lines.append(f"|{task}|{d['short_correct']}/32|{d['no_context_correct']}/32|{'是' if d['passed'] else '否'}|")
 result=dict(status=control['status'],utc=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,gate=gate,groups=[],seconds=control['seconds'])
 if control['status']=='complete':
  models={name:read(S/name/'result.json') for name in ['dense0','dense1','sparse0','sparse1']};metadata=read(E/'data/task-quality-v0/formal.json');truth={(x['item_id'],x['variant']):x for x in metadata}
  for name,m in models.items():
   assert m['status']=='complete' and m['split']=='formal' and m['optimizer_updates']==0 and m['calibration_replay_max_abs_error']<=1e-6
   assert m['protocol_sha256']==gate['protocol_sha256'] and m['selected_tasks']==gate['selected_tasks']
   assert m['k']==(0 if name.startswith('dense') else 16) and m['seed']==2026091560+int(name[-1])
   assert len(m['predictions'])==len(gate['selected_tasks'])*96*4
   for x in m['predictions']:
    t=truth[x['item_id'],x['variant']];assert x['gold']==t['gold'] and x['length']==t['length']
    assert x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold']) and x['prefill_seconds']>0
  lines+=['','## 正式题：各96题，两种子','','准确率均为两个种子的平均；bootstrap按同一题共同抽样两个种子。时间比是逐题 sparse/dense prefill时间比的中位数，小于1表示更快。','','|任务|输入|dense正确率|K16正确率|差距(pp)|描述性95%区间(pp)|prefill时间比|5pp容忍范围证据|','|---|---|---:|---:|---:|---|---:|---|']
  for task in gate['selected_tasks']:
   for variant in p['formal_variants']:
    ds=[];ss=[];ratios=[];seedrows=[]
    for seed in [0,1]:
     d={x['item_id']:x for x in models[f'dense{seed}']['predictions'] if x['task']==task and x['variant']==variant};s={x['item_id']:x for x in models[f'sparse{seed}']['predictions'] if x['task']==task and x['variant']==variant};assert set(d)==set(s) and len(d)==96
     order=sorted(d);dv=np.array([d[k]['correct'] for k in order],dtype=float);sv=np.array([s[k]['correct'] for k in order],dtype=float);ds.append(dv);ss.append(sv);ratios.extend(s[k]['prefill_seconds']/d[k]['prefill_seconds'] for k in order)
     seedrows.append(dict(seed=2026091560+seed,dense_accuracy=float(dv.mean()),sparse_accuracy=float(sv.mean()),regressions=int(((dv==1)&(sv==0)).sum()),improvements=int(((dv==0)&(sv==1)).sum()),answer_changes=sum(d[k]['prediction']!=s[k]['prediction'] for k in order)))
    delta=(np.stack(ss)-np.stack(ds)).mean(0);rng=np.random.default_rng(2026091599);boot=delta[rng.integers(0,96,size=(10000,96))].mean(1);ci=np.quantile(boot,[.025,.975]);denseacc=float(np.mean(ds));sparseacc=float(np.mean(ss));support=ci[0]>-.05 and all(x['dense_accuracy']>.25 for x in seedrows)
    row=dict(task=task,variant=variant,dense_accuracy=denseacc,sparse_accuracy=sparseacc,gap_pp=100*float(delta.mean()),descriptive_interval_pp=(100*ci).tolist(),median_paired_prefill_time_ratio=float(np.median(ratios)),supports_5pp_tolerance_descriptively=bool(support),seed_rows=seedrows)
    result['groups'].append(row);lines.append(f"|{task}|{variant}|{100*denseacc:.2f}%|{100*sparseacc:.2f}%|{row['gap_pp']:+.2f}|[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]|{row['median_paired_prefill_time_ratio']:.3f}|{'支持（描述性）' if support else '证据不足'}|")
  lines+=['','无原文对照用于检查题目是否依赖上下文，不作为长上下文能力或高质量证明。5pp列是预先冻结的探索性规则；96题和两个共享训练顺序的种子仍不足以确认跨任务等价。单次连续测量prefill不等于可靠硬件benchmark，且不含decode；原训练节时约9.2%应独立报告，不能拿来代替本轮prefill收益。']
 else:lines+=['','两类任务均未通过预先冻结的能力门槛，未打开正式题进行模型比较。不能用两个模型都不会做证明效果相同。后续先改进有明确依据的评测形式或模型能力，再建立新的独立试题，不反复筛选本轮正式题。']
 lines+=['','## UTC时间轴','','|任务|开始|结束|退出码|','|---|---|---|---:|']
 for x in control['runs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|{x['returncode']}|")
 lines+=['',f"控制器耗时{control['seconds']/60:.2f}分钟，不含准备、传输、存储、备份和空闲；账单未核实。全部原始预测、四个选项分数、输入、标签、配置、脚本与时间轴已归档。"]
 out=R/'results/task-quality-audit-v0';out.mkdir(exist_ok=False);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes());(R/'docs/task-quality-results-2026-09-15.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()

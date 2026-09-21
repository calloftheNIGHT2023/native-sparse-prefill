"""Offline audit of the fixed-weight diagnostic receipts; no tensor execution."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
M='logs/babylm-dw-cost-profile-backup-20260921/20260921T234120.236686Z/manifest.json'
m=json.loads((ROOT/M).read_text(encoding='utf-8'))
for f in m['files']:
 b=(ROOT/f['local_path']).read_bytes()
 assert len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256']
def read(suffix):
 f=next(f for f in m['files'] if f['path'].endswith(suffix))
 return (ROOT/f['local_path']).read_bytes()
s=json.loads(read('/summary.json'));p=json.loads(read('/protocol.json'))
rows=[json.loads(x) for x in read('/events.jsonl').splitlines()]
assert s['status']=='complete_read_only_cost_diagnostic'
assert hashlib.sha256(read('/events.jsonl')).hexdigest()==s['events_sha256']
original_protocol=(ROOT/'configs/babylm-dw-cost-profile-20260921-v0.json').read_bytes()
assert hashlib.sha256(original_protocol).hexdigest()==s['protocol_sha256']
assert json.loads(original_protocol)==p  # Output copy has Linux newlines; original file SHA remains authoritative.
c=Counter(x['type'] for x in rows)
assert [c[t] for t in ['forward_attempt','forward_complete','backward_attempt','backward_complete','warmup_complete','paired_measurement_complete','hook_equivalence','arm_complete']]==[36,36,32,32,4,16,16,2]
assert rows[-1]['counts']==s['counts'] and s['counts']['optimizer_updates']==s['counts']['scientific_updates']==0
for name,key in [('submitted_input_tokens','input_tokens'),('submitted_loss_tokens','loss_tokens')]:
 assert s['counts'][name]==sum(x[key] for x in rows if x['type']=='forward_attempt')
assert m['outer_live']['state']=='exited' and m['project_lock']=='free'
eq=[x for x in rows if x['type']=='hook_equivalence']
gradient_checks=0
for x in eq:
 z=x['comparison'];assert z['passed'] and z['loss']['passed'] and z['loss']['max_tolerance_ratio']<=1
 for g in z['gradients'].values():
  assert g['passed'] and (g.get('both_none') or g['max_tolerance_ratio']<=1)
  gradient_checks+=1
data={}
for arm,a in s['arms'].items():
 assert a['weights_unchanged'] and a['checkpoint_file_unchanged'] and a['model_state_before_sha256']==a['model_state_after_sha256']
 assert a['checkpoint_sha256']==p['arms'][arm]['checkpoint_sha256']
 assert a['checkpoint_protocol_sha256']==p['arms'][arm]['checkpoint_protocol_sha256']
 assert [r['window_index'] for r in a['rows']]==p['window_indices']
 denom=math.fsum(r['instrumented']['forward_stream_ms'] for r in a['rows'])
 sums=Counter()
 for r in a['rows']:
  assert r['equivalence_passed']
  actual=Counter()
  for part in r['instrumented']['forward_components']:actual[part['group']]+=part['stream_elapsed_ms']
  assert len(r['instrumented']['forward_components'])==50 and dict(actual)==r['forward_group_stream_ms']
  assert sum(actual.values())<=r['instrumented']['forward_stream_ms']+0.01
  sums.update(actual)
 shares={k:v/denom for k,v in sums.items()}
 assert all(abs(v-a['pooled_instrumented_forward_share'][k])<1e-12 for k,v in shares.items())
 f=math.fsum(r['bare']['forward_stream_ms'] for r in a['rows'])
 b=math.fsum(r['bare']['backward_stream_ms'] for r in a['rows'])
 data[arm]={'pooled_forward_shares':shares,'bare_forward_sum_ms':f,'bare_backward_sum_ms':b,'instrumented_forward_sum_ms':denom,
 'instrumented_minus_bare_forward_percent':100*(denom/f-1),'parameters_unchanged':True,'all_8_pairs_passed':True}
result={'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'input_manifest':M,'sha_verified_files':len(m['files']),
 'event_counts':dict(c),'counts':s['counts'],'named_gradient_records_checked':gradient_checks,'arms':data,'elapsed_wall_seconds':s['elapsed_wall_seconds'],
 'finished_utc':s['finished_utc'],'model_calls':0,'new_tensor_replay':False,'new_gradient_computation':False,'usd_cost':None,
 'scope':'Offline SHA/counter/logged loss+gradient tolerance/state-hash/forward-timing recomputation. Logged numerical comparisons verified; no independent fresh tensor replay.',
 'limitations':['Shared GPU at 100 percent utilization','Forward-only attribution','No optimizer/full-step accounting','Not corpus weighted','First backward cold retained','No causal kernel attribution or speed claim']}
(OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
lines=['# D/W 固定权重成本诊断','',
 '诊断已完成，约43.864秒。使用既有D/W最终模型，固定8个不同长度训练窗口，36次前向、32次反向、0参数更新；4次前向为无梯度热身。它用于定位当前实现开销，不是新训练或方法验证。','',
 '| 组 | GDN占带钩子前向 | 三层global占带钩子前向 | 8窗bare F+B流耗时合计 |',
 '|---|---:|---:|---:|']
for arm,a in data.items():lines.append(f"| {arm} | {a['pooled_forward_shares']['gdn_mixer']:.2%} | {a['pooled_forward_shares']['global_mixer']:.2%} | {(a['bare_forward_sum_ms']+a['bare_backward_sum_ms'])/1000:.6f}秒 |")
lines+=['','占比按模块区间总和除以同次带钩子的前向总和计算，不是各窗比例平均，也不是完整训练占比。CUDA区间包含调度等待及host发射间隙，不等于GPU核忙碌时间。',
 '','前后设备利用率均100%，有其他共享作业；D后W固定顺序且首个反向未热身。带钩子的前向合计反而比bare少约10.3%（D）/5.1%（W），不能把差值解释为钩子本身开销。上述F+B数值不构成W比D更快的证据。',
 '',f"离线复核了{len(m['files'])}份文本SHA、36F/32B/0更新原始事件计数、16对loss与{gradient_checks}条具名梯度比较记录、参数前后摘要及两检查点身份、全部组件与pooled汇总。模型执行时的钩子一致性阈值为atol=1e-6、rtol=1e-5；本审计核对保存的比较结果，没有另做张量重放。两组模型和权重文件均保持不变。",
 '', '下一项优先检查两组共同GDN路径的重复分段/同步操作，并在相同输出与梯度门槛下验证，见 `docs/babylm-common-gdn-next-step-2026-09-21.md`。当前没有实现新内核、证明端到端节约成本或建立论文创新。',
 '', '最终模型和本次原始文本均已本机逐SHA备份。进程已退出、项目锁已释放，自动监控仍暂停；没有操作共享服务器其他任务或电源。当前费率未知，不虚构美元费用。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'status':'passed','sha_files':len(m['files']),'gradient_records':gradient_checks,'arms':data,'model_calls':0}))

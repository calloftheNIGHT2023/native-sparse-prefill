"""Verify fresh-book predictions and frozen three-contrast paired analysis."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np,torch
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 name='fresh-book256';a=R/f'exports/{name}-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes'];dest=R/f'results/cloud-{name}-evidence-v0';dest.mkdir(exist_ok=True)
 with tarfile.open(a) as t:
  mn=name+'-manifest.json';entries=json.loads(t.extractfile(mn).read())['files'];names=[m.name for m in t];assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{mn}
  for e in entries:
   m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts;raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
 pp=dest/f'provenance/{name}-protocol-v0.json';assert sha(pp)==sha(R/f'provenance/{name}-protocol-v0.json');p=load(pp)
 for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
 out=dest/f'results/{name}-stage-v0';control=load(out/'result.json');assert control['status']=='complete' and control['optimizer_updates']==0 and control['protocol_sha256']==sha(pp)
 meta=load(dest/'data/fresh-book256-v0/tasks.json');manifest=load(dest/'data/fresh-book256-v0/manifest.json');arr=np.load(dest/'data/fresh-book256-v0/tasks.npz');flat=arr['input_ids'];offsets=arr['offsets'];assert len(meta)==1029 and len(manifest['families'])==256
 families=[f['family_id'] for f in manifest['families']];assert len(set(f['book_id'] for f in manifest['families']))==256 and len(set(families))==256
 for f in manifest['families']:
  assert sha(R/'data/fresh-book256-v0/raw'/f"{f['book_id']}.txt")==f['sha256']
 for family in families:
  indices=[i for i,x in enumerate(meta) if x['family_id']==family];assert len(indices)==4
  seqs=[flat[offsets[i]:offsets[i+1]] for i in indices];assert all(len(x)==32768 for x in seqs) and all(np.count_nonzero(seqs[0]!=x)==1 for x in seqs[1:])
 results={};records=[];base_gate=None;total=0
 for ci,cjob in enumerate(control['jobs']):
  j=p['jobs'][ci];assert cjob['name']==j['name'] and cjob['returncode']==0;d=out/j['name'];v=load(d/'result.json');assert v['job']==j and v['status']=='complete' and v['identity']==j['identity'] and v['step']==j['step'] and v['evaluation_k']==0 and v['optimizer_updates']==0 and v['task_predictions']==1029 and v['nll_forwards']==4
  assert v['eval_source_sha256']==sha(dest/'scripts/eval_fresh_book256_v0.py')==sha(d/'source.py') and v['environment']['gpu']==p['evaluation_gpu'];assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(v['calibration_values'],j['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
  mirror=R/('results/cloud-expanded76-evidence-v0' if j['step']==0 else 'results/cloud-matched-restore-training-evidence-v0');assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'];cp=torch.load(mirror/j['path'],map_location='cpu',weights_only=False);assert cp['step']==cp['data_cursor']==j['step'] and cp['identity']==j['identity'];params=cp['params']
  selected={f'model.layers.{i}.self_attn.{q}.B' for i in range(24) for q in ['q_proj','k_proj']} if j['restore_qk'] else set();assert set(v['selected_restore_B'])==selected and v['other_params_unchanged']
  for n in selected:params[n].zero_()
  if selected:
   artifact=torch.load(d/'restored-adapter.pt',map_location='cpu',weights_only=False);assert artifact['step']==j['step'] and all(torch.equal(artifact['params'][n],params[n]) for n in params)
  assert hashlib.sha256(b''.join(x.numpy().tobytes() for x in params.values())).hexdigest()==v['parameter_digest']
  preds=[json.loads(s) for s in (d/'task-predictions.jsonl').read_text().splitlines()];assert len(preds)==1029 and preds==v['word_predictions']
  for x,m in zip(preds,meta):
   assert all(x[k]==z for k,z in m.items()) and len(x['choice_logits'])==4 and np.isfinite(x['choice_logits']).all() and x['prediction']==int(np.argmax(x['choice_logits']))
   if m['gold'] is not None:assert x['correct']==(x['prediction']==m['gold'])
  q=np.array([[next(x['correct'] for x in preds if x['family_id']==f and x['gold']==g) for g in range(4)] for f in families],dtype=float).mean(axis=1)
  results[j['name']]=q;records.append(dict(name=j['name'],accuracy=float(q.mean()),all_four_accuracy=float((q==1).mean()),short_correct=sum(x['correct'] for x in preds if x['variant']=='short')));total+=1029
  if j['phase']=='base_gate':
   base_gate=v['base_gate'];assert base_gate['long_correct']==int(q.sum()*4) and base_gate['short_correct']==records[-1]['short_correct'];assert base_gate['passed']==(base_gate['short_correct']==4 and base_gate['long_correct']>=768)
 assert total==control['task_predictions'] and len(control['jobs'])==(9 if base_gate['passed'] else 1)
 contrasts=[];rng=np.random.default_rng(p['primary']['bootstrap_seed'])
 if base_gate['passed']:
  for comp in p['primary']['comparisons']:
   delta=np.stack([results[a]-results[b] for a,b in comp['pairs']]).mean(axis=0);idx=rng.integers(0,256,(20000,256));ci=np.quantile(delta[idx].mean(axis=1),p['primary']['quantiles']);contrasts.append(dict(name=comp['name'],difference_pp=float(delta.mean()*100),conditional98_333ci_pp=(ci*100).tolist(),noninferior=bool(ci[0]>-.05),per_seed_difference_pp=[float((results[a]-results[b]).mean()*100) for a,b in comp['pairs']]))
 result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,control=control,optimizer_updates=0,task_predictions=total,nll_forwards=4*len(control['jobs']),base_gate=base_gate,records=records,contrasts=contrasts,scope=p['scope']);target=R/f'results/{name}-audit-v0';target.mkdir(exist_ok=True);(target/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# 256本新背景的长程检索确认','','固定旧问题模板与答案词，每本新书一个32K背景、四个反事实答案；以书为配对单位。所有评测为密集注意力，已有检查点，不训练。','','基座门槛：'+json.dumps(base_gate),'','|模型|准确率|四个答案全对的书占比|短题正确数|','|---|---:|---:|---:|']
 for x in records:lines.append(f"|{x['name']}|{x['accuracy']*100:.2f}%|{x['all_four_accuracy']*100:.2f}%|{x['short_correct']}/4|")
 lines+=['','三项预先固定比较：'+json.dumps(contrasts,ensure_ascii=False),'','新背景不等于新任务模板，也不等于自然QA；四个反事实答案按同书聚类，两个旧训练种子先平均，三项使用98.333%区间。5pp界限固定。原有自然文本质量与计时来自旧审计，本轮只独立确认背景转移，不把旧结果当新证据。基座若失败，停止后续条件而不改变题。已知QK-Restore不作为原创。']
 (R/'docs/fresh-book256-results-2026-09-17.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',predictions=total,contrasts=contrasts)))
if __name__=='__main__':main()

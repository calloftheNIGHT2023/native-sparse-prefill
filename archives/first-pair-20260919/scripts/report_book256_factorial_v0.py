"""Audit missing factorial cells and retained old predictions before paired analysis."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np,torch
R=Path(__file__).resolve().parents[1];NAME='book256-factorial'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=R/f'exports/{NAME}-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes'];dest=R/f'results/cloud-{NAME}-evidence-v0';dest.mkdir(exist_ok=True)
 with tarfile.open(a) as t:
  mn=NAME+'-manifest.json';entries=json.loads(t.extractfile(mn).read())['files'];names=[m.name for m in t];assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{mn}
  for e in entries:
   m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts;raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
 pp=dest/f'provenance/{NAME}-protocol-v0.json';assert sha(pp)==sha(R/f'provenance/{NAME}-protocol-v0.json');p=load(pp)
 for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
 # Revalidate both parent archives and every corresponding extracted file used below.
 oldroots=[R/'results/cloud-fresh-book256-user-stop-v0',R/'results/cloud-fresh-book256-resume-evidence-v0']
 for parent,root in zip(p['parent_archives'],oldroots):
  assert sha(R/parent['path'])==parent['sha256']
  with tarfile.open(R/parent['path']) as t:
   for m in t:
    if m.isfile() and (m.name.startswith('results/fresh-book256') or m.name.startswith('provenance/fresh-book256')):
     assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
     assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==sha(root/m.name),m.name
 out=dest/f'results/{NAME}-stage-v0';control=load(out/'result.json');assert control['status']=='complete' and control['optimizer_updates']==0 and control['protocol_sha256']==sha(pp)
 assert len(control['jobs'])==8 and all(x['returncode']==0 for x in control['jobs']) and control['task_predictions']==8232
 meta=load(dest/'data/fresh-book256-v0/tasks.json');manifest=load(dest/'data/fresh-book256-v0/manifest.json');families=[f['family_id'] for f in manifest['families']];assert len(meta)==1029 and len(set(families))==256
 arr=np.load(dest/'data/fresh-book256-v0/tasks.npz');flat=arr['input_ids'];offsets=arr['offsets']
 assert len(set(f['book_id'] for f in manifest['families']))==256
 for f in manifest['families']:
  assert sha(R/'data/fresh-book256-v0/raw'/f"{f['book_id']}.txt")==f['sha256']
  ix=[i for i,x in enumerate(meta) if x['family_id']==f['family_id']];assert len(ix)==4
  seq=[flat[offsets[i]:offsets[i+1]] for i in ix];assert all(len(x)==32768 for x in seq) and all(np.count_nonzero(seq[0]!=x)==1 for x in seq[1:])
 oldp=load(oldroots[0]/'provenance/fresh-book256-protocol-v0.json');resp=load(oldroots[1]/'provenance/fresh-book256-resume-protocol-v0.json')
 assert sha(oldroots[0]/'provenance/fresh-book256-protocol-v0.json')==p['data_sha256']['provenance/fresh-book256-protocol-v0.json']
 assert sha(oldroots[1]/'provenance/fresh-book256-resume-protocol-v0.json')==p['data_sha256']['provenance/fresh-book256-resume-protocol-v0.json']
 evidence=[(j,oldroots[0]/'results/fresh-book256-stage-v0'/j['name'],oldp['source_sha256']['scripts/eval_fresh_book256_v0.py']) for j in oldp['jobs'][:6]]+[(j,oldroots[1]/'results/fresh-book256-resume-stage-v0'/j['name'],resp['source_sha256']['scripts/eval_fresh_book256_resume_v0.py']) for j in resp['jobs']]+[(j,out/j['name'],p['source_sha256']['scripts/eval_book256_factorial_v0.py']) for j in p['jobs']]
 results={};records=[]
 for j,d,source in evidence:
  v=load(d/'result.json');assert v['job']==j and v['status']=='complete' and v['identity']==j['identity'] and v['step']==j['step'] and v['evaluation_k']==0 and v['optimizer_updates']==0 and v['task_predictions']==1029 and v['nll_forwards']==4
  assert v['eval_source_sha256']==source==sha(d/'source.py') and v['environment']['gpu']==p['evaluation_gpu']
  assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(v['calibration_values'],j['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
  mirror=R/('results/cloud-expanded76-evidence-v0' if j['step']==0 else 'results/cloud-matched-restore-training-evidence-v0');assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256']
  assert sha(mirror/j['training_result'])==j['training_result_sha256']
  cp=torch.load(mirror/j['path'],map_location='cpu',weights_only=False);assert cp['step']==cp['data_cursor']==j['step'] and cp['identity']==j['identity'];params=cp['params']
  selected={f'model.layers.{i}.self_attn.{q}.B' for i in range(24) for q in ['q_proj','k_proj']} if j['restore_qk'] else set();assert set(v['selected_restore_B'])==selected and v['other_params_unchanged']
  for n in selected:params[n].zero_()
  if selected:
   artifact=torch.load(d/'restored-adapter.pt',map_location='cpu',weights_only=False);assert artifact['step']==j['step'] and set(artifact['params'])==set(params) and all(torch.equal(artifact['params'][n],params[n]) for n in params)
  assert hashlib.sha256(b''.join(x.numpy().tobytes() for x in params.values())).hexdigest()==v['parameter_digest']
  preds=[json.loads(s) for s in (d/'task-predictions.jsonl').read_text().splitlines()];assert len(preds)==1029 and preds==v['word_predictions']
  for x,m in zip(preds,meta):
   assert all(x[k]==z for k,z in m.items()) and len(x['choice_logits'])==4 and np.isfinite(x['choice_logits']).all() and x['prediction']==int(np.argmax(x['choice_logits']))
   if m['gold'] is not None:assert x['correct']==(x['prediction']==m['gold'])
  q=np.array([[next(x['correct'] for x in preds if x['family_id']==f and x['gold']==g) for g in range(4)] for f in families],dtype=float).mean(axis=1)
  assert j['name'] not in results;results[j['name']]=q
  records.append(dict(name=j['name'],accuracy=float(q.mean()),all_four_accuracy=float((q==1).mean()),short_correct=sum(x['correct'] for x in preds if x['variant']=='short')))
 assert len(results)==17
 rng=np.random.default_rng(p['primary']['bootstrap_seed']);contrasts=[]
 for comp in p['primary']['comparisons']:
  perseed=np.stack([sum(w*results[name] for name,w in terms) for terms in comp['seed_terms']]);delta=perseed.mean(axis=0);idx=rng.integers(0,256,(p['primary']['draws'],256));ci=np.quantile(delta[idx].mean(axis=1),p['primary']['quantiles'])
  row=dict(name=comp['name'],kind=comp['kind'],difference_pp=float(delta.mean()*100),conditional98_75ci_pp=(ci*100).tolist(),per_seed_difference_pp=(perseed.mean(axis=1)*100).tolist())
  if comp['kind']=='noninferiority':row['noninferior']=bool(ci[0]>-p['primary']['accuracy_margin_pp']/100)
  contrasts.append(row)
 result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,control=control,optimizer_updates=0,task_predictions=8232,combined_task_predictions=17493,nll_forwards=32,records=records,contrasts=contrasts,scope=p['scope'])
 target=R/f'results/{NAME}-audit-v0';target.mkdir(exist_ok=True);(target/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# 书籍背景上的完整恢复对照','','这是看到上一批结果后固定的追加诊断，使用已暴露的书籍和两个旧训练种子。没有重新训练；补齐密集恢复、稀疏不恢复8组。不能算独立新任务确认。','','|条件|准确率|四答案全对书占比|短题|','|---|---:|---:|---:|']
 for x in records:lines.append(f"|{x['name']}|{100*x['accuracy']:.2f}%|{100*x['all_four_accuracy']:.2f}%|{x['short_correct']}/4|")
 lines+=['','固定4项对照（98.75%按书配对区间；交互不是等价检验）：',json.dumps(contrasts,ensure_ascii=False,indent=2),'','上一批原始3项比较保持不变。密集64步已通过相对稀疏128步的5pp容差，不能宣称128步稀疏是最低成本方案。比较恢复后的稀疏与恢复后的密集用于公平方法对照；恢复增益交互只作为当前两个种子、当前数据上的诊断。基座接近100%，天花板限制判别能力。Q/K恢复是已知方法；本轮不计时训练、不证明原创性。',p['scope']]
 (R/'docs/book256-factorial-results-2026-09-17.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',new_predictions=8232,contrasts=contrasts)))
if __name__=='__main__':main()

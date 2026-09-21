"""Freeze method-specific LR choices using calibration outcomes only."""
import math
def select_lrs(results,cfg):
 expected={(k,lr) for k in cfg['methods'] for lr in cfg['learning_rates']}
 assert len(results)==len(expected)
 seen=set();scores={}
 init=set()
 for r in results:
  assert r['status']=='complete' and r['phase']=='train' and r['step']==cfg['steps']
  i=r['identity'];assert i['seed']==cfg['seeds'][0]
  pair=(i['k'],i['lr']);assert pair in expected and pair not in seen;seen.add(pair);init.add(r['initial_sha256'])
  assert all(e['split']=='calibration' for e in r['evaluations'])
  e=[e for e in r['evaluations'] if e['step']==cfg['steps']];assert len(e)==1
  assert math.isfinite(e[0]['mean_nll'])
  scores[pair]=e[0]['mean_nll']
 assert seen==expected and len(init)==1
 return {str(k):min(cfg['learning_rates'],key=lambda lr:(scores[k,lr],lr)) for k in cfg['methods']}

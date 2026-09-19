"""Exhaustive finite-population check of the normalizer estimator's algebra."""
from datetime import datetime,timezone
from itertools import combinations
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
 out=ROOT/'results/head-mixture-math-v0.json'
 if out.exists(): raise FileExistsError('Preserve math check')
 x=torch.tensor([1.,3.,8.,10.],dtype=torch.float64)
 n,m,zs=len(x),2,2.
 population_z=zs+x.sum()
 estimates=torch.stack([zs+n*x[list(ids)].mean() for ids in combinations(range(n),m)])
 exact_variance=n*n/m*(1-m/n)*x.var(unbiased=True)
 retained_true=zs/population_z
 retained_estimates=zs/estimates
 observed_mae=(retained_estimates-retained_true).abs().mean()
 mae_bound=exact_variance.sqrt()/population_z
 assert torch.allclose(estimates.mean(),population_z,atol=1e-12,rtol=1e-12)
 assert torch.allclose(estimates.var(unbiased=False),exact_variance,atol=1e-12,rtol=1e-12)
 assert retained_estimates.mean()>=retained_true
 assert observed_mae<=mae_bound
 # Mixture-TV contraction check for many strictly positive head distributions.
 torch.manual_seed(812)
 max_bound_violation=0.
 for _ in range(100):
  p=torch.randn(5,11,dtype=torch.float64).softmax(-1)
  mass=torch.rand(5,dtype=torch.float64)+.01
  approximate=torch.rand(5,dtype=torch.float64)+.01
  w=mass/mass.sum(); v=approximate/approximate.sum()
  tv=(w@p-v@p).abs().sum()/2
  weight_tv=(w-v).abs().sum()/2
  mass_bound=(mass-approximate).abs().sum()/mass.sum()
  assert tv<=weight_tv+1e-12 and weight_tv<=mass_bound+1e-12
  max_bound_violation=max(max_bound_violation,(tv-weight_tv).item())
 record={'checked_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
  'subsets_enumerated':len(estimates),'true_partition':population_z.item(),
  'mean_partition_estimate':estimates.mean().item(),'exact_variance':exact_variance.item(),
  'enumerated_variance':estimates.var(unbiased=False).item(),
  'true_retained_mass':retained_true.item(),'mean_estimated_retained_mass':retained_estimates.mean().item(),
  'retained_mass_mae':observed_mae.item(),'mae_bound':mae_bound.item(),
  'mixture_cases_checked':100,'max_mixture_bound_violation':max_bound_violation,
  'scope':'Elementary algebra and exhaustive illustrative checks; not a novel theorem or empirical language-model result.'}
 out.write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record))


if __name__=='__main__':main()

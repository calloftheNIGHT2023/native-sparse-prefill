"""Freeze confirmation choices and create data disjoint from prior train/dev/test."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from router_author_control import generated, interventions, row_hashes, validate


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()


def main(a):
    out = a.output; out.mkdir(parents=True, exist_ok=False)
    configs = [dict(name='exact', method='exact', block=0, routes=0, offset=0, layers=[0,1])]
    configs += [dict(name=f'minmax32_r4_offset{s}', method='minmax', block=32, routes=4, offset=s, layers=[0,1]) for s in [0,8,16,24]]
    configs += [dict(name='minmax16_r4', method='minmax', block=16, routes=4, offset=0, layers=[0,1]),
                dict(name='minmax32_r4_layer0', method='minmax', block=32, routes=4, offset=16, layers=[0]),
                dict(name='minmax32_r4_layer1', method='minmax', block=32, routes=4, offset=16, layers=[1]),
                dict(name='mean32_r4_negative_control', method='mean', block=32, routes=4, offset=0, layers=[0,1])]
    plan = dict(frozen_utc=utc(), test_seed=2026091511, noise_seed=2026091512, rows=1024, configs=configs,
        checkpoint_paths=['results/router-author-falsification-lowlr-v0/checkpoint.pt',
                          'results/router-author-falsification-confirm-v0/checkpoint.pt'],
        primary='Check b32/r4 against exact on all splits, both model seeds, and all four fixed block origins.',
        pass_rule='Every compared prediction must equal the same-dtype exact reference; weaker accuracy evidence reported separately.',
        scope='Fresh data confirmation of post-hoc shortlist; not a new native-training method; no tuning on this data.',
        basis='Selected after complete 18-setting exposed-data audit; boundary shifts test source-bank alignment artifact.')
    plan['checkpoints'] = {p: sha(ROOT / p) for p in plan['checkpoint_paths']}
    (out / 'preregistration.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    test = generated(plan['test_seed'], plan['rows']); validate(test)
    hashes = row_hashes(test); assert len(hashes) == len(test['inputs'])
    exclusions = ['results/router-author-control-v0/data.pt',
                  'results/router-author-sparse-v0/evaluation-data.pt',
                  'results/router-author-exact-train-v0/evaluation-data.pt',
                  'results/router-author-falsification-evaluation-v0/evaluation-data.pt']
    checked = []
    for path in exclusions:
        old = torch.load(ROOT / path, map_location='cpu', weights_only=True)
        for split, part in old.items():
            if not isinstance(part, dict) or 'inputs' not in part: continue
            previous = row_hashes(part); assert not hashes & previous, (path, split)
            checked.append(dict(path=path, sha256=sha(ROOT / path), split=split, prior_unique_rows=len(previous), overlap=0))
    swapped, noisy, pairs = interventions(test)
    noise = torch.randint(0, 8192, test['inputs'].shape, generator=torch.Generator().manual_seed(plan['noise_seed']))
    noisy['inputs'] = torch.where(test['inputs'] == 0, noise, test['inputs']); validate(noisy, False)
    for i, (q, s, t) in enumerate(pairs):
        assert swapped['labels'][i,q] == test['inputs'][i,t] and swapped['labels'][i,q] != test['labels'][i,q]
    torch.save(dict(test=test, noisy=noisy, swapped=swapped, swap_positions=pairs), out / 'evaluation-data.pt')
    audit = dict(finished_utc=utc(), checked=checked, unique_clean_rows=1024,
                 data_sha256=sha(out / 'evaluation-data.pt'), all_swaps_valid=True, all_noise_labels_valid=True)
    (out / 'data-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (out / 'preparation-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); main(p.parse_args())

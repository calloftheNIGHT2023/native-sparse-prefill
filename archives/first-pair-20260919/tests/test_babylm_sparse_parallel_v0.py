"""No models/processes: independent synthetic evidence and subprocess oracles."""
from __future__ import annotations
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts import run_babylm_sparse_parallel_v0 as module

PROJECT = Path(__file__).resolve().parents[1]


class Fixture:
    def __init__(self, root):
        self.root = root
        self.nvidia_calls = 0
        self.child_calls = []
        self.q = json.loads((PROJECT / 'configs/babylm-a6000-scientific-pair-20260918-v0.json').read_text())
        for name in ('train.json', 'dev.json', 'tokens.bin'):
            path = root / 'data' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'synthetic only ' + name.encode())
        self.q.update(train_manifest='data/train.json', train_manifest_sha256=module.sha(root/'data/train.json'),
                      eval_manifest='data/dev.json', eval_manifest_sha256=module.sha(root/'data/dev.json'), pod_id='old-pod')
        self.p = copy.deepcopy(self.q)
        self.p.update(pod_id='new-pod', single_pod_ceiling_usd=75, paid_ceiling_usd=75, stage_spent_usd=0.1)
        self.pin('parent_scientific_protocol', self.q)
        release = dict(schema_version=1, kind='old_sparse_schedule_release',
                       parent_scientific_protocol_sha256=self.p['parent_scientific_protocol_sha256'],
                       old_pod_id='old-pod', new_pod_id='new-pod', old_sparse_schedule_disabled=True,
                       old_sparse_not_running=True, verified_utc='2026-09-18T01:00:00+00:00')
        self.pin('old_sparse_release', release)
        source = root / 'src/babylm_hybrid/synthetic.py'
        source.parent.mkdir(parents=True); source.write_text('synthetic source receipt\n')
        sources = {str(source): module.sha(source)}
        self.data = {'manifest_path':str(root/'data/train.json'), 'manifest_sha256':self.p['train_manifest_sha256'],
                     'artifacts': {str(root/'data/tokens.bin'):module.sha(root/'data/tokens.bin')},
                     'tokenizer_sha256':'a'*64, 'total_windows':192}
        self.initial = {'embedding.weight':'b'*64, 'layers.3.mixer.indexer.q_proj.weight':'c'*64}
        counts = {'updates':12, 'engineering_updates':12, 'scientific_updates':0, 'word_exposures':1920,
                  'input_tokens':3072, 'loss_tokens':2880, 'windows':192, 'forward_calls':192, 'backward_calls':192}
        for prefix,mode in [('reference_dense_preflight','dense'), ('reference_sparse_preflight','sparse'), ('new_sparse_preflight','sparse')]:
            q = dict(self.p, scope='gpu_preflight', max_updates=12, eval_initial=False, eval_final=False,
                     eval_every_updates=0, pod_id='new-pod' if prefix.startswith('new_') else 'old-pod')
            self.pin(prefix+'_protocol', q)
            initial = dict(self.initial) if mode=='sparse' else {'embedding.weight':'b'*64}
            summary = {'scope':'gpu_preflight', 'mode':mode, 'status':'max_updates_reached',
                'protocol_sha256':module.protocol_hash(q), 'counts':counts, 'eval_counts':{'forward_calls':0,'evaluations':0},
                'source_hashes':sources, 'data_fingerprint':self.data, 'initial_parameter_hashes':initial,
                'parameter_counts':{'total':96866328 if mode=='sparse' else 95391000}, 'cursor':{'epoch':0,'position':192}}
            self.pin(prefix+'_summary',summary)
            if mode=='sparse':
                start={k:summary[k] for k in ('protocol_sha256','source_hashes','data_fingerprint','initial_parameter_hashes')}
                events=[dict(start,event_id=1,type='run_start',mode=mode)]
                for i in range(1,13):
                    update={'event_id':i+1,'type':'update','mode':mode,'counts':dict(counts,updates=i),
                            'window_ids':[{'epoch':0,'window_index':i-1}], 'cursor':{'epoch':0,'position':i},
                            'lr':{'backbone':0.00001,'indexer':0.00003}, 'lr_word_position':i*160,
                            'loss_tokens_this_update':240}
                    update.update({key:float(i) for key in module.REPLAY_FIELDS})
                    events.append(update)
                events.append(dict(event_id=14,type='run_stop',mode=mode,counts=counts,status='max_updates_reached'))
                self.pin(prefix+'_events',events, lines=True)
        report = json.loads((PROJECT/'results/babylm-a6000-preflight-20260918-v0/gpu-numerics.json').read_text())
        report['source_hashes']=sources
        report['hardware']['nvidia_smi']={'returncode':0,'stdout':'NVIDIA RTX A6000, 49140 MiB, GPU-new, 580.159.04\n'}
        self.pin('numerics',report)
        self.save()

    def pin(self,prefix,obj,lines=False):
        path=self.root/(prefix+'.jsonl' if lines else prefix+'.json')
        path.write_bytes((b''.join(module.canonical(x)+b'\n' for x in obj)) if lines else module.canonical(obj)+b'\n')
        self.p[prefix+'_path']=str(path); self.p[prefix+'_sha256']=module.sha(path)

    def saved(self,prefix):
        return json.loads(Path(self.p[prefix+'_path']).read_text())

    def save(self):
        self.path=self.root/'protocol.json'
        self.path.write_bytes(module.canonical(self.p)+b'\n')
        self.hash=module.sha(self.path)

    def process(self,command,**kwargs):
        if command[0]=='nvidia-smi':
            self.nvidia_calls+=1
            return SimpleNamespace(returncode=0,stdout='GPU-new, 580.159.04\n')
        self.child_calls.append(command)
        assert command[command.index('--mode')+1]=='sparse'
        assert '--resume-checkpoint' not in command
        protocol=json.loads(Path(command[command.index('--protocol')+1]).read_text())
        out=Path(command[command.index('--output-dir')+1]); out.mkdir()
        s=self.saved('new_sparse_preflight_summary')
        s.update(scope='scientific',status='word_budget_reached',protocol_sha256=module.protocol_hash(protocol),
                 counts={'updates':14000,'scientific_updates':14000,'engineering_updates':0,'word_exposures':99999900})
        (out/'summary.json').write_bytes(module.canonical(s)+b'\n')
        return SimpleNamespace(returncode=0)


class SparseParallelTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.f=Fixture(self.root)
        self.root_patch=patch.object(module,'ROOT',self.root); self.root_patch.start()
        self.env_patch=patch.dict('os.environ',{'RUNPOD_POD_ID':'new-pod'}); self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop(); self.root_patch.stop(); self.temp.cleanup()

    def launch(self,execute=True,out='output'):
        return module.launch(self.f.path,self.f.hash,self.root/out,execute)

    def test_01_plan_starts_nothing(self):
        with patch.object(module.subprocess,'run',side_effect=AssertionError('No subprocess in plan')):
            result=self.launch(False)
        self.assertEqual(result['status'],'plan_only_no_processes_started')
        self.assertFalse((self.root/'output').exists())

    def test_02_complete_fresh_sparse_only(self):
        with patch.object(module.subprocess,'run',side_effect=self.f.process):
            result=self.launch()
        self.assertEqual(len(self.f.child_calls),1)
        self.assertEqual(self.f.nvidia_calls,1)
        self.assertEqual(result['status'],'scientific_sparse_complete_quality_not_yet_adjudicated')
        self.assertEqual(len(result['preflight_replay']['comparisons']),84)
        self.assertTrue(all(x['absolute_error']==0 for x in result['preflight_replay']['comparisons']))
        self.assertFalse(list((self.root/'logs').glob('*.lock')))
        self.assertEqual(result['jobs'][0]['counts']['engineering_updates'],0)

    def test_03_recipe_hash_and_budget_tamper_rejected_before_process(self):
        original=copy.deepcopy(self.f.p)
        for key,value in [('learning_rate',0.0004),('eval_every_updates',125),('milestone_word_exposures',[1]),
                          ('single_pod_ceiling_usd',76),('hard_timeout_seconds_per_run',float('inf')),
                          ('resume_checkpoint','forbidden.pt')]:
            with self.subTest(key=key):
                self.f.p=copy.deepcopy(original); self.f.p[key]=value
                if not math_isfinite(value):
                    # Raw JSON parser accepts Infinity; launcher must explicitly reject.
                    self.f.path.write_text(json.dumps(self.f.p)); self.f.hash=module.sha(self.f.path)
                else:self.f.save()
                with patch.object(module.subprocess,'run',side_effect=AssertionError('No subprocess')):
                    with self.assertRaises(ValueError):self.launch()
        self.f.p=original;self.f.save()
        with self.assertRaises(ValueError):module.launch(self.f.path,'0'*64,self.root/'bad',True)

    def test_04_release_gpu_and_nested_numerics_rejected(self):
        release=self.f.saved('old_sparse_release'); release['old_sparse_schedule_disabled']=False
        self.f.pin('old_sparse_release',release);self.f.save()
        with patch.object(module.subprocess,'run',side_effect=AssertionError('No subprocess')):
            with self.assertRaisesRegex(ValueError,'not explicitly released'):self.launch(out='release')
        release['old_sparse_schedule_disabled']=True;self.f.pin('old_sparse_release',release)
        numerical=self.f.saved('numerics');numerical['checks']['cpu_cuda_fp32_sparse']['measurements']['lm_loss']['passed']=False
        self.f.pin('numerics',numerical);self.f.save()
        with patch.object(module.subprocess,'run',side_effect=AssertionError('No subprocess')):
            with self.assertRaisesRegex(ValueError,'Nested numerical'):self.launch(out='nested')
        numerical['checks']['cpu_cuda_fp32_sparse']['measurements']['lm_loss']['passed']=True
        numerical['checks']['cpu_cuda_fp32_sparse']['measurements']['lm_loss']['threshold']['rtol']=0.1
        self.f.pin('numerics',numerical);self.f.save()
        with patch.object(module.subprocess,'run',side_effect=AssertionError('No subprocess')):
            with self.assertRaisesRegex(ValueError,'Nested numerical threshold'):self.launch(out='threshold')
        numerical['checks']['cpu_cuda_fp32_sparse']['measurements']['lm_loss']['threshold']['rtol']=0.001
        self.f.pin('numerics',numerical);self.f.save()
        with patch.object(module.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='GPU-other, 580.159.04\n')):
            with self.assertRaisesRegex(ValueError,'UUID/driver'):self.launch(out='gpu')

    def test_05_outside_original_replay_threshold_preserves_failure(self):
        path=Path(self.f.p['new_sparse_preflight_events_path'])
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[12]['loss_token_weighted_ce']=12.02  # Allowed difference is 0.0121.
        self.f.pin('new_sparse_preflight_events',rows,lines=True);self.f.save()
        with patch.object(module.subprocess,'run',side_effect=self.f.process):
            with self.assertRaisesRegex(ValueError,'full-candidate replay failed'):self.launch()
        self.assertEqual(self.f.child_calls,[])
        result=json.loads((self.root/'output/stage.json').read_text())
        self.assertEqual(result['status'],'stopped_with_failure')
        failed=[x for x in result['preflight_replay']['comparisons'] if not x['passed']]
        self.assertEqual([(x['update'],x['field']) for x in failed],[(12,'loss_token_weighted_ce')])

    def test_06_existing_lock_output_and_child_failure(self):
        locks=self.root/'logs';locks.mkdir()
        occupied=locks/'babylm-gpu-preflight.lock';occupied.write_text('another process')
        with self.assertRaises(FileExistsError):self.launch(out='locked')
        self.assertEqual(occupied.read_text(),'another process')
        self.assertFalse((locks/'babylm-scientific-pair.lock').exists())
        occupied.unlink()
        used=self.root/'used';used.mkdir();(used/'evidence').write_text('preserve')
        with self.assertRaisesRegex(ValueError,'Fresh empty'):self.launch(out='used')
        def fail(command,**kwargs):
            if command[0]=='nvidia-smi':return self.f.process(command,**kwargs)
            raise subprocess.TimeoutExpired(command,1)
        with patch.object(module.subprocess,'run',side_effect=fail):
            with self.assertRaises(subprocess.TimeoutExpired):self.launch(out='timeout')
        r=json.loads((self.root/'timeout/stage.json').read_text())
        self.assertEqual(r['jobs'][0]['status'],'hard_timeout_child_killed_and_waited')
        self.assertFalse(list(locks.glob('*.lock')))

    def test_07_initial_indexer_or_data_order_mismatch_rejected(self):
        original=self.f.saved('new_sparse_preflight_summary')
        s=copy.deepcopy(original);s['initial_parameter_hashes']['layers.3.mixer.indexer.q_proj.weight']='d'*64
        self.f.pin('new_sparse_preflight_summary',s)
        path=Path(self.f.p['new_sparse_preflight_events_path']);rows=[json.loads(l) for l in path.read_text().splitlines()]
        rows[0]['initial_parameter_hashes']=s['initial_parameter_hashes'];self.f.pin('new_sparse_preflight_events',rows,lines=True);self.f.save()
        with patch.object(module.subprocess,'run',side_effect=self.f.process):
            with self.assertRaisesRegex(ValueError,'indexer initialization'):self.launch(out='indexer')
        self.f.pin('new_sparse_preflight_summary',original)
        rows[0]['initial_parameter_hashes']=original['initial_parameter_hashes'];rows[6]['window_ids']=[{'epoch':0,'window_index':999}]
        self.f.pin('new_sparse_preflight_events',rows,lines=True);self.f.save()
        with patch.object(module.subprocess,'run',side_effect=self.f.process):
            with self.assertRaisesRegex(ValueError,'window_ids'):self.launch(out='order')
        self.assertEqual(self.f.child_calls,[])


def math_isfinite(value):
    import math
    return not isinstance(value,float) or math.isfinite(value)


if __name__=='__main__':unittest.main(verbosity=2)

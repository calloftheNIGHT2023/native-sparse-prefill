"""Mock subprocess tests only: no CUDA, model, SSH or scientific computation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_babylm_scientific_pair_v0 as launcher

STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
ARTIFACT_ROOT = PROJECT/'logs'/('babylm-scientific-launcher-mocks-'+STAMP)


class ScientificLauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = ARTIFACT_ROOT/self._testMethodName
        self.root = self.directory/'project'
        self.preflight = self.root/'preflight'
        self.preflight.mkdir(parents=True)
        self.output = self.root/'science'
        source = self.root/'fixture-source.py'; source.write_text('synthetic source receipt',encoding='utf-8')
        data = self.root/'train-manifest.json'; data.write_text('{}',encoding='utf-8')
        self.sources = {str(source):launcher.sha(source)}
        self.initial = {'embedding.weight':'a'*64}
        self.data = {'manifest_sha256':launcher.sha(data),'artifacts':{str(data):launcher.sha(data)}}
        p = json.loads((PROJECT/'configs/babylm-de-gpu-preflight-v0.draft.json').read_text(encoding='utf-8'))
        p.update(train_manifest=str(data),train_manifest_sha256=launcher.sha(data),eval_manifest=None,
                 hourly_rate_usd=.53,stage_spent_usd=.2,paid_ceiling_usd=10)
        for mode in ('dense','sparse'):
            launcher.write(self.preflight/f'{mode}-protocol.json',p)
            destination=self.preflight/mode; destination.mkdir()
            launcher.write(destination/'summary.json',{'scope':'gpu_preflight','mode':mode,'status':'max_updates_reached',
                'protocol_sha256':launcher.protocol_hash(p),'counts':{'updates':12},'source_hashes':self.sources,
                'initial_parameter_hashes':self.initial,'data_fingerprint':self.data})
        self.numerical={'passed':True,'status':'passed','source_hashes':self.sources,
                        'hardware':{'nvidia_smi':{'returncode':0,'stdout':'0, fake GPU, GPU-fixture, 123.4, 49140 MiB'}}}
        launcher.write(self.preflight/'gpu-numerics.json',self.numerical)
        self.stage={'status':'paired_preflight_complete_not_scientific_training',
                    'jobs':[{'label':label,'returncode':0} for label in ('gpu-numerics','dense','sparse')]}
        launcher.write(self.preflight/'stage.json',self.stage)
        self.p={**p,'scope':'scientific','launch_allowed':True,'pod_id':'test-pod',
                 'pair_stage_ceiling_usd':150.,'paid_ceiling_usd':150.,'max_updates':20000,
                 'max_wall_seconds':100,'hard_timeout_seconds_per_run':110,
                 'expected_stop_reason':'word_budget_reached',
                 'preflight_stage_path':str(self.preflight/'stage.json'),
                 'preflight_stage_sha256':launcher.sha(self.preflight/'stage.json'),
                 'preflight_numerics_sha256':launcher.sha(self.preflight/'gpu-numerics.json')}
        self.protocol=self.root/'science-protocol.json'
        self.calls=[]; self.failure_mode=None

    def stub(self, command, **kwargs):
        if command[0]=='nvidia-smi':
            self.calls.append('hardware_read')
            return subprocess.CompletedProcess(command,0,'GPU-fixture, 123.4\n','')
        mode=command[command.index('--mode')+1]; self.calls.append(mode)
        self.assertNotIn('--resume-checkpoint',command)
        self.assertGreater(kwargs['timeout'],0)
        if self.failure_mode=='returncode' and mode=='dense':
            return subprocess.CompletedProcess(command,17)
        if self.failure_mode=='timeout' and mode=='dense':
            raise subprocess.TimeoutExpired(command,kwargs['timeout'])
        path=Path(command[command.index('--protocol')+1]); p=launcher.load(path)
        destination=Path(command[command.index('--output-dir')+1]); destination.mkdir()
        launcher.write(destination/'summary.json',{'scope':'scientific','mode':mode,'status':'word_budget_reached',
            'protocol_sha256':launcher.protocol_hash(p),'source_hashes':self.sources,'initial_parameter_hashes':self.initial,
            'data_fingerprint':self.data,'counts':{'updates':10,'engineering_updates':0,'scientific_updates':10,
                                                  'word_exposures':100,'input_tokens':150,'loss_tokens':140},
            'cursor':{'epoch':1,'position':2},'eval_counts':{'forward_calls':0}})
        return subprocess.CompletedProcess(command,0)

    def invoke(self,p=None,execute=True):
        launcher.write(self.protocol,p or self.p)
        with mock.patch.object(launcher,'ROOT',self.root),mock.patch.dict('os.environ',{'RUNPOD_POD_ID':'test-pod'}),\
             mock.patch.object(launcher.subprocess,'run',side_effect=self.stub):
            return launcher.launch(self.protocol,launcher.sha(self.protocol),self.output,execute)

    def assert_unlocked(self):
        for name in ('babylm-scientific-pair.lock','babylm-gpu-preflight.lock'):
            self.assertFalse((self.root/'logs'/name).exists())

    def test_01_plan_zero_processes(self):
        result=self.invoke({**self.p,'launch_allowed':False,'paid_ceiling_usd':None},execute=False)
        self.assertEqual(result['status'],'plan_only_no_processes_started')
        self.assertEqual(self.calls,[]); self.assertFalse(self.output.exists())

    def test_02_budget_and_preflight_gates(self):
        for changed in ({'launch_allowed':False},{'hourly_rate_usd':None},{'paid_ceiling_usd':149},
                        {'stage_spent_usd':150},{'hard_timeout_seconds_per_run':float('inf')},
                        {'resume_checkpoint':'old.pt'}):
            with self.subTest(changed=changed),self.assertRaises((ValueError,TypeError)):
                self.invoke({**self.p,**changed})
            self.assertEqual(self.calls,[])
        self.stage['status']='running_sparse'; launcher.write(self.preflight/'stage.json',self.stage)
        self.p['preflight_stage_sha256']=launcher.sha(self.preflight/'stage.json')
        with self.assertRaisesRegex(ValueError,'has not completed'):
            self.invoke()
        self.assert_unlocked(); self.assertEqual(self.calls,[])

    def test_03_order_and_fresh_seeds_and_cost(self):
        result=self.invoke()
        self.assertEqual(self.calls,['hardware_read','dense','sparse'])
        self.assertEqual(result['status'],'scientific_pair_complete_quality_not_yet_adjudicated')
        d=launcher.load(self.output/'dense-protocol.json'); e=launcher.load(self.output/'sparse-protocol.json')
        self.assertGreaterEqual(e['stage_spent_usd'],d['stage_spent_usd'])
        for key in launcher.MATCH_FIELDS:
            self.assertEqual(d[key],e[key])
        self.assert_unlocked()
        with self.assertRaisesRegex(ValueError,'Fresh scientific'):
            self.invoke()

    def test_04_dense_failure_blocks_sparse(self):
        self.failure_mode='returncode'
        with self.assertRaisesRegex(RuntimeError,'dense failed'):
            self.invoke()
        self.assertEqual(self.calls,['hardware_read','dense'])
        saved=launcher.load(self.output/'stage.json')
        self.assertEqual(saved['status'],'stopped_with_failure'); self.assertIn('Traceback',saved['traceback'])
        self.assert_unlocked()

    def test_05_timeout_and_own_lock_cleanup(self):
        self.failure_mode='timeout'
        with self.assertRaises(subprocess.TimeoutExpired):
            self.invoke()
        saved=launcher.load(self.output/'stage.json')
        self.assertEqual(saved['jobs'][0]['status'],'hard_timeout_child_killed_and_waited')
        self.assertEqual(self.calls,['hardware_read','dense']); self.assert_unlocked()

    def test_06_foreign_lock_survives_and_partial_own_lock_releases(self):
        directory=self.root/'logs'; directory.mkdir()
        guard=directory/'babylm-gpu-preflight.lock'; guard.write_text('other job')
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(guard.read_text(),'other job')
        self.assertFalse((directory/'babylm-scientific-pair.lock').exists())
        self.assertEqual(self.calls,[])

    def test_07_candidate_mismatch_and_numeric_literal_true(self):
        p={**self.p,'backbone_seed':11}
        with self.assertRaisesRegex(ValueError,'backbone_seed'):
            self.invoke(p)
        self.assertEqual(self.calls,[]); self.assert_unlocked()
        self.output=self.root/'science-second-attempt'
        self.numerical['passed']='true'; launcher.write(self.preflight/'gpu-numerics.json',self.numerical)
        self.p['preflight_numerics_sha256']=launcher.sha(self.preflight/'gpu-numerics.json')
        with self.assertRaisesRegex(ValueError,'explicitly pass'):
            self.invoke()
        self.assertEqual(self.calls,[]); self.assert_unlocked()


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScientificLauncherTests))
    report={'utc':launcher.utc(),'passed':result.wasSuccessful(),'tests_run':result.testsRun,
            'failures':[str(x) for x in result.failures],'errors':[str(x) for x in result.errors],
            'actual_subprocesses':0,'model_forwards':0,'gpu_calls':0,'scientific_updates':0,
            'artifacts':str(ARTIFACT_ROOT),'source_sha256':launcher.sha(PROJECT/'scripts/run_babylm_scientific_pair_v0.py')}
    launcher.write(PROJECT/'results/babylm-scientific-launcher-mock-v0.json',report)
    with (PROJECT/'results/babylm-scientific-launcher-mock-v0-history.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(report)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)

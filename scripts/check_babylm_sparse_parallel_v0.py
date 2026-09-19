"""Preserve each mocked-launcher test attempt; no real subprocess/model/GPU."""
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    started=datetime.now(timezone.utc)
    name='babylm-sparse-parallel-mock-v0-'+started.strftime('%Y%m%dT%H%M%S%fZ')
    stream=io.StringIO()
    if len(sys.argv)>1:
        sys.path.insert(0,str(ROOT/'tests'))
        suite=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
    else:
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_babylm_sparse_parallel_v0.py')
    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    paths=['scripts/run_babylm_sparse_parallel_v0.py','tests/test_babylm_sparse_parallel_v0.py',
           'scripts/run_babylm_scientific_pair_v0.py','scripts/check_babylm_sparse_parallel_v0.py']
    report={'started_utc':started.isoformat(),'completed_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'mocked_launcher_only_no_real_model_process','passed':result.wasSuccessful(),
        'tests_run':result.testsRun,'selection':sys.argv[1:] or ['all seven tests'],
        'failures':len(result.failures),'errors':len(result.errors),
        'real_model_forward_calls':0,'real_model_backward_calls':0,'real_optimizer_updates':0,
        'scientific_updates':0,'gpu_work':0,'real_child_processes':0,
        'source_hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        'log':str(ROOT/'logs'/f'{name}.log')}
    (ROOT/'logs'/f'{name}.log').write_text(stream.getvalue(),encoding='utf-8')
    (ROOT/'results'/f'{name}.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (ROOT/'results'/'babylm-sparse-parallel-mock-v0.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(stream.getvalue());print(json.dumps(report,indent=2))
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())

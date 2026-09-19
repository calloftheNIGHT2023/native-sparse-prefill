"""Launch one frozen diagnostic after local quality evidence audit, then collect it."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,time,sys,traceback
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    state=R/'logs/full-step-cost-chain-v0.json';start=time.monotonic()
    try:
        while time.monotonic()-start<2100:
            v=load(R/'logs/common-dense-quality-v0-local-collector.json')
            if v['status']=='complete_verified':break
            assert v['status']!='failed_needs_inspection',v
            save(state,dict(status='waiting_for_quality_audit',utc=utc()));time.sleep(30)
        else:raise RuntimeError('Timed out without launching; inspect quality collector')
        assert load(R/'results/common-dense-quality-audit-v0/result.json')['status']=='verified'
        c=load(R/'logs/cloud-connection-current.json');host=c['ssh_user']+'@'+c['ssh_host'];root=c['root'];opts=['-i',c['ssh_key_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15']
        scp=['scp',*opts,'-P',str(c['ssh_port'])];ssh=['ssh',*opts,'-p',str(c['ssh_port']),host]
        a=R/'exports/full-step-cost-launch-v0.tar.gz';proof=load(a.with_suffix('.json'))
        subprocess.run(scp+[str(a),host+':'+root+'/exports/'+a.name],check=True,timeout=120)
        code='''from pathlib import Path
import json,tarfile,hashlib,subprocess
r=Path(ROOT);a=r/'exports/full-step-cost-launch-v0.tar.gz'
assert hashlib.sha256(a.read_bytes()).hexdigest()==HASH
assert json.loads((r/'results/common-dense-quality-stage-v0/result.json').read_text())['status']=='complete'
with tarfile.open(a) as t:
 for e in json.loads(t.extractfile('migration-manifest.json').read())['files']:
  n=e['path'];assert not Path(n).is_absolute() and '..' not in Path(n).parts
  m=t.getmember(n);assert m.isfile();raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes']
  f=r/n;f.parent.mkdir(parents=True,exist_ok=True)
  if f.exists():assert f.read_bytes()==raw,n
  else:f.write_bytes(raw)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not (r/'results/full-step-cost-stage-v0').exists()
assert not (r/'logs/full-step-cost-controller-v0.pid').exists()
with (r/'logs/full-step-cost-controller-v0.log').open('w') as f:
 p=subprocess.Popen([PYTHON,'-u',str(r/'scripts/run_full_step_cost_stage_v0.py')],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(r/'logs/full-step-cost-controller-v0.pid').write_text(str(p.pid))
print(json.dumps({'pid':p.pid,'status':'launched'}))
'''.replace('ROOT',repr(root)).replace('HASH',repr(proof['sha256'])).replace('PYTHON',repr(c['python']))
        # No retry after an uncertain launch; a human-visible failure requires inspection.
        p=subprocess.run(ssh+['python3','-'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=45);assert p.returncode==0,p.stderr[-2000:]
        launch=json.loads(p.stdout);save(R/'logs/full-step-cost-launch-v0.json',launch)
        c=load(R/'logs/control-state.json');c.update(status='full_step_cost_v0_running',updated_utc=utc(),active_benchmark_jobs=1,active_evaluation_jobs=0)
        c['full_step_cost']=dict(status='running',controller_pid=launch['pid'],protocol='provenance/full-step-cost-protocol-v0.json',planned_diagnostic_updates=64,collector='scripts/collect_full_step_cost_v0.py');save(R/'logs/control-state.json',c)
        with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：完整训练步成本诊断启动\n\n计划两种子、各两个分支、每分支16次更新，共64诊断更新。protocol已冻结，不是正式质量训练；主成本包括优化器和输入搬运。\n')
        (R/'STATE.md').write_text('# 当前状态：统一密集NLL已核验，完整训练步成本运行中\n\n'+utc()+'。原立项不变：稀疏继续预训练省成本、质量接近，不要求推理加速。\n\n统一NLL报告docs/common-dense-quality-results-2026-09-16.md。正在运行full-step-cost-v0，计划64诊断更新；不要重复启动。控制器/采集器见logs/control-state.json和logs/full-step-cost-v0-local-collector.json。\n\n已核验累计5760科学更新、235诊断更新、31786任务预测；在途诊断不计入已完成。\n',encoding='utf-8')
        for n in ['STATE.md','TIMELINE.md','logs/control-state.json']:
            subprocess.run(scp+[str(R/n),host+':'+root+'/'+n],check=True,timeout=45)
        save(state,dict(status='launched_collecting',utc=utc(),launch=launch))
        subprocess.run([sys.executable,'-u',str(R/'scripts/collect_full_step_cost_v0.py')],cwd=R,check=True)
        save(state,dict(status='complete_verified',utc=utc()))
    except Exception:
        save(state,dict(status='failed_needs_inspection',utc=utc(),error=traceback.format_exc(),no_retry_or_duplicate_launch=True));raise
if __name__=='__main__':main()

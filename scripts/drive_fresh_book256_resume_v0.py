"""Wait for the single environment restoration, then launch the frozen continuation once."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 c=json.loads((R/'logs/cloud-connection-current.json').read_text());assert c['pod_id']=='xc2y1iadcrgy31';opts=['-i',c['ssh_key_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15'];host=c['ssh_user']+'@'+c['ssh_host'];ssh=['ssh',*opts,'-p',str(c['ssh_port']),host];scp=['scp',*opts,'-P',str(c['ssh_port'])];state=R/'logs/fresh-book256-resume-driver-v0.json';start=time.monotonic()
 def active():assert not json.loads((R/'logs/control-state.json').read_text(encoding='utf-8')).get('user_stop_requested'), 'User requested stop'
 def remote(code):
  q=subprocess.run(ssh+['python3','-'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=45);assert q.returncode==0,q.stderr[-1500:];return json.loads(q.stdout)
 try:
  while time.monotonic()-start<2100:
   active();v=remote('from pathlib import Path\nimport json\nr=Path('+repr(c['root'])+')\nprint((r/"logs/resume-environment-20260917-v0.json").read_text())');save(state,dict(status='waiting_environment',utc=utc(),environment=v))
   if v['status']=='complete':break
   assert v['status']!='failed',v.get('error');time.sleep(30)
  else:raise RuntimeError('Environment wait cap reached; inspect existing restoration, do not relaunch')
  save(R/'logs/new-pod-resume-environment-20260917.json',v);assert v['python']==c['python'] and v['model_hashes_verified'] and v['environment']['gpu']=='NVIDIA GeForce RTX 4090'
  p=json.loads((R/'provenance/fresh-book256-resume-protocol-v0.json').read_text());files={j['path']:j['sha256'] for j in p['jobs']};files.update({j['training_result']:j['training_result_sha256'] for j in p['jobs']})
  code='from pathlib import Path\nimport json,hashlib,subprocess\nr=Path('+repr(c['root'])+')\nfiles='+repr(files)+'\nfor n,h in files.items():\n assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n\nassert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()\nprint(json.dumps(dict(status="verified",files=len(files))))';pre=remote(code);save(R/'logs/fresh-book256-resume-dependency-preflight-v0.json',pre)
  active();save(state,dict(status='launch_starting_do_not_retry',utc=utc()))
  q=subprocess.run([sys.executable,str(R/'scripts/launch_frozen_stage.py'),'fresh-book256-resume'],cwd=R,capture_output=True,text=True,encoding='utf-8',timeout=180);(R/'logs/fresh-book256-resume-launch-driver-v0.log').write_text(q.stdout+q.stderr,encoding='utf-8');assert q.returncode==0,q.stderr[-1500:];launch=json.loads(q.stdout.strip());assert launch['status']=='launched'
  x=json.loads((R/'logs/control-state.json').read_text(encoding='utf-8'));x.update(status='fresh_book256_resume_v0_running',updated_utc=utc(),active_evaluation_jobs=3,automation_status='ACTIVE',next_action='Monitor existing fresh-book256-resume controller and driver-managed collector; do not duplicate launch. Three remaining128step conditions only,321 carried records plus2766new tasks and2diagnostic replays. Original three contrasts and margins unchanged. After combined audit, assess training-cost evidence and prior-art limits before next bounded experiment.')
  x['fresh_book256_resume']=dict(status='running',pid=launch['pid'],collector='driver_managed',protocol='provenance/fresh-book256-resume-protocol-v0.json',expected_new_predictions=2766,expected_diagnostic_replays=2,maximum_seconds=3600,optimizer_updates=0);save(R/'logs/control-state.json',x)
  (R/'STATE.md').write_text('# 当前状态：跨Pod断点续跑已启动\n\n'+utc()+'\n\n用户已恢复持续推进。新Pod xc2y1iadcrgy31，4090，精确环境及模型哈希通过；连接见logs/cloud-connection-current.json。控制器'+str(launch['pid'])+'，采集由drive_fresh_book256_resume_v0.py管理，禁止重复启动。\n\n保留6组完成结果和第7组321条；当前3个剩余条件共新增2766预测，另做2条保存预测重放，需logits误差<=1e-6；各检查点4个校准NLL误差<=1e-6。原始数据、比较、5pp容差不变。新冻结协议7d0d7f682dad90ad06b7923202853ae83d1edec30d3ccb440549e8e1e11b304e。\n\n最多1小时；按此前$0.74/h估计上限$0.74，不含准备闲置存储，新Pod费率API查询不可用故不是已核实账单。0新训练更新，已有累计6784科学更新、299诊断更新、53382记录预测，完成后仅计入新增和诊断重放。\n\n继续监控和联合审计；不把已知QK恢复包装为新方法，不要求推理加速。heartbeat ACTIVE。\n',encoding='utf-8')
  with (R/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write('\n\n## '+utc()+'：新Pod环境核验通过，断点续跑启动\n\n控制器'+str(launch['pid'])+'，原精确torch2.8.0+cu128、transformers4.57.6、扩展SHA一致；全模型哈希通过，未改检查点或容差。只补3个剩余条件，其中321行复用，2766新预测、2诊断重放、12校准NLL，0更新。启动协议7d0d7f682dad90ad06b7923202853ae83d1edec30d3ccb440549e8e1e11b304e；启动包ac7d52f2136e757326172aef55820aab3dc6ec0faba57c7f49632f05afd329d6。保留环境修复日志、原中断和新UTC，1小时硬上限。\n')
  for n in ['STATE.md','TIMELINE.md','logs/control-state.json']:subprocess.run(scp+[str(R/n),host+':'+c['root']+'/'+n],check=True,timeout=45)
  save(state,dict(status='collecting',utc=utc(),controller=launch));print(json.dumps(dict(status='launched_and_collecting',pid=launch['pid'])),flush=True)
  subprocess.run([sys.executable,str(R/'scripts/collect_fresh_book256_resume_v0.py')],cwd=R,check=True)
  save(state,dict(status='complete_verified',utc=utc()))
 except Exception:
  save(state,dict(status='needs_inspection_do_not_relaunch',utc=utc(),error=traceback.format_exc()));raise
if __name__=='__main__':main()

"""Verify hashes, current checkpoint predictions and restoration without training."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def file_sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main(a):
    start=time.perf_counter();manifest=ROOT/'MIGRATION_MANIFEST.json';hashes=0
    if manifest.exists():
        for entry in json.loads(manifest.read_text(encoding='utf-8'))['files']:
            p=(ROOT/entry['path']).resolve();p.relative_to(ROOT.resolve());assert file_sha(p)==entry['sha256'],entry['path'];hashes+=1
    else:
        # Local source check before an archive exists: recheck all three result manifests.
        schedule='schedule-screen-cloud-v0' if (ROOT/'results/schedule-screen-cloud-v0').exists() else 'schedule-screen-v0'
        for base in [schedule,'frozen-router-capacity-v0','joint-token-router-v0']:
            folder=ROOT/'results'/base
            for row in json.loads((folder/'manifest.json').read_text()):assert file_sha(folder/row['path'])==row['sha256'],row['path'];hashes+=1
    if a.hash_only:print(json.dumps(dict(files_verified=hashes)));return
    sys.path.insert(0,str(ROOT/'src'));import torch
    from resume_joint_token_router import restore_checkpoint
    from run_frozen_router import evaluate,save,now,weights_sha
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.device=='cuda':
        assert torch.cuda.is_available(),'CUDA unavailable'
        # Actual operation catches a wheel that does not support the new GPU architecture.
        z=torch.randn(16,16,device='cuda');assert torch.isfinite(z@z).all()
    data=torch.load(ROOT/'results/joint-token-router-v0/data-and-initialization.pt',map_location='cpu',weights_only=True);rows=[]
    for name in ['exact_r16_shadow','learned_r16','learned_r64']:
        folder=ROOT/'results/joint-token-router-v0'/name;ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert ck['steps']==12520 and ck['epoch']==39
        model,ix,route,opt,iopt,scheduler=restore_checkpoint(ck,a.device);model.eval();route.collect_aux=False
        expected=json.loads((folder/'result.json').read_text());assert weights_sha(model.state_dict())==expected['final_backbone_hash']
        for key in ['fresh','swapped']:
            actual=evaluate(model,data[key],a.device);assert actual['predictions']==expected[key]['predictions'],(name,key,'prediction mismatch; inspect numerical drift before training')
        rows.append(dict(method=name,completed_epochs=40,global_step=12520,main_optimizer_states=len(opt.state),indexer_optimizer_states=len(iopt.state),main_lr=opt.param_groups[0]['lr'],scheduler_epoch=scheduler.last_epoch,predictions_verified=5120))
        route.restore()
    report=dict(utc=now(),device=a.device,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,torch=str(torch.__version__),files_verified=hashes,rows=rows,checkpoint_predictions_verified=15360,scientific_optimizer_updates=0,technical_optimizer_updates=0,wall_seconds=time.perf_counter()-start)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);save(a.output,report)
    print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--hash-only',action='store_true');p.add_argument('--output',type=Path);main(p.parse_args())

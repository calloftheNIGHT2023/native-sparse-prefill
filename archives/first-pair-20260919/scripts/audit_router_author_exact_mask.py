"""Frozen full-QK selection upper reference; dense selection cost, no updates."""
import json,math,shutil,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from router_author_control import make_model,evaluate
from frozen_routing import TokenIndexer,FrozenRouter
from run_frozen_router import now,save,sha,weights_sha

@torch.no_grad()
def main():
    torch.set_num_threads(4);src=ROOT/'results/router-author-sparse-v0';old=ROOT/'results/router-author-control-v0';out=ROOT/'results/router-author-exact-mask-v0';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py');shutil.copy2(ROOT/'docs/router-author-exact-mask-2026-09-14.md',out/'plan.md');ck=torch.load(old/'lr2/checkpoint.pt',map_location='cpu',weights_only=False);data=torch.load(src/'evaluation-data.pt',map_location='cpu',weights_only=True);reference=json.loads((src/'dense-reference.json').read_text(encoding='utf-8'));rows=[]
    for condition in ['dense','exact_top8_both','exact_top8_second']:
        m=make_model(ck['config'],ck['model']);m.eval();before=weights_sha(m.state_dict());route=None;score_errors=[]
        if condition!='dense':
            ix=torch.nn.ModuleList([TokenIndexer(128,128) for _ in range(2)])
            for index,layer in zip(ix,m.backbone.layers):
                index.initialize(layer.sequence_mixer,'copy');h=torch.randn(2,16,128,generator=torch.Generator().manual_seed(2026091629));q,k,_=layer.sequence_mixer.Wqkv(h).chunk(3,-1);expected=q@k.transpose(-1,-2)/math.sqrt(128);actual=index(h);torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5);score_errors.append(float((actual-expected).abs().max()))
            route=FrozenRouter(m,ix)
            if condition=='exact_top8_second':route.exact_layers={0}
        row=dict(condition=condition,score_copy_max_errors=score_errors)
        for key in ['test','swapped','noisy']:
            row[key]=evaluate(m,data[key],'cpu',batch=32)
            if condition=='dense':assert row[key]['predictions']==reference[key]['predictions']
        if route:route.restore()
        assert weights_sha(m.state_dict())==before;rows.append(row)
    result=dict(utc=now(),status='complete',optimizer_updates=0,conditions=rows,checkpoint_sha256=sha(old/'lr2/checkpoint.pt'),scope='Frozen trained backbone and exact full-QK selection. No learned low-rank training claim and no computational speedup.');save(out/'result.json',result);save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(dict(utc=result['utc'],optimizer_updates=0,conditions=[dict(condition=r['condition'],test=r['test']['accuracy'],swapped=r['swapped']['accuracy'],noisy=r['noisy']['accuracy']) for r in rows])))

if __name__=='__main__':main()

"""Zero-update post-fit source coverage and fixed-mask reachability diagnostics."""
import json,shutil,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_frozen_router import now,save,sha,weights_sha
from router_pressure_models import load_checkpoint,fixed_mask
from frozen_routing import select

@torch.no_grad()
def main():
    torch.set_num_threads(4);timer=time.perf_counter();src=ROOT/'results/router-pressure-v0';out=ROOT/'results/router-pressure-diagnostics-v0';out.mkdir(exist_ok=False)
    shutil.copy2(__file__,out/'source.py');shutil.copy2(ROOT/'docs/router-pressure-diagnostics-2026-09-14.md',out/'diagnostics-plan.md')
    result=json.loads((src/'result.json').read_text());data=torch.load(ROOT/'data/router-pressure-kv16-v0/data.pt',map_location='cpu',weights_only=True)
    records=[];diagnostic_answers=0
    for run in result['runs']:
        name=run['method'];ck=torch.load(src/name/'checkpoint.pt',map_location='cpu',weights_only=False);model,ix,route=load_checkpoint(ck,'cpu');before=weights_sha(model.state_dict());masks={};hooks=[]
        def make_hook(i):
            def hook(module,inputs):
                x=inputs[0];b,n,_=x.shape
                if name=='dense':mask=torch.ones(n,n,dtype=torch.bool).tril().expand(b,n,n)
                elif name=='fixed_values6':mask=fixed_mask(n,'cpu').expand(b,n,n)
                else:mask=select(ix[i](x))
                masks[i]=mask
            return hook
        for i,layer in enumerate(model.backbone.layers):hooks.append(layer.sequence_mixer.register_forward_pre_hook(make_hook(i)))
        covered=[0,0];answers=0;predictions=[];candidate_hits=0;source_correct=[0]*16;source_total=[0]*16;predicted_source_histogram=[0]*16;conditional=[dict(covered_correct=0,uncovered_correct=0,covered_total=0,uncovered_total=0) for _ in range(2)]
        for first in range(0,1024,32):
            x=data['test']['inputs'][first:first+32];y=data['test']['labels'][first:first+32];logits=model(x);b,q=torch.where(y!=-100);pred=logits.argmax(-1)[b,q];correct=pred==y[b,q];predictions.extend(pred.tolist())
            source=(x[b,:32:2]==x[b,q,None]).to(torch.int32).argmax(-1)*2+1
            assert torch.equal(x[b,source],y[b,q]);answers+=len(b)
            hit=x[b,1:32:2]==pred[:,None];candidate_hits+=int(hit.any(-1).sum())
            for j in range(16):
                target_is_j=source==2*j+1;source_total[j]+=int(target_is_j.sum());source_correct[j]+=int((target_is_j&correct).sum());predicted_source_histogram[j]+=int(hit[:,j].sum())
            for i in range(2):
                c=masks[i][b,q,source];covered[i]+=int(c.sum());conditional[i]['covered_correct']+=int((c&correct).sum());conditional[i]['uncovered_correct']+=int((~c&correct).sum());conditional[i]['covered_total']+=int(c.sum());conditional[i]['uncovered_total']+=int((~c).sum())
        assert predictions==run['test']['predictions'];diagnostic_answers+=answers
        records.append(dict(method=name,answers=answers,source_value_selected=[c/answers for c in covered],conditional=conditional,prediction_in_source_value_bank=candidate_hits/answers,correct_by_source_record=source_correct,total_by_source_record=source_total,predicted_source_record_histogram=predicted_source_histogram))
        for h in hooks:h.remove()
        if name=='fixed_values6':
            # Self is already present: the product includes both attention layers and residual paths.
            mask=fixed_mask(128,'cpu');assert mask.diag().all();reachable=(mask.float()@mask.float())>0
            details=[];same_logits=0;max_error=0.;sub_count=0;pair_correct=0;changes=0
            for first in range(0,1024,32):
                x=data['test']['inputs'][first:first+32];z=data['swapped']['inputs'][first:first+32]
                b=torch.arange(len(x));spec=torch.tensor(data['swap_positions'][first:first+32]);q,source,alt=spec.T
                assert torch.equal(x[b,q],z[b,q]);a=model(x)[b,q];v=model(z)[b,q];p=a.argmax(-1);sp=v.argmax(-1)
                eligible=~reachable[q,source]&~reachable[q,alt];diff=(a-v).abs().amax(-1);count=int(eligible.sum());sub_count+=count
                if count:
                    torch.testing.assert_close(a[eligible],v[eligible],rtol=0,atol=0)
                    same_logits+=int((diff[eligible]==0).sum());max_error=max(max_error,float(diff[eligible].max()))
                    assert torch.equal(p[eligible],sp[eligible]);pair_correct+=int(((p==x[b,source]).long()+(sp==z[b,source]).long())[eligible].sum())
                changes+=int((p!=sp).sum())
                for j in range(len(x)):details.append(dict(row=first+j,query=int(q[j]),source=int(source[j]),alternative=int(alt[j]),both_outside_receptive_field=bool(eligible[j]),logit_max_error=float(diff[j]),original_prediction=int(p[j]),swapped_prediction=int(sp[j])))
            diagnostic_answers+=2048;assert pair_correct<=sub_count
            save(out/'fixed-paired-rows.json',details);save(out/'fixed-reachability.json',dict(rows=1024,both_changed_sources_unreachable=sub_count,identical_logits=same_logits,max_logit_error_unreachable=max_error,unreachable_pair_correct_answers=pair_correct,unreachable_pair_answers=2*sub_count,all_prediction_changes=changes,optimizer_updates=0,scope='Architecture-specific fixed-mask correctness check, not a learned-routing necessity theorem.'))
        route.restore();assert before==weights_sha(model.state_dict())
    summary=dict(status='complete',utc=now(),optimizer_updates=0,diagnostic_answers=diagnostic_answers,coverage=records,fixed_diagnostic_executed=any(r['method']=='fixed_values6' for r in result['runs']),wall_seconds=time.perf_counter()-timer,scope='Descriptive final-checkpoint source selection; coverage is neither attention importance nor causal necessity. Learned selectors still score every token.')
    save(out/'summary.json',summary);save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(summary))

if __name__=='__main__':main()

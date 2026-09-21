"""Investigate, without training, a failed frozen-logit migration tolerance."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from router_author_control import make_model,selected_logits
from zoology_sparse_schedule import install
from chunked_topk_attention import install_chunked,select_causal_topk,ChunkedTopKAttention

torch.set_num_threads(4)
out=ROOT/'results/chunked-topk-numerics-v0';out.mkdir(parents=True,exist_ok=False)
cp=ROOT/'results/router-author-falsification-lowlr-v0/checkpoint.pt'
c=torch.load(cp,map_location='cpu',weights_only=False)
data=torch.load(ROOT/'results/router-author-falsification-evaluation-v0/evaluation-data.pt',map_location='cpu',weights_only=True)['noisy']
a=make_model(c['config'],c['model']);b=make_model(c['config'],c['model']);install(a,'native');install_chunked(b)
a.eval();b.eval();capture={}
def hook(name):
    def f(module,inputs,output):capture[name]=(inputs[0].detach().clone(),output.detach().clone())
    return f
for side,model in [('old',a),('new',b)]:
    for i,layer in enumerate(model.backbone.layers):layer.sequence_mixer.inner_attn.register_forward_hook(hook((side,i)))
reports=[]
with torch.no_grad():
    for first in range(0,1024,16):
        x=data['inputs'][first:first+16];y=data['labels'][first:first+16]
        la,target=selected_logits(a,x,y);lb,_=selected_logits(b,x,y)
        if torch.allclose(la,lb,rtol=1e-4,atol=1e-4):continue
        report=dict(first_row=first,max_abs_logit_difference=float((la-lb).abs().max()),
                    logit_entries_over_tolerance=int((~torch.isclose(la,lb,rtol=1e-4,atol=1e-4)).sum()),
                    answers_equal=bool(torch.equal(la.argmax(-1),lb.argmax(-1))),layers=[])
        for i in range(2):
            old_input,old_output=capture['old',i];new_input,new_output=capture['new',i]
            q,k,v=[z.permute(0,2,1,3).reshape(16,256,128) for z in old_input.unbind(2)]
            ids=select_causal_topk(q,k)
            mask=torch.zeros(16,256,256,dtype=torch.bool)
            for g in range(16):
                for row in range(256):mask[g,row,ids[g,row][ids[g,row]>=0]]=True
            old_mask=a.backbone.layers[i].sequence_mixer.inner_attn.last_mask[:,0]
            same_input=ChunkedTopKAttention(dropout=0).eval()(old_input)
            q2,k2,v2=[z.permute(0,2,1,3).reshape(16,256,128) for z in new_input.unbind(2)]
            ids2=select_causal_topk(q2,k2)
            mask2=torch.zeros_like(mask)
            for g in range(16):
                for row in range(256):mask2[g,row,ids2[g,row][ids2[g,row]>=0]]=True
            changed=(mask2!=old_mask).any(-1)
            margins=[]
            for g,row in torch.nonzero(changed).tolist():
                if row>=8:
                    scores=(q[g,row]@(k[g,:row-1]/128**.5).T)
                    top=scores.topk(7).values
                    margins.append(dict(row=first+g,query=row,old_boundary_gap=float(top[5]-top[6]),
                                        changed_edges=int((mask2[g,row]!=old_mask[g,row]).sum())))
            report['layers'].append(dict(layer=i,input_max_abs_difference=float((old_input-new_input).abs().max()),
                output_max_abs_difference=float((old_output-new_output).abs().max()),
                same_input_chunk_output_max_abs_difference=float((old_output-same_input).abs().max()),
                same_input_selected_edges_different=int((mask!=old_mask).sum()),
                actual_selected_edges_different=int((mask2!=old_mask).sum()),changed_query_margins=margins))
        reports.append(report);print(json.dumps(report),flush=True)
        break
(out/'diagnostic.json').write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
    optimizer_updates=0,reports=reports),indent=2),encoding='utf-8')

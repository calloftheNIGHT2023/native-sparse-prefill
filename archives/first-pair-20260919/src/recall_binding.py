"""Position-only predecessor attention as an explicit mechanistic control."""
import torch

def install_binding_control(model,cfg):
    if cfg.get('binding_control')!='predecessor_first_layer':return
    n=cfg['sequence_length'];q=torch.arange(n)
    allowed=torch.zeros((n,n),dtype=torch.bool);allowed[q,(q-1).clamp_min(0)]=True
    model.gpt_neox.layers[0].attention.register_buffer('_binding_allowed',allowed,persistent=False)
    def restrict(module,args,kwargs):
        h=args[0];length=h.shape[1]
        assert length==n and kwargs.get('layer_past') is None,'This CPU diagnostic does not support decoding/cache'
        m=module._binding_allowed[None,None]
        existing=kwargs.get('attention_mask')
        if existing is not None:m=m & (existing==0)
        assert m.any(-1).all()
        kwargs['attention_mask']=torch.zeros_like(m,dtype=h.dtype).masked_fill(~m,torch.finfo(h.dtype).min)
        return args,kwargs
    model.gpt_neox.layers[0].attention.register_forward_pre_hook(restrict,with_kwargs=True)

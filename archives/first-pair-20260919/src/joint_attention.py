"""Differentiable CPU reference for joint LM/indexer training, not a fast kernel.

All main Q/K/V projections receive LM gradients. The detached indexer receives
only KL gradients. This is a pure GPTNeoX proxy, not the Qwen hybrid architecture.
"""
import types
import torch
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from sparse_reference import make_layout, move_layout, select_blocks, expand_blocks, subset_kl
from indexer_calibration import make_indexer
from trace_math import target_from_logits
from gathered_core import gathered_attention
from routing_rules import route_blocks


class JointAttention:
    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg
        device=next(model.parameters()).device
        self.layout = move_layout(make_layout([0] * cfg['sequence_length'], cfg['block_size']),device)
        self.indexers = torch.nn.ModuleList([
            make_indexer(model.config.hidden_size, cfg)
            for _ in model.gpt_neox.layers]).to(device)
        self.mode = 'dense'
        self.collect_aux = False
        self.losses, self.stats = [], []
        self.originals = []
        for layer_id, layer in enumerate(model.gpt_neox.layers):
            self.originals.append(layer.attention.forward)
            layer.attention.forward = types.MethodType(self.forward_for(layer_id), layer.attention)

    def reset(self, mode, collect_aux):
        self.mode, self.collect_aux = mode, collect_aux
        self.losses, self.stats = [], []

    def restore(self):
        for layer, original in zip(self.model.gpt_neox.layers, self.originals):
            layer.attention.forward = original

    def forward_for(self, layer_id):
        def forward(module, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
            if hidden_states.shape[:2] != (1, self.cfg['sequence_length']):
                raise ValueError('Reference requires batch 1, fixed full-length single document')
            if kwargs.get('layer_past') is not None or kwargs.get('head_mask') is not None:
                raise ValueError('KV cache and head masking unsupported')
            if module.attention_dropout != 0 or self.model.config.hidden_dropout != 0:
                raise ValueError('This paired reference requires zero dropout')
            n, width = hidden_states.shape[1:]
            qkv = module.query_key_value(hidden_states).reshape(1,n,-1,3*module.head_size).transpose(1,2)
            q,k,v = qkv.chunk(3,-1)
            q,k = apply_rotary_pos_emb(q,k,*position_embeddings)
            scores = None
            if self.mode == 'sparse' or self.collect_aux:
                scores = self.indexers[layer_id](hidden_states[0],self.layout)
            chosen = (route_blocks(scores,self.layout,self.cfg['selected_blocks'],self.cfg.get('selection_rule','learned'))
                      if self.mode == 'sparse' else self.layout.visible_blocks)
            mask = expand_blocks(chosen,self.layout)
            if self.mode == 'self_only':
                if self.collect_aux: raise ValueError('Self-only is an evaluation control')
                mask = torch.eye(n,dtype=torch.bool,device=hidden_states.device)
            if self.mode == 'sparse' and self.cfg.get('core_backend') == 'gathered':
                gathered,teacher,_ = gathered_attention(q[0],k[0],v[0],chosen,self.layout)
                output = gathered.transpose(0,1).reshape(1,n,width)
                probs = None  # avoid recreating a dense attention tensor
            else:
                logits = (q @ k.transpose(-2,-1)) * module.scaling
                # Match transformers GPTNeoX eager's float32 softmax exactly.
                probs = logits.masked_fill(~mask[None,None],torch.finfo(logits.dtype).min).softmax(-1,dtype=torch.float32).to(v.dtype)
                output = (probs @ v).transpose(1,2).reshape(1,n,width)
                if self.collect_aux:
                    with torch.no_grad():
                        teacher = target_from_logits(logits[0].detach(), chosen, self.layout)
            if self.collect_aux:
                self.losses.append(subset_kl(scores,teacher,chosen))
            late = self.cfg['late_query_start']
            self.stats.append({'layer':layer_id,
                'retained_pairs':int(mask.sum()),
                'late_retained_fraction':float(mask[late:].sum()/self.layout.causal_mask[late:].sum())})
            return module.dense(output), probs
        return forward

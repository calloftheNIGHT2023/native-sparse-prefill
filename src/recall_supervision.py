"""Known multi-query recall supervision as a trainability control, not a method."""
import torch

def add_training_queries(inputs,meta,cfg):
    """Change only query tokens; never put target answer tokens in query region."""
    assert cfg.get('query_format')=='key_last'
    positions=cfg['training_query_positions']
    assert len(positions)==cfg['records'] and positions[-1]==cfg['sequence_length']-1
    assert min(positions)-2>=cfg['evidence_end_exclusive']
    target=meta['value_position']-2
    records=sorted(p for p in meta['record_positions'] if p!=target)+[target]
    out=inputs.clone(); labels=[]
    for q,p in zip(positions,records):
        assert p+3<q-2
        out[:,q-2]=cfg['question_marker'];out[:,q-1]=cfg['separator'];out[:,q]=inputs[:,p+1]
        labels.append(inputs[:,p+2])
    # All supervised values are already in the earlier record region.
    labels=torch.stack(labels,dim=1)
    assert torch.equal(out[:,-3:],inputs[:,-3:])
    return out,labels

def query_loss(model,inputs,labels,cfg):
    hidden=model.gpt_neox(inputs,use_cache=False).last_hidden_state
    positions=cfg['training_query_positions']
    if cfg['supervision']=='last_only':
        logits=model.embed_out(hidden[:,-1]); targets=labels[:,-1]
    elif cfg['supervision']=='all_queries':
        logits=model.embed_out(hidden[:,positions]).reshape(-1,cfg['vocab_size']);targets=labels.reshape(-1)
    else: raise ValueError(cfg['supervision'])
    return torch.nn.functional.cross_entropy(logits,targets)

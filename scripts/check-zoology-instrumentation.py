"""One technical optimizer step per path, to verify the logging wrapper is inert."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from zoology_entry import configuration,LanguageModel,Trainer,set_determinism
from run_zoology_baseline import LoggedTrainer,LocalLogger,tensor_hash,save

def main():
    out=ROOT/'results/zoology-instrumentation-check-v0';out.mkdir(exist_ok=False);torch.set_num_threads(4);set_determinism(123)
    cfg=configuration();model=LanguageModel(cfg.model);a=copy.deepcopy(model);b=copy.deepcopy(model)
    data=torch.load(ROOT/'results/zoology-basic-cpu-v1/train-data.pt',weights_only=True)
    batches=[(data['inputs'][:2],data['labels'][:2],[{},{}])]
    class NullLogger:
        def log(self,metrics):pass
    ordinary=Trainer(model=a,train_dataloader=batches,test_dataloader=batches,device='cpu',logger=NullLogger())
    logged=LoggedTrainer(model=b,train_dataloader=batches,test_dataloader=batches,device='cpu',logger=LocalLogger(out),out=out,cfg=cfg.model_dump(serialize_as_any=True),max_seconds=60)
    for trainer in [ordinary,logged]:
        trainer.loss_fn=torch.nn.CrossEntropyLoss();trainer.optimizer=torch.optim.AdamW(trainer.model.parameters(),lr=.001,weight_decay=.1)
    rng=torch.get_rng_state();ordinary.train_epoch(0);after_ordinary=torch.get_rng_state()
    torch.set_rng_state(rng);logged.train_epoch(0);after_logged=torch.get_rng_state()
    assert tensor_hash(a.state_dict())==tensor_hash(b.state_dict())
    assert torch.equal(after_ordinary,after_logged)
    result=dict(upstream_and_logged_weights_bitwise_equal=True,torch_rng_after_step_equal=True,
        loss=float(logged.last_grad_norm),technical_optimizer_steps=2,research_training_runs=0,
        scope='Two copies, same one batch, same CPU dropout RNG, upstream train_epoch versus wrapper; not a scientific training run.')
    result['gradient_norm']=result.pop('loss');save(out/'result.json',result);print(json.dumps(result))

if __name__=='__main__':main()

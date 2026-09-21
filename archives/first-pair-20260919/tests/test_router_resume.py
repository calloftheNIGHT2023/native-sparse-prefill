import copy,io,random,sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from resume_joint_token_router import restore_checkpoint,pack,train_step
from run_frozen_router import weights_sha

class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Gather/scatter backward can accumulate in different orders with CPU threads.
        torch.set_num_threads(1)
        cls.ck=torch.load(ROOT/'results/joint-token-router-v0/learned_r16/checkpoint.pt',map_location='cpu',weights_only=False)
        data=torch.load(ROOT/'results/joint-token-router-v0/data-and-initialization.pt',map_location='cpu',weights_only=True)
        cls.x=data['train']['inputs'][:2];cls.y=data['train']['labels'][:2]
    def test_parameters_optimizers_scheduler_and_rng_restore(self):
        model,ix,route,opt,iopt,sched=restore_checkpoint(copy.deepcopy(self.ck),'cpu')
        self.assertEqual(weights_sha(model.state_dict()),weights_sha(self.ck['model']))
        self.assertEqual(weights_sha(ix.state_dict()),weights_sha(self.ck['indexers']))
        self.assertEqual(opt.param_groups[0]['lr'],self.ck['main_optimizer']['param_groups'][0]['lr'])
        for k,v in self.ck['scheduler'].items():self.assertEqual(sched.state_dict()[k],v)
        for current,stored in [(opt.state_dict(),self.ck['main_optimizer']),(iopt.state_dict(),self.ck['indexer_optimizer'])]:
            for group,saved in zip(current['param_groups'],stored['param_groups']):
                for name,value in saved.items():self.assertEqual(group[name],value)
            for key,state in current['state'].items():
                for name,value in state.items():torch.testing.assert_close(value,stored['state'][key][name],rtol=0,atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(),self.ck['torch_rng']));self.assertEqual(random.getstate(),self.ck['python_rng']);self.assertTrue(np.array_equal(np.random.get_state()[1],self.ck['numpy_rng'][1]));route.restore()
    def test_two_steps_equal_save_restore_continuation(self):
        # Three disposable in-memory technical optimizer steps; no scientific artifact changes.
        m,ix,route,opt,iopt,sch=restore_checkpoint(copy.deepcopy(self.ck),'cpu')
        train_step(m,ix,route,opt,iopt,self.x,self.y);sch.step()
        mid=copy.deepcopy(pack(m,ix,opt,iopt,sch,self.ck,40,12521,'cpu'))
        expected_metrics=train_step(m,ix,route,opt,iopt,self.x,self.y);expected_main=weights_sha(m.state_dict());expected_ix=weights_sha(ix.state_dict());expected_rng=torch.get_rng_state().clone();route.restore()
        stream=io.BytesIO();torch.save(mid,stream);stream.seek(0);restored=torch.load(stream,map_location='cpu',weights_only=False)
        n,jx,r,nopt,jopt,ns=restore_checkpoint(restored,'cpu')
        actual_metrics=train_step(n,jx,r,nopt,jopt,self.x,self.y)
        self.assertEqual(actual_metrics,expected_metrics);self.assertEqual(weights_sha(n.state_dict()),expected_main);self.assertEqual(weights_sha(jx.state_dict()),expected_ix);self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng));self.assertEqual(ns.state_dict(),sch.state_dict());r.restore()

if __name__=='__main__':unittest.main()

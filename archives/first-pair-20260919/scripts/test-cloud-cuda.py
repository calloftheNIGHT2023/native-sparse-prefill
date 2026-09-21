import sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests')); sys.path.insert(0,str(ROOT/'src'))
from test_joint_attention import JointTests
from joint_attention import JointAttention
class CudaJointTests(JointTests):
    def setup_models(self):
        original,model,wrapper,x=super().setup_models()
        state=wrapper.indexers.state_dict(); cfg=wrapper.cfg; wrapper.restore()
        original=original.cuda(); model=model.cuda(); wrapper=JointAttention(model,cfg)
        wrapper.indexers.double(); wrapper.indexers.load_state_dict(state)
        return original,model,wrapper,x.cuda()
if __name__=='__main__':
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32=False
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CudaJointTests))
    sys.exit(not result.wasSuccessful())

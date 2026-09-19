import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_router_freshdata import generated,validate,swapped,tensor_hash,row_hashes

class FreshDataTests(unittest.TestCase):
    def test_generation_reproducible_and_rng_isolated(self):
        torch.manual_seed(55);np.random.seed(71);tr=torch.get_rng_state();nr=np.random.get_state()
        a=generated(2026091501,24);validate(a);self.assertTrue(torch.equal(tr,torch.get_rng_state()));after=np.random.get_state();self.assertEqual(nr[0],after[0]);self.assertTrue(np.array_equal(nr[1],after[1]));self.assertEqual(nr[2:],after[2:])
        self.assertEqual(tensor_hash(a),tensor_hash(generated(2026091501,24)));self.assertFalse(set(row_hashes(a))&set(row_hashes(generated(2026091502,24))))
    def test_swaps_keep_queries_and_change_correct_value(self):
        a=generated(2026091591,24);validate(a);b,positions=swapped(a)
        for i,(q,source,alt) in enumerate(positions):
            self.assertEqual(int(b['labels'][i,q]),int(a['inputs'][i,alt]));self.assertNotEqual(int(b['labels'][i,q]),int(a['labels'][i,q]));self.assertEqual(int((b['labels'][i]!=-100).sum()),1);self.assertTrue(torch.equal(a['inputs'][i,32:],b['inputs'][i,32:]))

if __name__=='__main__':unittest.main()

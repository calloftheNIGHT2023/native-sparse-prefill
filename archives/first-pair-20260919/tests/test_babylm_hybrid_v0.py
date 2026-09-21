"""CPU engineering checks only: no BabyLM data, GPU, or scientific training.

The expectations here are stated independently of the attention implementation.
Counts distinguish optimizer calls from forwards/backwards and scientific work.
"""
import copy
import io
import json
from dataclasses import replace
from pathlib import Path
import unittest

import torch

from src.babylm_hybrid.config import HybridConfig
from src.babylm_hybrid.model import build_model, parameter_counts


COUNTS = {"model_forward_calls": 0, "backward_calls": 0,
          "engineering_optimizer_steps": 0, "scientific_optimizer_steps": 0,
          "real_training_tokens": 0, "gpu_calls": 0}
MEASUREMENTS = {}


def forward(model, *args, **kwargs):
    COUNTS["model_forward_calls"] += 1
    return model(*args, **kwargs)


def backward(loss):
    COUNTS["backward_calls"] += 1
    loss.backward()


def step(optimizer):
    optimizer.step()
    COUNTS["engineering_optimizer_steps"] += 1


def is_indexer(name):
    return ".indexer." in name


def backbone_state(model):
    return {name: value for name, value in model.state_dict().items() if not is_indexer(name)}


def fixed_tokens(length=24, batch=1):
    return (torch.arange(length * batch).reshape(batch, length) * 7 + 11) % 97


def grad_present(model, indexer=False):
    return {n: p.grad for n, p in model.named_parameters() if is_indexer(n) == indexer}


class HybridEngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def assert_tensor_close(self, actual, expected, atol=2e-6, rtol=2e-5):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_01_shared_initialization_and_rng_isolation(self):
        cfg = HybridConfig()
        before = torch.get_rng_state().clone()
        dense = build_model(cfg, "dense", 731, 991)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        sparse = build_model(cfg, "sparse", 731, 991)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        sparse_other = build_model(cfg, "sparse", 731, 992)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        dense_state, sparse_state = backbone_state(dense), backbone_state(sparse)
        self.assertEqual(set(dense_state), set(sparse_state))
        self.assertTrue(all(torch.equal(dense_state[n], sparse_state[n]) for n in dense_state))
        self.assertTrue(all(torch.equal(v, backbone_state(sparse_other)[n]) for n, v in sparse_state.items()))
        ix = {n: p for n, p in sparse.named_parameters() if is_indexer(n)}
        other = dict(sparse_other.named_parameters())
        self.assertTrue(ix)
        self.assertTrue(any(not torch.equal(p, other[n]) for n, p in ix.items()))
        self.assertTrue(all(p.requires_grad for p in sparse.parameters()))
        MEASUREMENTS["tiny_dense_parameters"] = parameter_counts(dense)
        MEASUREMENTS["tiny_sparse_parameters"] = parameter_counts(sparse)
        MEASUREMENTS["paired_backbone_exact"] = True
        MEASUREMENTS["caller_cpu_rng_unchanged"] = True

    def test_02_full_support_dense_sparse_lm_equivalence(self):
        cfg = replace(HybridConfig(), selected_complete_blocks=64)
        dense = build_model(cfg, "dense", 731, 991)
        sparse = build_model(cfg, "sparse", 731, 991)
        ids = fixed_tokens(19, batch=2)
        seg = torch.tensor([[0]*7 + [1]*9 + [-1]*3, [0]*19])
        d, e = forward(dense, ids, seg), forward(sparse, ids, seg)
        self.assert_tensor_close(d.logits, e.logits)
        self.assert_tensor_close(d.lm_loss, e.lm_loss)
        backward(d.lm_loss)
        backward(e.lm_loss)
        dgrad, egrad = grad_present(dense), grad_present(sparse)
        self.assertEqual(set(dgrad), set(egrad))
        worst = 0.0
        for name in dgrad:
            self.assertIsNotNone(dgrad[name], name)
            self.assertIsNotNone(egrad[name], name)
            self.assert_tensor_close(dgrad[name], egrad[name], atol=3e-6, rtol=3e-5)
            worst = max(worst, (dgrad[name]-egrad[name]).abs().max().item())
        MEASUREMENTS["full_support"] = {
            "max_logit_abs_difference": (d.logits-e.logits).abs().max().item(),
            "max_backbone_gradient_abs_difference": worst,
            "valid_loss_queries": d.token_loss_count,
        }

    def test_03_sparse_gradient_contract(self):
        model = build_model(HybridConfig(), "sparse", 731, 991)
        ids, seg = fixed_tokens(28), torch.zeros(1, 28, dtype=torch.long)
        out = forward(model, ids, seg)
        self.assertTrue(torch.isfinite(out.aux_loss))
        self.assertGreater(out.aux_loss.item(), 0)
        backward(out.lm_loss)
        self.assertTrue(all(g is None or torch.count_nonzero(g) == 0 for g in grad_present(model, True).values()))
        self.assertTrue(any(g is not None and torch.count_nonzero(g) > 0 for g in grad_present(model).values()))
        model.zero_grad(set_to_none=True)
        out = forward(model, ids, seg)
        backward(out.aux_loss)
        self.assertTrue(all(g is None or torch.count_nonzero(g) == 0 for g in grad_present(model).values()))
        index_grads = grad_present(model, True)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in index_grads.values()))
        self.assertTrue(any(torch.count_nonzero(g) > 0 for g in index_grads.values()))
        MEASUREMENTS["sparse_gradient_contract"] = {
            "lm_to_indexer_zero": True, "auxiliary_to_backbone_zero": True,
            "auxiliary_loss": out.aux_loss.item(), "attention_stats": out.attention_stats,
        }

    def test_04_full_parameter_update_coverage(self):
        coverage = {}
        for mode in ("dense", "sparse"):
            with self.subTest(mode=mode):
                model = build_model(HybridConfig(), mode, 731, 991)
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
                out = forward(model, fixed_tokens(24), aux_weight=0.5 if mode == "sparse" else 0.0)
                backward(out.loss)
                no_grad = [n for n, p in model.named_parameters() if p.grad is None]
                self.assertEqual(no_grad, [], f"unused trainable parameters: {no_grad}")
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
                step(optimizer)
                groups = {"embedding": [], "gdn": [], "global": [], "ffn": [], "norm": [], "indexer": []}
                for n, p in model.named_parameters():
                    if is_indexer(n):
                        group = "indexer"
                    elif n.startswith("embedding."):
                        group = "embedding"
                    elif ".ffn." in n:
                        group = "ffn"
                    elif ".mixer.core." in n:
                        group = "gdn"
                    elif ".mixer." in n:
                        group = "global"
                    else:
                        group = "norm"
                    groups[group].append((n, bool(torch.count_nonzero(p.detach()-before[n]))))
                for group, parameters in groups.items():
                    if group == "indexer" and mode == "dense":
                        self.assertEqual(parameters, [])
                    else:
                        self.assertTrue(parameters, group)
                        self.assertTrue(any(changed for _, changed in parameters), group)
                coverage[mode] = {g: {"tensors": len(values), "changed_tensors": sum(c for _, c in values)}
                                  for g, values in groups.items()}
        MEASUREMENTS["optimizer_update_parameter_group_coverage"] = coverage

    def test_05_causal_future_isolation(self):
        for mode in ("dense", "sparse"):
            with self.subTest(mode=mode):
                model = build_model(HybridConfig(), mode, 731, 991)
                ids = fixed_tokens(24)
                changed = ids.clone()
                changed[:, 11:] = (changed[:, 11:]+37) % 97
                a, b = forward(model, ids), forward(model, changed)
                self.assert_tensor_close(a.logits[:, :11], b.logits[:, :11])

    def test_06_segments_repeated_labels_and_padding_isolation(self):
        for mode in ("dense", "sparse"):
            with self.subTest(mode=mode):
                model = build_model(HybridConfig(), mode, 731, 991)
                ids = fixed_tokens(24)
                seg = torch.tensor([[3]*8+[4]*4+[-1]*2+[3]*8+[-1]*2])
                changed = ids.clone()
                changed[:, :14] = (changed[:, :14]+23) % 97
                changed[:, 22:] = (changed[:, 22:]+17) % 97
                a, b = forward(model, ids, seg), forward(model, changed, seg)
                self.assert_tensor_close(a.logits[:, 14:22], b.logits[:, 14:22])
                standalone = forward(model, ids[:, 14:22], torch.zeros(1, 8, dtype=torch.long))
                self.assert_tensor_close(a.logits[:, 14:22], standalone.logits)
                self.assertEqual(a.token_loss_count, 7+3+7)
                self.assertEqual(torch.count_nonzero(a.logits[seg < 0]).item(), 0)
                changed_pad = ids.clone()
                changed_pad[seg < 0] = (changed_pad[seg < 0]+31) % 97
                pad_out = forward(model, changed_pad, seg)
                self.assert_tensor_close(a.logits[seg >= 0], pad_out.logits[seg >= 0])
                self.assert_tensor_close(a.lm_loss, pad_out.lm_loss)
                all_pad = forward(model, ids, torch.full_like(seg, -1))
                self.assertEqual(all_pad.token_loss_count, 0)
                self.assertEqual(all_pad.loss.item(), 0.0)
                self.assertTrue(torch.isfinite(all_pad.logits).all())

    def test_07_loss_query_mask_and_empty_targets(self):
        model = build_model(HybridConfig(), "sparse", 731, 991)
        ids, seg = fixed_tokens(12), torch.zeros(1, 12, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[:, [1, 6, 11]] = True  # final query has no next-token target
        out = forward(model, ids, seg, loss_mask=mask)
        self.assertEqual(out.token_loss_count, 2)
        expected = torch.nn.functional.cross_entropy(out.logits[:, [1, 6]].reshape(-1, 97), ids[:, [2, 7]].reshape(-1))
        self.assert_tensor_close(out.lm_loss, expected)
        empty = forward(model, ids[:, :1], seg[:, :1])
        self.assertEqual(empty.token_loss_count, 0)
        self.assertEqual(empty.loss.item(), 0)
        backward(empty.loss)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_08_checkpoint_optimizer_rng_cursor_replay(self):
        cfg = HybridConfig()
        model = build_model(cfg, "sparse", 731, 991)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        torch.manual_seed(1147)
        cursor = {"synthetic_batches": 0, "synthetic_token_positions": 0}

        def one_step(m, opt, where):
            batch = torch.randint(cfg.vocab_size, (1, 24))
            opt.zero_grad(set_to_none=True)
            out = forward(m, batch, aux_weight=0.5)
            backward(out.loss)
            step(opt)
            where["synthetic_batches"] += 1
            where["synthetic_token_positions"] += batch.numel()
            return batch.detach().clone(), out.logits.detach().clone(), out.loss.detach().clone()

        one_step(model, optimizer, cursor)
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "cpu_rng": torch.get_rng_state(), "data_cursor": copy.deepcopy(cursor),
                    "config": cfg.to_dict(), "mode": model.mode}, buffer)
        expected = one_step(model, optimizer, cursor)
        expected_rng = torch.get_rng_state().clone()
        expected_weights = {n: p.detach().clone() for n, p in model.named_parameters()}
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        buffer.seek(0)
        saved = torch.load(buffer, map_location="cpu", weights_only=True)
        replay = build_model(cfg, "sparse", 0, 0)
        replay.load_state_dict(saved["model"])
        replay_optimizer = torch.optim.AdamW(replay.parameters(), lr=1e-3, weight_decay=0.01)
        replay_optimizer.load_state_dict(saved["optimizer"])
        replay_cursor = saved["data_cursor"]
        torch.set_rng_state(saved["cpu_rng"])
        actual = one_step(replay, replay_optimizer, replay_cursor)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(replay_cursor, cursor)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for n, p in replay.named_parameters():
            self.assertTrue(torch.equal(p.detach(), expected_weights[n]), n)
        got_state = replay_optimizer.state_dict()
        self.assertEqual(expected_optimizer["param_groups"], got_state["param_groups"])
        for parameter, state in expected_optimizer["state"].items():
            for name, value in state.items():
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(value, got_state["state"][parameter][name]))
                else:
                    self.assertEqual(value, got_state["state"][parameter][name])
        MEASUREMENTS["checkpoint_replay"] = {
            "second_update_bitwise_exact": True,
            "includes_optimizer_rng_and_data_cursor": True,
            "replayed_synthetic_token_positions": 24,
            "checkpoint_bytes": buffer.getbuffer().nbytes,
        }

    def test_09_candidate_parameter_count_only(self):
        path = Path(__file__).resolve().parents[1] / "configs/babylm-native-sparse-v0.candidate.json"
        candidate = json.loads(path.read_text(encoding="utf-8-sig"))
        cfg = HybridConfig.from_candidate(candidate)
        # Real CPU construction checks constructor compatibility; no large-model forward.
        model = build_model(cfg, "sparse", 731, 991)
        counts = parameter_counts(model)
        self.assertEqual(counts["trainable"], counts["total"])
        self.assertGreater(counts["backbone"], 80_000_000)
        self.assertLess(counts["total"], 120_000_000)
        MEASUREMENTS["candidate_instantiated_cpu_parameters"] = counts
        MEASUREMENTS["candidate_forward_calls"] = 0
        del model


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Minimal text-causal scoring contract for the pinned BabyLM 2026 evaluator.

Accepts already normalized ``sentences``/``completions`` records. No dataset,
labels, remote weights, tokenizer fitting, accuracy selection or training here.
The official reader/aggregator may use HybridCausalAdapter directly; this file
does not claim to implement the whole official benchmark pipeline.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

UPSTREAM_COMMIT = "6f825c291e2c4c78ad33b1935fd64d45f52642dc"
TOKENIZER_SHA256 = "230b9d6993dcaf32f40cec2c44ac79d7d713213616ba2d8e45ad4a96e5a9dbe6"
LENGTH_NORMALIZED_TASKS = frozenset({"global_piqa_parallel", "global_piqa_nonparallel"})
SUPPORTED_TASKS = frozenset({"blimp", "ewok", "entity_tracking", "comps"}) | LENGTH_NORMALIZED_TASKS


class CausalScoringError(RuntimeError):
    """Failure retaining attempted/completed work; original cause is chained."""
    def __init__(self, message, *, counters, task, candidate_index):
        super().__init__(message)
        self.counters = dict(counters)
        self.task = task
        self.candidate_index = candidate_index


class FixedLocalTokenizer:
    """Processor-shaped wrapper for the already fixed, train-only BPE artifact."""
    def __init__(self, tokenizer_path):
        from tokenizers import Tokenizer
        path = Path(tokenizer_path)
        content = path.read_bytes()
        self.sha256 = hashlib.sha256(content).hexdigest()
        if self.sha256 != TOKENIZER_SHA256:
            raise ValueError("Tokenizer differs from the frozen training artifact")
        spec = json.loads(content)
        if spec.get("post_processor") is not None:
            raise ValueError("This contract requires the frozen no-postprocessor tokenizer")
        self.backend = Tokenizer.from_str(content.decode("utf-8"))
        self.pad_token_id = self.backend.token_to_id("<|pad|>")
        self.bos_token_id = self.backend.token_to_id("<|bos|>")
        self.eos_token_id = self.backend.token_to_id("<|eos|>")
        if (self.pad_token_id, self.bos_token_id, self.eos_token_id) != (0, 1, 2):
            raise ValueError("Frozen special token mapping changed")

    def __call__(self, text, return_offsets_mapping=True):
        if not isinstance(text, str) or not return_offsets_mapping:
            raise ValueError("Text and offsets are required")
        # Official causal processing delegates special tokens to the processor.
        # There is no postprocessor in this artifact: no BOS/EOS are inserted.
        encoded = self.backend.encode(text, add_special_tokens=True)
        return {"input_ids": encoded.ids, "attention_mask": encoded.attention_mask,
                "offset_mapping": encoded.offsets}


def prepare_record(record, processor, max_input_tokens=2048):
    """Equivalent to official text process_causal_sentences on the valid domain.

    The pinned implementation mistakes flat one-token IDs for a nested image
    batch, and empty tokens have no offset[0]. Fail clearly for <=1 token rather
    than silently invent BOS or scores. No truncation is performed.
    """
    if not isinstance(max_input_tokens, int) or max_input_tokens < 1:
        raise ValueError("max_input_tokens must be a positive integer")
    sentences, completions = record["sentences"], record["completions"]
    if not sentences or len(sentences) != len(completions):
        raise ValueError("Nonempty, aligned sentence/completion lists required")
    if record.get("image") is not None:
        raise ValueError("Only text-causal scoring is covered")
    result = {}
    for i, (sentence, completion) in enumerate(zip(sentences, completions)):
        if not isinstance(sentence, str) or not isinstance(completion, str) or not sentence.endswith(completion):
            raise ValueError("Completion must be an exact suffix of the full sentence")
        encoded = processor(text=sentence, return_offsets_mapping=True)
        ids, mask, offsets = encoded["input_ids"], encoded["attention_mask"], encoded["offset_mapping"]
        if len(ids) < 2 or any(not isinstance(t, int) for t in ids):
            raise ValueError("Official flat text contract requires at least two tokens")
        if len(ids) - 1 > max_input_tokens:
            raise ValueError("Sequence exceeds the declared input bound; no truncation allowed")
        if not (len(ids) == len(mask) == len(offsets)) or any(m != 1 for m in mask):
            raise ValueError("Processor must produce unpadded, aligned text tokens and offsets")
        # Text path has start_idx=0 in the official source. Keep its offset[0]
        # addition exactly, including the treatment of tokens overlapping suffix.
        start_char = len(sentence) - len(completion) + offsets[0][0]
        prefix = f"sentence_{i}"
        result[prefix + "_tokens"] = torch.tensor(ids, dtype=torch.long)
        result[prefix + "_attn_mask"] = torch.tensor(mask, dtype=torch.long)
        result[prefix + "_phrase_mask"] = torch.tensor([int(end > start_char) for _, end in offsets], dtype=torch.long)
        result[prefix + "_image"] = None
    return result


def collate_records(processed, pad_idx):
    """Official right padding followed by token/target shift, per candidate."""
    if not processed:
        raise ValueError("Empty batch")
    n = sum(key.endswith("_tokens") for key in processed[0])
    if any(sum(k.endswith("_tokens") for k in item) != n for item in processed):
        raise ValueError("Every record in a batch must have the same candidate count")
    result = {}
    for i in range(n):
        prefix = f"sentence_{i}"
        tokens = pad_sequence([p[prefix + "_tokens"] for p in processed], batch_first=True, padding_value=pad_idx)
        result[prefix + "_inputs"], result[prefix + "_targets"] = tokens[:, :-1], tokens[:, 1:]
        for name in ("attn_mask", "phrase_mask"):
            padded = pad_sequence([p[prefix + "_" + name] for p in processed], batch_first=True, padding_value=0)
            result[prefix + "_" + name] = padded[:, :-1] if name == "attn_mask" else padded[:, 1:]
    return result


class HybridCausalAdapter(nn.Module):
    """HF-call-shaped wrapper usable by the official causal compute function."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        if input_ids.ndim != 2 or not input_ids.shape[1] or attention_mask.shape != input_ids.shape:
            raise ValueError("Expected aligned nonempty [batch, time] inputs/mask")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("Attention mask must be binary")
        valid = attention_mask.bool()
        if not bool(valid[:, 0].all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("Only nonempty right-padded sequences are supported")
        segment_ids = torch.where(valid, torch.zeros_like(input_ids), -torch.ones_like(input_ids))
        output = self.model(input_ids, segment_ids=segment_ids, aux_weight=0.0)
        return {"logits": output.logits}


def score_records(model, processor, records, task, *, device="cpu", temperatures=(1.0,), max_input_tokens=2048):
    """Score one normalized batch without labels or temperature selection.

    Output scores are [temperature][record][candidate]; all temperatures must
    be fixed by the caller before evaluation. Default equals official scripts.
    This function uses no_grad and restores every module's original mode.
    """
    if task not in SUPPORTED_TASKS:
        raise ValueError("Unsupported task; reading/AoA/finetuning/multimodal are separate pipelines")
    temperatures = tuple(float(t) for t in temperatures)
    if not temperatures or len(set(temperatures)) != len(temperatures) or any(not math.isfinite(t) or t <= 0 for t in temperatures):
        raise ValueError("Temperatures must be unique, finite and positive")
    records = list(records)
    processed = [prepare_record(r, processor, max_input_tokens) for r in records]
    batch = collate_records(processed, processor.pad_token_id)
    candidates = len(records[0]["sentences"])
    adapter = HybridCausalAdapter(model)
    modes = [(module, module.training) for module in model.modules()]
    scores = [[] for _ in temperatures]
    counts = {"model_forward_attempts": 0, "model_forward_calls": 0,
              "backward_calls": 0, "optimizer_updates": 0,
              "records": len(records), "candidate_sequences": candidates * len(records),
              "source_tokens": sum(len(p[k]) for p in processed for k in p if k.endswith("_tokens")),
              "forward_input_tokens": 0, "padded_forward_positions": 0, "scored_target_tokens": 0,
              "submitted_forward_input_tokens": 0, "submitted_padded_forward_positions": 0}
    target_counts = []
    candidate_index = None
    try:
        adapter.eval()
        with torch.no_grad():
            for i in range(candidates):
                candidate_index = i
                prefix = f"sentence_{i}"
                inputs = batch[prefix + "_inputs"].to(device)
                attn = batch[prefix + "_attn_mask"].to(device)
                targets = batch[prefix + "_targets"].to(device)
                phrase_mask = batch[prefix + "_phrase_mask"].to(device)
                counts["model_forward_attempts"] += 1
                counts["submitted_forward_input_tokens"] += int(attn.sum())
                counts["submitted_padded_forward_positions"] += inputs.numel()
                logits = adapter(input_ids=inputs, attention_mask=attn)["logits"]
                counts["model_forward_calls"] += 1
                counts["forward_input_tokens"] += int(attn.sum())
                counts["padded_forward_positions"] += inputs.numel()
                if logits.shape[:2] != inputs.shape or not bool(torch.isfinite(logits).all()):
                    raise ValueError("Nonfinite or misaligned model logits; evaluation failed")
                target_counts.append(phrase_mask.sum(1).cpu())
                for j, temperature in enumerate(temperatures):
                    log_probs = F.log_softmax(logits / temperature, dim=-1)
                    if not bool(torch.isfinite(log_probs).all()):
                        raise ValueError("Nonfinite temperature-scaled log probabilities")
                    token_scores = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    value = (token_scores * phrase_mask).sum(1)
                    if task in LENGTH_NORMALIZED_TASKS:
                        value = value / phrase_mask.sum(1).clamp(min=1)
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError("Nonfinite completion scores")
                    scores[j].append(value.cpu())
                counts["scored_target_tokens"] += int(phrase_mask.sum())
    except Exception as exc:
        raise CausalScoringError(str(exc), counters=counts, task=task,
                                candidate_index=candidate_index) from exc
    finally:
        for module, mode in modes:
            module.training = mode
    scored_counts = torch.stack(target_counts, dim=1)
    return {"upstream_commit": UPSTREAM_COMMIT, "task": task, "temperatures": list(temperatures),
            "normalization": "completion_token_mean" if task in LENGTH_NORMALIZED_TASKS else "completion_token_sum",
            "scores": [torch.stack(values, dim=1).tolist() for values in scores],
            "scored_target_counts": scored_counts.tolist(),
            "scorable_candidates": (scored_counts > 0).tolist(),
            "ranking_allowed_per_record": (scored_counts > 0).all(1).tolist(), "counters": counts,
            "tokenizer_sha256": getattr(processor, "sha256", None),
            "official_benchmark_executed": False}

"""Pinned BabyLM BLiMP/EWoK record decoding and label-based aggregation.

Pure in-memory functions: no benchmark files, weights, network or model calls.
The separately bounded runner owns file verification and raw JSONL traversal.
Rules are from babylm-org/babylm-eval commit below, not a new task definition.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import PurePosixPath
import re

import torch

UPSTREAM_COMMIT = "6f825c291e2c4c78ad33b1935fd64d45f52642dc"
SUPPORTED_TASKS = frozenset({"blimp", "ewok"})
RECORD_SCHEMA = "babylm-official-task-record-v0"


def _sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string")
    return value


def _text(mapping, key):
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _no_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError(f"Non-JSON numerical constant: {value}")


def _identity(task, source):
    return _sha256(_canonical({"schema": RECORD_SCHEMA, "task": task,
        "source_path": source["path"], "source_sha256": source["file_sha256"],
        "line_number": source["line_number"],
        "raw_line_sha256": source["raw_line_sha256"],
        "canonical_raw_sha256": source["canonical_raw_sha256"]}))


def normalize_record(raw_record, *, task, source_path, line_number, source_sha256,
                     raw_line=None, full_sentence_scores=False):
    """Decode one raw object, retaining source identity outside task metadata.

    BLiMP supplements use task='blimp' and the official no-'field' branch.
    Extra raw keys are allowed, hashed, and ignored for task metadata, just as
    the official decoder ignores them. They cannot silently form new subgroups.
    ``raw_line`` should retain its original line terminator; its exact UTF-8
    hash differs from the canonical object hash. Without it that hash is None.
    """
    if task not in SUPPORTED_TASKS:
        raise ValueError("Only audited task names 'blimp' and 'ewok' are supported")
    if not isinstance(raw_record, dict):
        raise ValueError("Raw record must be a JSON object")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("source_path must be an explicit nonempty logical path")
    source_path = source_path.replace("\\", "/")
    if PurePosixPath(source_path).suffix != ".jsonl":
        raise ValueError("Source association must name a JSONL file")
    if type(line_number) is not int or line_number < 1:
        raise ValueError("line_number is the positive, one-based original physical line")
    _valid_sha(source_sha256, "source_sha256")
    if type(full_sentence_scores) is not bool:
        raise ValueError("full_sentence_scores must be boolean")
    canonical_raw = _canonical(raw_record)
    if raw_line is not None:
        if not isinstance(raw_line, str):
            raise ValueError("raw_line must be exactly one physical JSONL line")
        body = raw_line[:-2] if raw_line.endswith("\r\n") else raw_line[:-1] if raw_line.endswith(("\r", "\n")) else raw_line
        if not body or "\r" in body or "\n" in body:
            raise ValueError("raw_line must be exactly one physical JSONL line")
        decoded = json.loads(raw_line, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
        if _canonical(decoded) != canonical_raw:
            raise ValueError("raw_line content differs from supplied raw_record")

    if task == "blimp":
        good, bad = _text(raw_record, "sentence_good"), _text(raw_record, "sentence_bad")
        if "field" in raw_record:
            field = _text(raw_record, "field")
            if field == "syntax_semantics":
                field = "syntax/semantics"
            metadata = {"field": field, "UID": _text(raw_record, "UID"),
                        "linguistics_term": _text(raw_record, "linguistics_term")}
        else:
            metadata = {"field": "supplement", "UID": PurePosixPath(source_path).stem,
                        "linguistics_term": "supplement"}
        sentences, prefixes, completions = [good, bad], [None, None], [good, bad]
    else:
        context1, context2, target = (_text(raw_record, key) for key in ("Context1", "Context2", "Target1"))
        # This is the pinned paired task: one Target1 under two contexts. Target2
        # is not used. The upstream full_sentence_scores argument is unused.
        sentences = [" ".join([context1, target]), " ".join([context2, target])]
        prefixes = [context1, context2]
        completions = [" " + target, " " + target]
        metadata = {"UID": _text(raw_record, "Domain"),
                    "context_type": _text(raw_record, "ContextType"),
                    "context_contrast": _text(raw_record, "ContextDiff"),
                    "target_contrast": _text(raw_record, "TargetDiff")}
    source = {"path": source_path, "line_number": line_number, "file_sha256": source_sha256,
              "raw_line_sha256": _sha256(raw_line) if raw_line is not None else None,
              "canonical_raw_sha256": _sha256(canonical_raw),
              "raw_line_available": raw_line is not None}
    return {"sentences": sentences, "prefixes": prefixes, "completions": completions,
            "label": 0, **metadata, "metadata": metadata.copy(), "task": task,
            "upstream_commit": UPSTREAM_COMMIT, "record_schema": RECORD_SCHEMA,
            "source": source, "record_id": _identity(task, source),
            "normalization_options": {"full_sentence_scores_requested": full_sentence_scores,
                "full_sentence_scores_changes_pinned_decoder": False}}


def _validate_records(records, task):
    if not records:
        raise ValueError("Cannot aggregate an empty task")
    seen = set()
    expected_keys = ({"field", "UID", "linguistics_term"} if task == "blimp" else
                     {"UID", "context_type", "context_contrast", "target_contrast"})
    for record in records:
        if record.get("task") != task or record.get("upstream_commit") != UPSTREAM_COMMIT or record.get("record_schema") != RECORD_SCHEMA:
            raise ValueError("Task/commit/record-schema mismatch; do not mix tasks")
        record_id = _valid_sha(record.get("record_id"), "record_id")
        if record_id in seen:
            raise ValueError("Duplicate record_id would count the same source line twice")
        seen.add(record_id)
        source = record.get("source")
        if not isinstance(source, dict) or _identity(task, source) != record_id:
            raise ValueError("Source identity does not match record_id")
        metadata = record.get("metadata")
        if not isinstance(metadata, dict) or set(metadata) != expected_keys:
            raise ValueError("Metadata must contain exactly the pinned task categories")
        for key in metadata:
            if _text(metadata, key) != record.get(key):
                raise ValueError("Task metadata differs from normalized official field")
        if type(record.get("label")) is not int or record["label"] != 0:
            raise ValueError("Pinned BLiMP and EWoK decoders assign label 0")
        for key in ("sentences", "prefixes", "completions"):
            if not isinstance(record.get(key), list) or len(record[key]) != 2:
                raise ValueError("Pinned paired task requires two aligned candidates")
        if any(not isinstance(s, str) or not s.strip() for s in record["sentences"] + record["completions"]):
            raise ValueError("Candidate sentences/completions must be nonempty strings")
        if any(not sentence.endswith(completion) for sentence, completion in zip(record["sentences"], record["completions"])):
            raise ValueError("Candidate completion must remain an exact sentence suffix")


def _validate_score_result(records, score_result):
    if not isinstance(score_result, dict):
        raise ValueError("score_result must be the structured scorer output")
    task = score_result.get("task")
    if task not in SUPPORTED_TASKS or score_result.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Unsupported task or scorer commit")
    if score_result.get("normalization") != "completion_token_sum":
        raise ValueError("Pinned BLiMP/EWoK require completion_token_sum scores")
    _validate_records(records, task)
    temperatures = score_result.get("temperatures")
    if not isinstance(temperatures, (list, tuple)) or not temperatures:
        raise ValueError("A nonempty, preselected temperature list is required")
    if any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in temperatures) or len(set(temperatures)) != len(temperatures):
        raise ValueError("Temperatures must be finite, positive and unique")
    temperatures = [float(t) for t in temperatures]
    n = len(records)
    scores = score_result.get("scores")
    if not isinstance(scores, list) or len(scores) != len(temperatures):
        raise ValueError("Scores must have one slice per fixed temperature")
    for values in scores:
        if not isinstance(values, list) or len(values) != n:
            raise ValueError("Scores do not align to every source record")
        for row in values:
            if not isinstance(row, list) or len(row) != 2 or any(type(s) not in (int, float) or not math.isfinite(s) for s in row):
                raise ValueError("Each paired score must be finite; NaN/Inf cannot become predictions")
    counts = score_result.get("scored_target_counts")
    if not isinstance(counts, list) or len(counts) != n or any(not isinstance(row, list) or len(row) != 2 or
            any(type(c) is not int or c <= 0 for c in row) for row in counts):
        raise ValueError("Every candidate needs a positive number of scored target tokens")
    scorable, rankable = score_result.get("scorable_candidates"), score_result.get("ranking_allowed_per_record")
    if (not isinstance(scorable, list) or len(scorable) != n or
            any(not isinstance(row, list) or len(row) != 2 or any(value is not True for value in row) for row in scorable) or
            not isinstance(rankable, list) or len(rankable) != n or any(value is not True for value in rankable)):
        raise ValueError("Scorer must explicitly mark all candidates/records rankable")
    if "record_ids" in score_result and score_result["record_ids"] != [r["record_id"] for r in records]:
        raise ValueError("Scorer record IDs differ from the declared source order")
    sizes = score_result.get("batch_sizes", [n])
    if not isinstance(sizes, list) or not sizes or any(type(size) is not int or size <= 0 for size in sizes) or sum(sizes) != n:
        raise ValueError("batch_sizes must describe the complete, nonempty scoring batches")
    return task, temperatures, scores, sizes


def aggregate_scores(records, score_result, *, tie_seed=0):
    """Official per-category accuracy and mean of UID accuracies, per T.

    There is no temperature selection. All scores must be from one task and
    the same ordered records. Ties use the official uniform torch.randint rule
    with an isolated CPU generator; batch_sizes reproduces upstream's RNG
    order: batch, temperature, record. The caller aggregates only after all
    scoring batches finish; never restart tie RNG separately for each batch.
    """
    records = list(records)
    task, temperatures, scores, sizes = _validate_score_result(records, score_result)
    if type(tie_seed) is not int or not 0 <= tie_seed < 2**63:
        raise ValueError("tie_seed must be an explicit integer in [0, 2**63)")
    generator = torch.Generator(device="cpu").manual_seed(tie_seed)
    chosen = [[None] * len(records) for _ in temperatures]
    ties = [[None] * len(records) for _ in temperatures]
    start = 0
    for size in sizes:
        for ti in range(len(temperatures)):
            for ri in range(start, start + size):
                row = scores[ti][ri]
                maximum = max(row)
                indices = [ci for ci, value in enumerate(row) if value == maximum]
                ties[ti][ri] = indices
                chosen[ti][ri] = indices[torch.randint(len(indices), (1,), generator=generator).item()] if len(indices) > 1 else indices[0]
        start += size

    results = []
    for ti, temperature in enumerate(temperatures):
        counts = {key: {"total": {}, "correct": {}} for key in records[0]["metadata"]}
        predictions = {}
        uid_bounds = defaultdict(lambda: {"total": 0, "minimum": 0.0, "maximum": 0.0, "expected": 0.0})
        items = []
        total_correct, tie_count = 0, 0
        item_bounds = {"minimum": 0.0, "maximum": 0.0, "expected": 0.0}
        for ri, record in enumerate(records):
            selection, indices, label = chosen[ti][ri], ties[ti][ri], record["label"]
            correct = int(selection == label)
            total_correct += correct
            is_tie = len(indices) > 1
            tie_count += int(is_tie)
            lower = float(indices == [label])
            upper = float(label in indices)
            expected = upper / len(indices)
            for key, value in record["metadata"].items():
                counts[key]["total"][value] = counts[key]["total"].get(value, 0) + 1
                counts[key]["correct"][value] = counts[key]["correct"].get(value, 0) + correct
            uid = record["UID"]
            bucket = predictions.setdefault(uid, {"predictions": []})["predictions"]
            prediction_id = f"{uid}_{len(bucket)}"
            pred = (record["sentences"] if task == "ewok" else record["completions"])[selection]
            bucket.append({"id": prediction_id, "pred": pred})
            uid_bounds[uid]["total"] += 1
            for name, value in (("minimum", lower), ("maximum", upper), ("expected", expected)):
                uid_bounds[uid][name] += value
                item_bounds[name] += value
            items.append({"record_id": record["record_id"], "source": record["source"].copy(),
                          "UID": uid, "prediction_id": prediction_id, "pred": pred,
                          "candidate_scores": list(scores[ti][ri]), "chosen_index": selection,
                          "label": label, "correct": bool(correct), "tied_indices": list(indices),
                          "is_tie": is_tie, "correctness_minimum": lower,
                          "correctness_maximum": upper, "correctness_expected": expected})
        accuracies = {key: {category: 100.0 * count_dict["correct"][category] / total
                           for category, total in count_dict["total"].items()}
                      for key, count_dict in counts.items()}
        # Exactly run.process_results for the two supported task names.
        official_average = sum(accuracies["UID"].values()) / len(accuracies["UID"])
        results.append({"temperature": temperature, "counts_by_metadata": counts,
                        "accuracy_by_metadata_pct": accuracies,
                        "official_uid_mean_accuracy_pct": official_average,
                        "diagnostic_item_accuracy_pct": 100.0 * total_correct / len(records),
                        "official_predictions": predictions, "items": items, "tie_count": tie_count,
                        "tie_uncertainty": {
                            "interpretation": "Range and expectation over uniform tie-breaking only; not statistical confidence intervals",
                            "item_accuracy_pct": {name: 100.0 * value / len(records) for name, value in item_bounds.items()},
                            "uid_mean_accuracy_pct": {name: sum(100.0 * bucket[name] / bucket["total"] for bucket in uid_bounds.values()) / len(uid_bounds)
                                                      for name in item_bounds}},
                        "record_count": len(records), "uid_count": len(accuracies["UID"])})
    order = [record["record_id"] for record in records]
    return {"schema": "babylm-official-task-aggregation-v0", "upstream_commit": UPSTREAM_COMMIT,
            "task": task, "temperatures": temperatures, "temperature_selection": "none",
            "tie_seed": tie_seed, "tie_rng": "isolated torch.Generator(cpu)",
            "tie_execution_order": "batch_then_temperature_then_record",
            "batch_sizes": sizes.copy(), "record_ids": order,
            "record_order_sha256": _sha256(_canonical(order)), "results": results,
            "scope": "In-memory decoding and aggregation only; not a complete official benchmark execution",
            "model_forward_calls": 0, "backward_calls": 0, "optimizer_updates": 0}

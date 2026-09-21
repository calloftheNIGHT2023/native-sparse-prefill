"""Bounded tiny synthetic GPU numerics preflight, never scientific training.

Prepared CLI only until an external launcher authorizes execution and imposes a
hard process timeout. Normal path: 9 model forwards, 9 backwards, 3 AdamW steps.
No data files, pretrained weights, tokenizers, downloads, or package installs.
--help exits before loading torch or querying CUDA. Any failed check exits 1.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FORWARDS = 20
MAX_OPTIMIZER_STEPS = 6
THRESHOLDS = {
    "cpu_cuda_fp32": {"atol": 1e-4, "rtol": 1e-3},
    "gpu_full_support": {"atol": 1e-6, "rtol": 1e-5},
    "checkpoint_replay": {"atol": 1e-6, "rtol": 1e-5},
}
# Thresholds are fixed in source before any results exist. No adaptive relaxation.
COUNTS = {
    "model_forward_attempts": 0, "model_forward_calls": 0,
    "cpu_model_forward_calls": 0, "cuda_model_forward_calls": 0,
    "backward_attempts": 0, "backward_calls": 0,
    "cpu_backward_calls": 0, "cuda_backward_calls": 0,
    "optimizer_step_attempts": 0, "engineering_optimizer_steps": 0,
    "scientific_optimizer_steps": 0, "scientific_training_tokens": 0,
    "synthetic_input_tokens": 0, "synthetic_loss_tokens": 0,
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def record_check(report, name, result):
    report["checks"][name] = result
    require(result["passed"], f"Numerical check failed: {name}; see recorded measurements")


def finite_number(value):
    return float(value) if math.isfinite(float(value)) else None


def compare_tensors(expected, actual, threshold):
    if expected is None or actual is None:
        return {"passed": expected is actual, "expected_none": expected is None,
                "actual_none": actual is None}
    if expected.shape != actual.shape:
        return {"passed": False, "expected_shape": list(expected.shape), "actual_shape": list(actual.shape)}
    reference = expected.detach().to(device="cpu", dtype=torch.float64)
    observed = actual.detach().to(device="cpu", dtype=torch.float64)
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(observed).all())
    difference = (observed - reference).abs()
    tolerance = threshold["atol"] + threshold["rtol"] * reference.abs()
    return {"passed": finite and bool((difference <= tolerance).all()),
            "finite": finite, "elements": reference.numel(),
            "max_abs_error": finite_number(difference.max().item()) if reference.numel() else 0.,
            "max_relative_error_denom_floor_1e12": finite_number((difference / reference.abs().clamp_min(1e-12)).max().item()) if reference.numel() else 0.,
            "max_tolerance_ratio": finite_number((difference / tolerance).max().item()) if reference.numel() else 0.,
            "threshold": threshold}


def compare_maps(expected, actual, threshold):
    if set(expected) != set(actual):
        return {"passed": False, "missing_names": sorted(set(expected) - set(actual)),
                "unexpected_names": sorted(set(actual) - set(expected))}
    checks = {name: compare_tensors(expected[name], actual[name], threshold) for name in expected}
    return {"passed": all(row["passed"] for row in checks.values()),
            "tensor_count": len(checks), "failed_names": [n for n, row in checks.items() if not row["passed"]],
            "max_abs_error": max((row.get("max_abs_error") or 0.) for row in checks.values()) if checks else 0.,
            "per_tensor": checks}


def is_indexer(name):
    return ".indexer." in name


def parameter_gradients(model, backbone_only=False):
    return {name: parameter.grad for name, parameter in model.named_parameters()
            if not backbone_only or not is_indexer(name)}


def backbone_state(model):
    return {name: value for name, value in model.state_dict().items() if not is_indexer(name)}


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode("utf-8"))
        h.update(str(value.dtype).encode("ascii"))
        h.update(str(tuple(value.shape)).encode("ascii"))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def forward_backward(model, inputs, auxiliary_weight):
    require(COUNTS["model_forward_attempts"] < MAX_FORWARDS, "Forward-attempt cap reached")
    model.zero_grad(set_to_none=True)
    COUNTS["model_forward_attempts"] += 1
    prediction = model(inputs, aux_weight=auxiliary_weight)
    if inputs.device.type == "cuda":
        torch.cuda.synchronize(inputs.device)
    COUNTS["model_forward_calls"] += 1
    COUNTS[f"{inputs.device.type}_model_forward_calls"] += 1
    COUNTS["synthetic_input_tokens"] += inputs.numel()
    COUNTS["synthetic_loss_tokens"] += int(prediction.token_loss_count)
    require(bool(torch.isfinite(prediction.loss)), "Nonfinite tiny synthetic loss")
    COUNTS["backward_attempts"] += 1
    prediction.loss.backward()
    if inputs.device.type == "cuda":
        torch.cuda.synchronize(inputs.device)
    COUNTS["backward_calls"] += 1
    COUNTS[f"{inputs.device.type}_backward_calls"] += 1
    return prediction


def gradient_group_report(model):
    groups = {}
    for label, indexer in [("backbone", False), ("indexer", True)]:
        parameters = [(n, p) for n, p in model.named_parameters() if is_indexer(n) == indexer]
        missing = [n for n, p in parameters if p.grad is None]
        nonfinite = [n for n, p in parameters if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        nonzero = [n for n, p in parameters if p.grad is not None and bool((p.grad != 0).any())]
        squared = sum(float(p.grad.detach().double().square().sum().cpu()) for _, p in parameters if p.grad is not None)
        groups[label] = {"parameter_tensors": len(parameters), "missing_gradients": missing,
                         "nonfinite_gradients": nonfinite, "nonzero_gradient_tensors": len(nonzero),
                         "zero_gradient_names": [n for n, p in parameters if p.grad is not None and n not in nonzero],
                         "gradient_l2_norm": finite_number(math.sqrt(squared)),
                         "passed": bool(parameters) and not missing and not nonfinite and bool(nonzero)}
    return {"passed": all(g["passed"] for g in groups.values()), "groups": groups}


def synthetic_update(model, optimizer, device):
    require(COUNTS["optimizer_step_attempts"] < MAX_OPTIMIZER_STEPS, "Optimizer-step cap reached")
    # Exercise Python, NumPy, CPU torch and CUDA RNG state in the replay, while
    # remaining entirely synthetic. No text corpus or tokenizer is involved.
    draws = {"python": random.randrange(97), "numpy": int(np.random.randint(0, 97)),
             "torch_cpu": int(torch.randint(0, 97, (1,), device="cpu").item())}
    inputs = torch.randint(0, 97, (1, 32), device=device)
    inputs = (inputs + sum(draws.values())) % 97
    optimizer.zero_grad(set_to_none=True)
    prediction = forward_backward(model, inputs, 1.0)
    require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
            "Nonfinite gradient before replay optimizer step")
    COUNTS["optimizer_step_attempts"] += 1
    optimizer.step()
    torch.cuda.synchronize(device)
    COUNTS["engineering_optimizer_steps"] += 1
    return {"draws": draws, "inputs": inputs.detach().cpu().clone(),
            "logits": prediction.logits.detach().cpu().clone(),
            "loss": prediction.loss.detach().cpu().clone()}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_equal(a, b):
    numpy_equal = (a["numpy"][0] == b["numpy"][0]
                   and np.array_equal(a["numpy"][1], b["numpy"][1])
                   and a["numpy"][2:] == b["numpy"][2:])
    return (a["python"] == b["python"] and numpy_equal
            and torch.equal(a["torch_cpu"], b["torch_cpu"])
            and len(a["torch_cuda"]) == len(b["torch_cuda"])
            and all(torch.equal(x, y) for x, y in zip(a["torch_cuda"], b["torch_cuda"])))


def flatten_tensors(value, prefix="root"):
    tensors, other = {}, {}
    if torch.is_tensor(value):
        tensors[prefix] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            a, b = flatten_tensors(child, f"{prefix}.{key}")
            tensors.update(a); other.update(b)
    elif isinstance(value, (tuple, list)):
        other[prefix + ".container_type"] = type(value).__name__
        for i, child in enumerate(value):
            a, b = flatten_tensors(child, f"{prefix}[{i}]")
            tensors.update(a); other.update(b)
    else:
        other[prefix] = value
    return tensors, other


def run_checks(report, output, device_string):
    # Lazy imports keep --help source-only and ensure unavailable CUDA is a
    # recorded failure, never a silent CPU replacement.
    global torch, np
    import torch
    import numpy as np
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from dataclasses import replace
    from src.babylm_hybrid.config import HybridConfig
    from src.babylm_hybrid import model as model_module
    build_model = model_module.build_model
    sources = [Path(__file__), ROOT / "src/babylm_hybrid/config.py",
               ROOT / "src/babylm_hybrid/model.py", ROOT / "src/babylm_hybrid/attention.py",
               Path(inspect.getfile(model_module.Qwen3NextGatedDeltaNet))]
    report["source_hashes"] = {str(p.resolve()): sha(p) for p in sources}
    report["runtime"] = {"python": sys.version, "platform": platform.platform(),
                         "torch": str(torch.__version__), "numpy": np.__version__,
                         "cuda_compiled_version": torch.version.cuda,
                         "cudnn_version": torch.backends.cudnn.version()}
    require(device_string.startswith("cuda"), "Only CUDA devices are accepted")
    require(torch.cuda.is_available(), "CUDA unavailable; GPU numerics cannot pass on CPU")
    device = torch.device(device_string)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    report["hardware"] = {"device": str(device), "name": properties.name,
                          "total_memory_bytes": properties.total_memory,
                          "compute_capability": [properties.major, properties.minor],
                          "multiprocessors": properties.multi_processor_count,
                          "visible_cuda_device_count": torch.cuda.device_count()}
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
        report["hardware"]["nvidia_smi"] = {"returncode": query.returncode, "stdout": query.stdout.strip(), "stderr": query.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        report["hardware"]["nvidia_smi"] = {"unavailable": str(error)}
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats(device)
    report["runtime"].update({"dtype": "float32", "torch_num_threads": torch.get_num_threads(),
                              "tf32_matmul": False, "tf32_cudnn": False,
                              "deterministic_algorithms": True,
                              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")})
    cfg = HybridConfig()  # 4 tiny layers, width 32, vocabulary 97; NOT 97M model.
    report["tiny_config"] = cfg.to_dict()
    require(cfg.selected_complete_blocks == 2 and cfg.block_size == 4, "Unexpected sparse test configuration")
    random.seed(719); np.random.seed(719); torch.manual_seed(719); torch.cuda.manual_seed_all(719)
    cpu_dense = build_model(cfg, "dense", 731, 991).float()
    cpu_sparse = build_model(cfg, "sparse", 731, 991).float()
    report["parameters"] = {"dense": model_module.parameter_counts(cpu_dense),
                            "sparse": model_module.parameter_counts(cpu_sparse)}
    require(report["parameters"]["sparse"]["total"] < 1_000_000, "Preflight unexpectedly exceeds tiny model limit")
    d_state, e_state = backbone_state(cpu_dense), backbone_state(cpu_sparse)
    exact = set(d_state) == set(e_state) and all(torch.equal(d_state[n], e_state[n]) for n in d_state)
    record_check(report, "shared_seeded_backbone_exact", {"passed": exact,
                 "dense_backbone_sha256": state_hash(d_state), "sparse_backbone_sha256": state_hash(e_state)})
    gpu_dense, gpu_sparse = copy.deepcopy(cpu_dense).to(device), copy.deepcopy(cpu_sparse).to(device)
    fixed_cpu = ((torch.arange(32, device="cpu").reshape(1, 32) * 7 + 11) % 97).long()
    fixed_gpu = fixed_cpu.to(device)
    for name, cpu_model, gpu_model, auxiliary in [
        ("dense", cpu_dense, gpu_dense, 0.0), ("sparse", cpu_sparse, gpu_sparse, 1.0)
    ]:
        expected = forward_backward(cpu_model, fixed_cpu, auxiliary)
        actual = forward_backward(gpu_model, fixed_gpu, auxiliary)
        measurements = {"logits": compare_tensors(expected.logits, actual.logits, THRESHOLDS["cpu_cuda_fp32"]),
                        "lm_loss": compare_tensors(expected.lm_loss, actual.lm_loss, THRESHOLDS["cpu_cuda_fp32"]),
                        "aux_loss": compare_tensors(expected.aux_loss, actual.aux_loss, THRESHOLDS["cpu_cuda_fp32"]),
                        "gradients": compare_maps(parameter_gradients(cpu_model), parameter_gradients(gpu_model), THRESHOLDS["cpu_cuda_fp32"])}
        record_check(report, f"cpu_cuda_fp32_{name}", {"passed": all(v["passed"] for v in measurements.values()), "measurements": measurements})
        if name == "sparse":
            kept = sum(s["logical_kept_pairs"] for s in actual.attention_stats)
            dense = sum(s["logical_dense_causal_pairs"] for s in actual.attention_stats)
            record_check(report, "truly_sparse_gpu_support", {"passed": 0 < kept < dense,
                         "kept_pairs": kept, "dense_pairs": dense, "retained_fraction": kept / dense})
            record_check(report, "sparse_gpu_gradient_groups", gradient_group_report(gpu_model))
    full_sparse = build_model(replace(cfg, selected_complete_blocks=64), "sparse", 731, 991).float().to(device)
    dense_out = forward_backward(gpu_dense, fixed_gpu, 0.0)
    full_out = forward_backward(full_sparse, fixed_gpu, 0.0)
    full_checks = {"logits": compare_tensors(dense_out.logits, full_out.logits, THRESHOLDS["gpu_full_support"]),
                   "lm_loss": compare_tensors(dense_out.lm_loss, full_out.lm_loss, THRESHOLDS["gpu_full_support"]),
                   "backbone_gradients": compare_maps(parameter_gradients(gpu_dense, True), parameter_gradients(full_sparse, True), THRESHOLDS["gpu_full_support"])}
    all_support = all(s["logical_kept_pairs"] == s["logical_dense_causal_pairs"] for s in full_out.attention_stats)
    record_check(report, "gpu_full_support_dense_sparse_equivalence", {
        "passed": all_support and all(v["passed"] for v in full_checks.values()),
        "all_causal_support_selected": all_support, "measurements": full_checks})
    optimizer = torch.optim.AdamW(gpu_sparse.parameters(), lr=1e-3, weight_decay=0.01)
    synthetic_update(gpu_sparse, optimizer, device)  # Establish nonempty Adam state.
    saved = {"model": gpu_sparse.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(),
             "config": cfg.to_dict(), "engineering_optimizer_step": 1}
    checkpoint = output.with_name(f"{output.stem}.{report['attempt_id']}.checkpoint.pt")
    torch.save(saved, checkpoint)
    report["checkpoint"] = {"path": str(checkpoint), "sha256": sha(checkpoint), "bytes": checkpoint.stat().st_size}
    expected_step = synthetic_update(gpu_sparse, optimizer, device)
    expected_model = {n: value.detach().cpu().clone() for n, value in gpu_sparse.state_dict().items()}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_rng = rng_state()
    # This checkpoint was created locally by this very invocation and hash-checked.
    require(sha(checkpoint) == report["checkpoint"]["sha256"], "Checkpoint changed before restore")
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    replay = build_model(cfg, "sparse", 731, 991).float().to(device)
    replay.load_state_dict(loaded["model"], strict=True)
    replay_optimizer = torch.optim.AdamW(replay.parameters(), lr=1e-3, weight_decay=0.01)
    replay_optimizer.load_state_dict(loaded["optimizer"])
    restore_rng(loaded["rng"])
    replay_step = synthetic_update(replay, replay_optimizer, device)
    actual_rng = rng_state()
    expected_tensors, expected_values = flatten_tensors(expected_optimizer)
    actual_tensors, actual_values = flatten_tensors(replay_optimizer.state_dict())
    replay_checks = {"parameters": compare_maps(expected_model, replay.state_dict(), THRESHOLDS["checkpoint_replay"]),
                     "optimizer_tensors": compare_maps(expected_tensors, actual_tensors, THRESHOLDS["checkpoint_replay"]),
                     "logits": compare_tensors(expected_step["logits"], replay_step["logits"], THRESHOLDS["checkpoint_replay"]),
                     "loss": compare_tensors(expected_step["loss"], replay_step["loss"], THRESHOLDS["checkpoint_replay"])}
    exact_inputs = torch.equal(expected_step["inputs"], replay_step["inputs"])
    exact_draws = expected_step["draws"] == replay_step["draws"]
    exact_rng = rng_equal(expected_rng, actual_rng)
    exact_optimizer_values = expected_values == actual_values
    record_check(report, "checkpoint_optimizer_rng_replay", {
        "passed": exact_inputs and exact_draws and exact_rng and exact_optimizer_values
                  and all(v["passed"] for v in replay_checks.values()),
        "synthetic_inputs_exact": exact_inputs, "python_numpy_cpu_draws_exact": exact_draws,
        "post_update_rng_exact": exact_rng, "optimizer_nontensor_state_exact": exact_optimizer_values,
        "measurements": replay_checks})
    report["hardware"]["peak_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["hardware"]["peak_cuda_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    require(COUNTS["model_forward_calls"] == 9 and COUNTS["engineering_optimizer_steps"] == 3,
            "Unexpected normal-path execution counts")


def write_result(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    history = output.with_name(output.name + ".jsonl")
    # Preserve an existing latest result even if it predates this history file.
    if output.exists() and not history.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(previous, ensure_ascii=False, allow_nan=False) + "\n")
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    temporary = output.with_name(output.name + ".tmp-" + report["attempt_id"])
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Latest JSON result; attempts append to OUTPUT.jsonl")
    parser.add_argument("--device", default="cuda:0", help="CUDA device only (default cuda:0); no CPU fallback")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started = time.perf_counter()
    report = {"schema_version": 1, "scope": "tiny_synthetic_gpu_numerics_engineering_only",
              "attempt_id": uuid.uuid4().hex, "started_utc": utc(), "status": "running", "passed": False,
              "thresholds": THRESHOLDS, "limits": {"model_forward_attempts": MAX_FORWARDS,
              "optimizer_step_attempts": MAX_OPTIMIZER_STEPS, "normal_path_forwards": 9,
              "normal_path_optimizer_steps": 3, "hard_timeout": "required_external_launcher"},
              "checks": {}, "counts": COUNTS, "source_hashes": {str(Path(__file__).resolve()): sha(Path(__file__))},
              "scientific_claim": "No training-quality or speed claim; no BabyLM data consumed"}
    try:
        run_checks(report, output, args.device)
        report["status"], report["passed"] = "passed", True
    except BaseException as error:
        report["status"], report["passed"] = "failed", False
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        report["completed_utc"] = utc()
        report["elapsed_seconds"] = time.perf_counter() - started
        report["counts"] = dict(COUNTS)
        write_result(output, report)
    print(json.dumps({"status": report["status"], "passed": report["passed"],
                      "output": str(output), "counts": report["counts"],
                      "elapsed_seconds": report["elapsed_seconds"]}, allow_nan=False), flush=True)
    if not report["passed"]:
        print(report["failure"]["message"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

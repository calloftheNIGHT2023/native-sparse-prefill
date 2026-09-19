"""Archive bounded synthetic CPU training-engine validation and call counts."""
import hashlib
import argparse
import importlib.util
import io
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="*", help="Specific test method names; absent runs this suite")
    args = parser.parse_args()
    start, tick = utc(), time.perf_counter()
    path = ROOT / "tests/test_babylm_training_v0.py"
    spec = importlib.util.spec_from_file_location("test_babylm_training_v0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stamp = start.replace(":", "").replace("-", "").replace(".", "")
    module.ARTIFACT_ROOT = ROOT / "logs" / f"babylm-training-engine-fixtures-{stamp}"
    stream = io.StringIO()
    if args.tests:
        suite = unittest.TestSuite(module.TrainingEngineTests(name) for name in args.tests)
    else:
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    original_forward = module.HybridLM.forward
    original_backward = module.torch.Tensor.backward
    original_step = module.torch.optim.AdamW.step

    def counted_forward(model, *args, **kwargs):
        module.COUNTS["model_forward_calls"] += 1
        return original_forward(model, *args, **kwargs)

    def counted_backward(tensor, *args, **kwargs):
        module.COUNTS["backward_calls"] += 1
        return original_backward(tensor, *args, **kwargs)

    def counted_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        module.COUNTS["engineering_optimizer_steps"] += 1
        if module.COUNTS["engineering_optimizer_steps"] > 40:
            raise RuntimeError("Engineering validation optimizer budget exceeded")
        return result

    with mock.patch.object(module.HybridLM, "forward", counted_forward), \
         mock.patch.object(module.torch.Tensor, "backward", counted_backward), \
         mock.patch.object(module.torch.optim.AdamW, "step", counted_step):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    end = utc()
    paths = [path, Path(__file__)] + sorted((ROOT / "src/babylm_hybrid").glob("*.py"))
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    report = {
        "schema_version": 1, "status": "passed" if result.wasSuccessful() else "failed",
        "scope": "Synthetic CPU training-engine validation only; no BabyLM scientific run",
        "started_utc": start, "completed_utc": end, "elapsed_seconds": time.perf_counter()-tick,
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "selected_tests": args.tests,
        "counts": module.COUNTS, "measurements": module.MEASUREMENTS,
        "source_sha256": hashes,
        "fixture_and_raw_failure_directory": str(module.ARTIFACT_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "environment": {"python": platform.python_version(), "torch": module.torch.__version__,
                        "device": "cpu", "threads": module.torch.get_num_threads()},
        "limitations": ["Synthetic word/token accounting fixtures only", "No real BabyLM training tokens",
                        "Production data allowlist is unchanged; loader boundary patched only in tests",
                        "No CUDA or scientific updates", "CPU correctness is not sparse speed or quality evidence"],
    }
    log = ROOT / "logs" / f"babylm-training-engine-cpu-v0-{stamp}.log"
    log_text = f"started_utc={start}\ncompleted_utc={end}\n" + stream.getvalue()
    report["test_outcomes"] = dict(re.findall(r"^(test_\w+) \([^\n]*\) \.\.\. (ok|FAIL|ERROR)$", stream.getvalue(), flags=re.MULTILINE))
    log.write_text(log_text, encoding="utf-8")
    report["test_log"] = str(log.relative_to(ROOT)).replace("\\", "/")
    for destination in [ROOT / "logs" / f"babylm-training-engine-cpu-v0-{stamp}.json"]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    # Keep every failed attempt and count rerun work, but do not repeat already
    # successful optimizer work simply to replace a filename-only assertion.
    outcomes, cumulative, attempts, all_measurements = {}, {key: 0 for key in module.COUNTS}, [], {}
    for attempt_file in sorted((ROOT / "logs").glob("babylm-training-engine-cpu-v0-*.json")):
        previous = json.loads(attempt_file.read_text(encoding="utf-8"))
        previous_outcomes = previous.get("test_outcomes")
        if previous_outcomes is None:
            text = (ROOT / previous["test_log"]).read_text(encoding="utf-8")
            previous_outcomes = dict(re.findall(r"^(test_\w+) \([^\n]*\) \.\.\. (ok|FAIL|ERROR)$", text, flags=re.MULTILINE))
        for name, status in previous_outcomes.items():
            outcomes[name] = {"status": status, "attempt": str(attempt_file.relative_to(ROOT)).replace("\\", "/"),
                              "source_sha256": previous["source_sha256"]}
        for key, value in previous["counts"].items():
            cumulative[key] += value
        all_measurements.update(previous.get("measurements", {}))
        attempts.append({"path": str(attempt_file.relative_to(ROOT)).replace("\\", "/"),
                         "status": previous["status"], "counts": previous["counts"]})
    required = unittest.defaultTestLoader.getTestCaseNames(module.TrainingEngineTests)
    combined = {**report, "status": "passed" if all(outcomes.get(name, {}).get("status") == "ok" for name in required) else "failed",
                "tests_run": len(outcomes), "current_attempt": {"tests_run": result.testsRun, "counts": module.COUNTS},
                "counts": cumulative, "measurements": all_measurements, "latest_test_outcomes": outcomes,
                "attempts": attempts,
                "verification_provenance": "Per-test source versions preserved; targeted reruns do not assert that unrelated earlier tests were repeated against changed sources."}
    (ROOT / "results/babylm-training-engine-cpu-v0.json").write_text(json.dumps(combined, indent=2)+"\n", encoding="utf-8")
    print(log_text)
    print(json.dumps({"status": report["status"], "counts": module.COUNTS,
                      "fixtures": str(module.ARTIFACT_ROOT)}, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

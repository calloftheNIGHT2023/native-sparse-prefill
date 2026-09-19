"""Preserve targeted instrumentation checks, attempts, and actual compute counts."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
from datetime import datetime, timezone
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="*")
    args = parser.parse_args()
    begin, tick = utc(), time.perf_counter()
    path = ROOT / "tests/test_babylm_instrumentation_v0.py"
    spec = importlib.util.spec_from_file_location("test_babylm_instrumentation_v0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stamp = begin.replace("-", "").replace(":", "").replace(".", "")
    module.ARTIFACT_ROOT = ROOT / "logs" / f"babylm-instrumentation-fixtures-{stamp}"
    previous_reports = sorted((ROOT / "logs").glob("babylm-instrumentation-cpu-v0-*.json"))
    previous_steps = sum(json.loads(p.read_text(encoding="utf-8"))["counts"]["engineering_optimizer_steps"] for p in previous_reports)
    suite = (unittest.TestSuite(module.InstrumentationTests(name) for name in args.tests) if args.tests
             else unittest.defaultTestLoader.loadTestsFromTestCase(module.InstrumentationTests))
    original_forward, original_backward = module.HybridLM.forward, module.torch.Tensor.backward
    original_step = module.torch.optim.AdamW.step

    def counted_forward(model, *args, **kwargs):
        module.COUNTS["model_forward_calls"] += 1
        return original_forward(model, *args, **kwargs)

    def counted_backward(tensor, *args, **kwargs):
        module.COUNTS["backward_calls"] += 1
        return original_backward(tensor, *args, **kwargs)

    def counted_step(optimizer, *args, **kwargs):
        if previous_steps + module.COUNTS["engineering_optimizer_steps"] >= 12:
            raise RuntimeError("Cumulative instrumentation audit budget of 12 optimizer calls exhausted")
        result = original_step(optimizer, *args, **kwargs)
        module.COUNTS["engineering_optimizer_steps"] += 1
        return result

    stream = io.StringIO()
    with mock.patch.object(module.HybridLM, "forward", counted_forward), \
         mock.patch.object(module.torch.Tensor, "backward", counted_backward), \
         mock.patch.object(module.torch.optim.AdamW, "step", counted_step):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    end = utc()
    source_paths = [path, Path(__file__), ROOT / "tests/test_babylm_training_v0.py"] + sorted((ROOT / "src/babylm_hybrid").glob("*.py"))
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    text = f"started_utc={begin}\ncompleted_utc={end}\n" + stream.getvalue()
    log = ROOT / "logs" / f"babylm-instrumentation-cpu-v0-{stamp}.log"
    log.write_text(text, encoding="utf-8")
    report = {"status": "passed" if result.wasSuccessful() else "failed", "started_utc": begin, "completed_utc": end,
              "elapsed_seconds": time.perf_counter()-tick, "tests_run": result.testsRun,
              "selected_tests": args.tests, "failures": len(result.failures), "errors": len(result.errors),
              "test_outcomes": dict(re.findall(r"^(test_\w+) \([^\n]*\) \.\.\. (ok|FAIL|ERROR)$", stream.getvalue(), re.MULTILINE)),
              "counts": module.COUNTS, "measurements": module.MEASUREMENTS, "source_sha256": hashes,
              "fixtures": str(module.ARTIFACT_ROOT.relative_to(ROOT)).replace("\\", "/"),
              "test_log": str(log.relative_to(ROOT)).replace("\\", "/"),
              "scope": "Synthetic CPU implementation audit only; not scientific training, language quality, or throughput evidence"}
    attempt_path = ROOT / "logs" / f"babylm-instrumentation-cpu-v0-{stamp}.json"
    attempt_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    outcomes, counts, measurements, attempts = {}, {k: 0 for k in module.COUNTS}, {}, []
    for attempt in previous_reports + [attempt_path]:
        entry = json.loads(attempt.read_text(encoding="utf-8"))
        for name, status in entry["test_outcomes"].items():
            outcomes[name] = {"status": status, "attempt": str(attempt.relative_to(ROOT)).replace("\\", "/"),
                              "source_sha256": entry["source_sha256"]}
        for key, value in entry["counts"].items():
            counts[key] += value
        measurements.update(entry["measurements"])
        attempts.append({"path": str(attempt.relative_to(ROOT)).replace("\\", "/"), "status": entry["status"], "counts": entry["counts"]})
    expected_tests = unittest.defaultTestLoader.getTestCaseNames(module.InstrumentationTests)
    combined = {**report, "status": "passed" if all(outcomes.get(name, {}).get("status") == "ok" for name in expected_tests) else "failed",
                "counts": counts, "measurements": measurements, "attempts": attempts, "latest_test_outcomes": outcomes,
                "provenance": "Each latest test retains exact source hashes; earlier successful work is not silently rerun or reattributed"}
    (ROOT / "results/babylm-instrumentation-cpu-v0.json").write_text(json.dumps(combined, indent=2)+"\n", encoding="utf-8")
    print(text)
    print(json.dumps({"status": report["status"], "this_attempt_counts": report["counts"], "cumulative_counts": counts}, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

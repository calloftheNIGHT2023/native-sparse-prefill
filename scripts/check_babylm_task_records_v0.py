"""Run synthetic pure-function task adapter checks; preserve attempts and sources."""
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
    path = ROOT / "tests/test_babylm_task_records_v0.py"
    spec = importlib.util.spec_from_file_location("test_babylm_task_records_v0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stamp = begin.replace("-", "").replace(":", "").replace(".", "")
    module.ARTIFACT_ROOT = ROOT / "logs" / f"babylm-task-record-fixtures-{stamp}"
    previous_reports = sorted((ROOT / "logs").glob("babylm-task-records-cpu-v0-*.json"))
    suite = (unittest.TestSuite(module.TaskRecordTests(name) for name in args.tests) if args.tests
             else unittest.defaultTestLoader.loadTestsFromTestCase(module.TaskRecordTests))

    def forbid(*args, **kwargs):
        raise RuntimeError("Pure task-record audit must not execute models, backward, or CUDA")

    stream = io.StringIO()
    with mock.patch.object(module.torch.nn.Module, "_call_impl", forbid), \
         mock.patch.object(module.torch.Tensor, "backward", forbid), \
         mock.patch.object(module.torch.cuda, "_lazy_init", forbid):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    end = utc()
    source_paths = [path, Path(__file__), ROOT / "src/babylm_hybrid/official_task_records.py",
                    module.PINNED / "manifest.json"] + [module.SOURCE / name for name in ("read_files.py", "compute_results.py", "run.py")]
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    text = f"started_utc={begin}\ncompleted_utc={end}\n" + stream.getvalue()
    log = ROOT / "logs" / f"babylm-task-records-cpu-v0-{stamp}.log"
    log.write_text(text, encoding="utf-8")
    report = {"status": "passed" if result.wasSuccessful() else "failed", "started_utc": begin, "completed_utc": end,
              "elapsed_seconds": time.perf_counter()-tick, "tests_run": result.testsRun,
              "selected_tests": args.tests, "failures": len(result.failures), "errors": len(result.errors),
              "test_outcomes": dict(re.findall(r"^(test_\w+) \([^\n]*\) \.\.\. (ok|FAIL|ERROR)$", stream.getvalue(), re.MULTILINE)),
              "counts": module.COUNTS, "measurements": module.MEASUREMENTS, "source_sha256": hashes,
              "fixtures": str(module.ARTIFACT_ROOT.relative_to(ROOT)).replace("\\", "/"),
              "test_log": str(log.relative_to(ROOT)).replace("\\", "/"),
              "scope": "Synthetic pure-function normalization/ranking audit; no full-model or real benchmark execution",
              "limitations": ["Not a complete BabyLM suite", "No benchmark files downloaded or scored",
                              "Normalizer is not a complete task-file loader", "Finite valid-score agreement with pinned upstream",
                              "Local seeded ties isolate global RNG; nonfinite and zero-target scores fail closed"]}
    for failed_test, _ in result.failures:
        report["test_outcomes"][getattr(failed_test, "test_case", failed_test)._testMethodName] = "FAIL"
    for errored_test, _ in result.errors:
        report["test_outcomes"][getattr(errored_test, "test_case", errored_test)._testMethodName] = "ERROR"
    attempt_path = ROOT / "logs" / f"babylm-task-records-cpu-v0-{stamp}.json"
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
    expected_tests = unittest.defaultTestLoader.getTestCaseNames(module.TaskRecordTests)
    combined = {**report, "status": "passed" if all(outcomes.get(name, {}).get("status") == "ok" for name in expected_tests) else "failed",
                "counts": counts, "measurements": measurements, "attempts": attempts, "latest_test_outcomes": outcomes,
                "provenance": "Per-test source versions retained; no silent reattribution of older passing tests"}
    (ROOT / "results/babylm-task-records-cpu-v0.json").write_text(json.dumps(combined, indent=2)+"\n", encoding="utf-8")
    print(text)
    print(json.dumps({"status": report["status"], "attempt_counts": report["counts"], "cumulative_counts": counts}, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

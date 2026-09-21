"""Run and archive bounded CPU engineering validation for the D/E reference."""
import hashlib
import importlib.util
import io
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main():
    start, tick = utc(), time.perf_counter()
    path = ROOT / "tests/test_babylm_hybrid_v0.py"
    spec = importlib.util.spec_from_file_location("test_babylm_hybrid_v0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    end = utc()
    source_paths = [path, Path(__file__), ROOT / "configs/babylm-native-sparse-v0.candidate.json"]
    source_paths += sorted((ROOT / "src/babylm_hybrid").glob("*.py"))
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source_paths}
    report = {
        "schema_version": 1, "status": "passed" if result.wasSuccessful() else "failed",
        "scope": "CPU engineering validation only; no scientific pretraining or cost/speed evidence",
        "started_utc": start, "completed_utc": end,
        "elapsed_seconds": time.perf_counter()-tick,
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "counts": module.COUNTS, "measurements": module.MEASUREMENTS,
        "source_sha256": hashes,
        "environment": {"python": platform.python_version(), "torch": module.torch.__version__,
                        "device": "cpu", "threads": module.torch.get_num_threads()},
        "limitations": ["No BabyLM training data consumed", "No GPU forward/backward or throughput evidence",
                        "Synthetic short inputs only", "No equivalence claim to the full Qwen4 architecture",
                        "Numerical tests do not establish language quality or a dense/sparse pretraining gap"],
    }
    log_text = f"started_utc={start}\ncompleted_utc={end}\n" + stream.getvalue()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    stamp = start.replace(":", "").replace("-", "").replace(".", "")
    log_path = ROOT / "logs" / f"babylm-hybrid-cpu-v0-{stamp}.log"
    log_path.write_text(log_text, encoding="utf-8")
    report["test_log"] = str(log_path.relative_to(ROOT)).replace("\\", "/")
    report_path = ROOT / "results/babylm-hybrid-cpu-v0.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    # Every failed attempt remains individually available when the current report is rerun.
    (ROOT / "logs" / f"babylm-hybrid-cpu-v0-{stamp}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(log_text)
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "report": str(report_path), "log": str(log_path)}, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

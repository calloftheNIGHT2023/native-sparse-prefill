"""Recalculate literature-stage budget scenarios, optionally with measured throughput.

This script never launches GPUs or connects to a cloud account.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def estimate(config, measured_tps=None):
    if measured_tps is not None and measured_tps <= 0:
        raise ValueError("Measured tokens/second must be positive")
    rows = []
    for s in config["stages"]:
        tokens = s["methods"] * s["seeds"] * s["tokens_per_run"]
        rates = (s["assumed_single_gpu_tokens_per_second_max"], s["assumed_single_gpu_tokens_per_second_min"])
        if measured_tps is not None:
            rates = (measured_tps, measured_tps)
        train_hours = [tokens / tps / 3600 for tps in rates]
        billed_hours = [h * (1 + s["extra_gpu_time_fraction"]) for h in train_hours]
        price = config["rates_usd_per_gpu_hour"][s["gpu"]]
        costs = [billed_hours[0] * price + s["misc_storage_usd_min"], billed_hours[1] * price + s["misc_storage_usd_max"]]
        rows.append({"stage": s["id"], "runs": s["methods"] * s["seeds"], "total_training_tokens": tokens, "gpu": s["gpu"], "training_gpu_hours_range": train_hours, "gpu_hours_with_reserve_range": billed_hours, "usd_range_with_reserve_and_storage": costs, "throughput_status": "user-supplied override" if measured_tps is not None else "unmeasured planning range"})
    profile = config["profiling"]
    profile_usd = [profile[k] * config["rates_usd_per_gpu_hour"][profile["gpu"]] for k in ("hours_min", "hours_max")]
    first = [rows[0]["usd_range_with_reserve_and_storage"][i] + profile_usd[i] for i in (0, 1)]
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), "price_source": config["price_source"], "profiling_gpu_usd_range": profile_usd, "stages": rows, "pilot_plus_profiling_usd_range": first, "note": "Unmeasured estimates, not a quotation, confidence interval, experiment result, or spending authorization."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/native-sparse-budget-v0.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens-per-second", type=float, help="Explicit scenario override; use stage-specific config for actual measured comparisons")
    args = parser.parse_args()
    result = estimate(json.loads(args.config.read_text(encoding="utf-8")), args.tokens_per_second)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)

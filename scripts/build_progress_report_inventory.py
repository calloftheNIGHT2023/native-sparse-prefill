"""Build the 2026-09-15 progress report's inventory from archived results only."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-09-15"
REPORT = ROOT / "docs" / f"research-progress-report-{DATE}.md"
INDEX = ROOT / "docs" / f"research-progress-training-index-{DATE}.md"
OUT = ROOT / "results" / f"research-progress-report-{DATE}"
SOURCES = [
    "cloud-amp-recovery-final-evidence-v0",
    "cloud-amp-lr-boundary-evidence-v0",
    "cloud-amp-confirmation-evidence-v1",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for archive in SOURCES:
        for path in (ROOT / "results" / archive / "results").rglob("result.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("phase") != "train" or raw.get("status") != "complete":
                continue
            identity = raw["identity"]
            updates = raw["optimizer_updates_this_process"]
            trace = raw["trace"]
            assert updates == raw["step"] == 256, path
            assert len(trace) == 256 and [t["step"] for t in trace] == list(range(1, 257)), path
            assert abs(sum(t["seconds"] for t in trace) - raw["training_seconds"]) < 0.001, path
            rows.append({
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": digest(path),
                "k": identity["k"], "seed": identity["seed"], "lr": identity["lr"],
                "started_utc": raw["started_utc"], "finished_utc": raw["finished_utc"],
                "optimizer_updates": updates, "trace_rows": len(trace),
                "training_seconds": raw["training_seconds"],
                "process_wall_seconds": raw["wall_seconds"],
                "config_sha256": identity["config_sha256"],
                "initial_sha256": raw["initial_sha256"],
            })
    rows.sort(key=lambda row: row["started_utc"])
    assert len(rows) == 15 and sum(r["optimizer_updates"] for r in rows) == 3840
    assert len({r["path"] for r in rows}) == 15

    grouped = []
    for k in [0, 4, 16]:
        group = [r for r in rows if r["k"] == k]
        grouped.append({
            "k": k, "trajectories": len(group),
            "optimizer_updates": sum(r["optimizer_updates"] for r in group),
            "training_seconds": sum(r["training_seconds"] for r in group),
            "process_wall_seconds": sum(r["process_wall_seconds"] for r in group),
        })
    assert [g["trajectories"] for g in grouped] == [5, 4, 6]
    stages = {"ability_pilot": 128, "primary_quality": 1536, "runtime_switch": 1536,
              "layer_switch": 1152, "matched_lr_and_base": 2304,
              "fixed_budget_route": 144, "additive_route": 144}
    assert sum(stages.values()) == 6944

    lines = ["# 当前AMP主实验：15条训练与日志索引", "",
             "本附件从三个本地证据镜像的原始 result.json 自动生成。仅统计完成的正式训练，不包括预检、评测或更早的小模型/2K/8K阶段。", "",
             "所有时间为2026年9月15日UTC；美国东部时间减4小时，北京时间加8小时。表内开始/结束采用模型进程记录，可能比控制器启动晚几秒。每条原始结果含256条逐步记录（损失、梯度范数、学习率、窗口编号、耗时及UTC）。", "",
             "|序号|注意力|种子|峰值LR|开始UTC|结束UTC|更新|训练秒|进程秒|原始日志|",
             "|---|---|---:|---:|---|---|---:|---:|---:|---|"]
    for i, r in enumerate(rows, 1):
        label = "密集" if r["k"] == 0 else f"K{r['k']}"
        start = datetime.fromisoformat(r["started_utc"]).strftime("%H:%M:%S")
        end = datetime.fromisoformat(r["finished_utc"]).strftime("%H:%M:%S")
        target = (ROOT / r["path"]).as_posix()
        lines.append(f"|{i}|{label}|{r['seed']}|{r['lr']:g}|{start}|{end}|256|"
                     f"{r['training_seconds']:.3f}|{r['process_wall_seconds']:.3f}|[result.json]({target})|")
    lines += ["", "## 统计口径与复核", "",
              f"- 正式轨迹：{len(rows)}；正式更新：3840；逐步日志：3840条。",
              f"- 训练循环累计：{sum(r['training_seconds'] for r in rows)/60:.6f}分钟。",
              f"- 训练进程累计：{sum(r['process_wall_seconds'] for r in rows)/60:.6f}分钟。",
              "- 密集5条、K4四条、K16六条。K16包括旧LR=0.0003第二种子复跑，不能因最终未选中而从历史开销中删除。",
              "- 43次预检/诊断更新按阶段记录另列，不包含在3840条正式逐步日志中。",
              "- 计时不等于账单；没有累计安装、独立评测、迁移、空闲和存储费用。",
              "- 本次逐条检查了256个连续step编号，并将逐步耗时之和与training_seconds复核；原始结果及使用的报告文件SHA256在清单中。",
              "", "## 原始结果位置与SHA256", "",
              "|序号|项目相对路径|SHA256|", "|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"|{i}|`{r['path']}`|`{r['sha256']}`|")
    lines += ["", f"[回到阶段报告]({REPORT.as_posix()})", ""]
    INDEX.write_text("\n".join(lines), encoding="utf-8")

    # Normalize Windows file links to drive-qualified absolute paths.
    report_text = REPORT.read_text(encoding="utf-8").replace("(/D:/", "(D:/")
    REPORT.write_text(report_text, encoding="utf-8")
    linked_paths = sorted(set(re.findall(r"\]\((D:/[^)]+)\)", report_text)))
    manifest_path = OUT / "evidence-manifest.json"
    missing = [p for p in linked_paths if Path(p) != manifest_path and not Path(p).exists()]
    assert not missing, missing
    stable_sources = []
    for p in linked_paths:
        path = Path(p)
        if path == manifest_path or path.name == "TIMELINE.md":
            continue
        stable_sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
    archives = []
    for name in ["task-route-chain-evidence-v0.tar.gz", "task-route-additive-evidence-v1.tar.gz"]:
        path = ROOT / "exports" / name
        archives.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
                         "bytes": path.stat().st_size,
                         "check_this_report": "file SHA256 only; extraction verification recorded in prior stage"})
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_cutoff_utc": "2026-09-15T23:15:33.894730+00:00",
        "scope": "Current AMP stage inventory; not all-project training or cloud billing",
        "formal_training_trajectories": 15, "formal_optimizer_updates": 3840,
        "diagnostic_optimizer_updates_from_stage_records": 43,
        "unique_training_target_positions": 1048576,
        "formal_training_target_exposures": 62914560,
        "training_seconds": sum(r["training_seconds"] for r in rows),
        "process_wall_seconds": sum(r["process_wall_seconds"] for r in rows),
        "billing_usd": None, "task_scored_forwards_by_stage": stages,
        "task_scored_forwards_total": 6944,
        "prediction_count_note": "Includes same questions across seeds, conditions and replays; not unique items.",
        "by_attention": grouped, "runs": rows, "source_files": stable_sources,
        "archives": archives,
        "report": {"path": REPORT.relative_to(ROOT).as_posix(), "sha256": digest(REPORT)},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "runs": 15, "trace_rows": 3840,
                      "report_characters": len(report_text), "local_links_checked": len(linked_paths),
                      "training_minutes": manifest["training_seconds"]/60,
                      "process_minutes": manifest["process_wall_seconds"]/60,
                      "manifest": str(manifest_path)}, ensure_ascii=True))


if __name__ == "__main__":
    main()

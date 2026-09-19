"""Index the already-frozen development ledger; no test data or tokenizer fit."""
from __future__ import annotations
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_babylm_windows_v0 import (
    COLUMNS, HEADER, make_windows, record_slice, word_token_intervals,
    partial_words, retained_edges, sha, utc,
)

RAW = ROOT / "data/babylm-dev-raw-v0"
LEDGER = ROOT / "data/babylm-dev-token-ledger-v0"
OUT = ROOT / "data/babylm-dev-windows-v0"


def select_fast_panel(index, summaries, per_source=8):
    """Select only by source/offset hash; no model scores or length filtering."""
    chosen = []
    for source in summaries:
        sid, name = source["source_index"], source["source"]
        candidates = []
        for idx in np.flatnonzero(index[:, 0] == sid):
            row = index[int(idx)]
            payload = f"babylm-dev-panel-v0|{name}|{int(row[2])}|{int(row[3])}"
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            candidates.append((digest, int(row[2]), int(row[3]), int(idx)))
        for rank, (digest, start, end, idx) in enumerate(sorted(candidates)[:per_source]):
            row = index[idx]
            chosen.append({"window_index": idx, "source_index": sid, "source": name,
                           "rank_in_source_zero_based": rank, "selection_sha256": digest,
                           "source_token_start": start, "source_token_end": end,
                           "word_exposures": int(row[8]), "input_tokens": int(row[9]),
                           "loss_tokens": int(row[10])})
    return {"panel_id": "babylm-dev-panel-v0", "selection_rule":
            "Up to 8 smallest SHA256('babylm-dev-panel-v0|source|start|end') per source; tie-break start,end,index; UTF-8; no score/length filtering",
            "per_source_maximum": per_source, "window_indices": [r["window_index"] for r in chosen],
            "selected_windows": len(chosen), "selected": chosen,
            "input_tokens": sum(r["input_tokens"] for r in chosen),
            "word_exposures": sum(r["word_exposures"] for r in chosen),
            "loss_tokens": sum(r["loss_tokens"] for r in chosen),
            "scope": "Fixed small internal dev NLL panel; not the official BabyLM fast benchmark or a substitute for full dev/final evaluation"}


def build():
    started, timer = utc(), time.perf_counter()
    ledger_path = LEDGER / "manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["status"] == "frozen_tokenizer_dev_record_ledger_verified"
    assert not ledger["tokenizer_retrained"] and not ledger["final_test_read"]
    for artifact in ledger["artifacts"]:
        path = LEDGER / artifact["path"]
        assert path.parent == LEDGER and sha(path) == artifact["sha256"]
    tokenizer_path = ROOT / ledger["tokenizer_path_relative_to_project"]
    assert tokenizer_path.is_relative_to(ROOT / "data/babylm-2026-tokenizer-16k-v0")
    assert sha(tokenizer_path) == ledger["tokenizer_sha256"]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    OUT.mkdir(exist_ok=True)
    target = OUT / "manifest.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        assert old["script_sha256"] == sha(Path(__file__))
        assert old["dev_ledger_manifest_sha256"] == sha(ledger_path)
        for artifact in old["artifacts"]:
            assert sha(OUT / artifact["path"]) == artifact["sha256"]
        print(json.dumps({"status": "already_verified", "manifest": str(target)}))
        return
    lock = OUT / "build.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "utc": started}, stream)
    try:
        rows, summaries, boundaries = [], [], []
        histogram = Counter()
        reencoded_records = 0
        for sid, source in enumerate(ledger["files"]):
            path = RAW / source["source"]
            assert path.parent == RAW and path.name.endswith(".dev")
            assert sha(path) == source["source_sha256"]
            with path.open("r", encoding="utf-8", newline="") as stream:
                texts = stream.readlines()
            offsets = np.load(LEDGER / source["offsets_file"], mmap_mode="r")
            words = np.load(LEDGER / source["words_file"], mmap_mode="r")
            ids = np.memmap(LEDGER / source["ids_file"], dtype="<u4", mode="r")
            assert len(texts) == len(words) == len(offsets) - 1
            # Blank lines have whitespace tokens and MUST survive this policy.
            assert np.all(np.diff(offsets) > 0) and int(offsets[-1]) == len(ids)
            assert all(len(text.split()) == int(word_count) for text, word_count in zip(texts, words))
            prefix = np.concatenate([np.array([0], dtype=np.uint64), np.cumsum(words, dtype=np.uint64)])
            headers = [i for i, text in enumerate(texts) if HEADER.match(text.strip())]
            header_starts = [int(offsets[i]) for i in headers]
            boundaries.append({"source_index": sid, "source": path.name,
                               "header_record_indices_zero_based": headers,
                               "header_source_token_starts": header_starts})
            cache = {}

            def count_part(record, left, right):
                size = int(offsets[record + 1] - offsets[record])
                assert 0 <= left < right <= size
                if left == 0 and right == size:
                    return int(words[record])
                if record not in cache:
                    encoding = tokenizer.encode(texts[record], add_special_tokens=False)
                    assert np.array_equal(np.asarray(encoding.ids, dtype=np.uint32),
                                          ids[int(offsets[record]):int(offsets[record + 1])])
                    assert tokenizer.decode(encoding.ids, skip_special_tokens=False) == texts[record]
                    cache[record] = word_token_intervals(texts[record], encoding)
                return partial_words(cache[record], left, right)

            local_rows, previous_end = [], 0
            for segment, start, end in make_windows(len(ids), header_starts, 2048):
                assert start == previous_end
                previous_end = end
                first, last, left, right = record_slice(offsets, start, end)
                if first == last:
                    exposure = count_part(first, left, right)
                else:
                    exposure = count_part(first, left, int(offsets[first + 1] - offsets[first]))
                    exposure += int(prefix[last] - prefix[first + 1])
                    exposure += count_part(last, 0, right)
                length = end - start
                local_rows.append([sid, segment, start, end, first, last,
                                   left, right, exposure, length, length - 1])
                histogram[length] += 1
            assert previous_end == len(ids)
            total_words = sum(row[8] for row in local_rows)
            duplicate_cuts = 0
            for row in local_rows[:-1]:
                end, record = row[3], row[5]
                if end < int(offsets[record + 1]):
                    cut = end - int(offsets[record])
                    spans = cache[record]
                    duplicate_cuts += int(np.count_nonzero((spans[:, 0] < cut) & (spans[:, 1] > cut)))
            assert total_words - int(prefix[-1]) == duplicate_cuts
            rows.extend(local_rows)
            reencoded_records += len(cache)
            summaries.append({"source_index": sid, "source": path.name,
                              "source_sha256": source["source_sha256"],
                              "token_ids_path_relative_to_project": (LEDGER / source["ids_file"]).relative_to(ROOT).as_posix(),
                              "token_ids_sha256": sha(LEDGER / source["ids_file"]),
                              "record_offsets_path_relative_to_project": (LEDGER / source["offsets_file"]).relative_to(ROOT).as_posix(),
                              "source_records": len(texts), "zero_word_records_preserved": int(np.count_nonzero(words == 0)),
                              "explicit_header_records": len(headers), "segments": len(set([0] + header_starts)),
                              "windows": len(local_rows), "original_words": int(prefix[-1]),
                              "window_word_exposures": total_words, "extra_word_exposures_due_to_cuts": duplicate_cuts,
                              "input_tokens": len(ids), "next_token_loss_positions": len(ids) - len(local_rows)})
            print(json.dumps({"event": "dev_source_windows_verified", **summaries[-1]}), flush=True)
        index = np.asarray(rows, dtype=np.uint64)
        assert index.shape == (len(rows), len(COLUMNS))
        assert int(index[:, 9].sum()) == ledger["total_text_tokens"]
        assert sum(row["original_words"] for row in summaries) == ledger["total_source_words"]
        assert sum(row["zero_word_records_preserved"] for row in summaries) == sum(r["zero_word_records"] for r in ledger["files"])
        assert np.all(index[:, 3] - index[:, 2] == index[:, 9])
        assert np.all(index[:, 9] - 1 == index[:, 10])
        np.save(OUT / "windows.u64.npy", index)
        np.testing.assert_array_equal(index, np.load(OUT / "windows.u64.npy"))
        (OUT / "known-boundaries.json").write_text(json.dumps(boundaries, indent=2) + "\n", encoding="utf-8")
        (OUT / "window-length-histogram.json").write_text(json.dumps(dict(sorted(histogram.items())), indent=2) + "\n", encoding="utf-8")
        panel = select_fast_panel(index, summaries)
        assert len(panel["window_indices"]) == len(set(panel["window_indices"]))
        assert panel["selected_windows"] <= 48
        (OUT / "fast-panel.json").write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
        dense = sum(n * (n + 1) // 2 * count for n, count in histogram.items())
        kept = sum(retained_edges(n) * count for n, count in histogram.items())
        artifacts = [OUT / name for name in ["windows.u64.npy", "known-boundaries.json", "window-length-histogram.json", "fast-panel.json"]]
        result = {"status": "window_index_word_accounting_and_coverage_verified", "split": "dev",
                  "started_utc": started, "completed_utc": utc(), "elapsed_seconds": time.perf_counter() - timer,
                  "script_sha256": sha(Path(__file__)), "shared_window_helpers_sha256": sha(ROOT / "scripts/prepare_babylm_windows_v0.py"),
                  "dev_ledger_manifest_sha256": sha(ledger_path), "source_revision": ledger["dev_revision"],
                  "tokenizer_sha256": ledger["tokenizer_sha256"], "tokenizer_retrained": False,
                  "columns": COLUMNS, "index_dtype": "uint64", "index_base": 0,
                  "interval_convention": "token starts inclusive, token ends exclusive; first/last records inclusive",
                  "max_window_tokens": 2048, "total_windows": len(index),
                  "total_original_words": ledger["total_source_words"], "word_exposures_per_pass": int(index[:, 8].sum()),
                  "extra_word_exposures_per_pass": int(index[:, 8].sum()) - ledger["total_source_words"],
                  "input_tokens_per_pass": int(index[:, 9].sum()), "next_token_loss_positions_per_pass": int(index[:, 10].sum()),
                  "zero_word_records_preserved": sum(r["zero_word_records_preserved"] for r in summaries),
                  "record_offset_reencodings": reencoded_records,
                  "tail_windows_retained": sum(count for n, count in histogram.items() if n < 2048),
                  "tokens_dropped": 0, "tokens_duplicated": 0, "inserted_bos_eos": 0,
                  "window_layout": {"single_segment_per_window": True, "reset_before_every_window":
                                    ["position", "attention", "GDN_recurrent_state", "GDN_short_convolution_state"],
                                    "reader": "scripts.prepare_babylm_windows_v0.iter_windows", "cross_window_attention": False},
                  "metadata_header_policy": "retain original tokens and words; reset before header; header belongs to following segment",
                  "continuity": "files and explicit headers reset; unknown upstream joins remain unverified",
                  "loss_policy": "all tokens are inputs; only within-window next-token targets, no target after final token",
                  "qsa64_block4_support_cardinality": {"dense_causal_edges_per_pass": dense, "retained_edges_per_pass": kept,
                                                       "retained_edge_fraction": kept / dense,
                                                       "scope": "input layout support cardinality only; not measured speed"},
                  "source_summaries": summaries, "fast_panel": "fast-panel.json",
                  "scientific_model_training_runs": 0, "scientific_optimizer_updates": 0,
                  "model_forward_calls": 0, "final_test_read": False, "gpu_hours": 0,
                  "artifacts": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in artifacts]}
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "windows": len(index), "tokens": result["input_tokens_per_pass"],
                          "loss_tokens": result["next_token_loss_positions_per_pass"], "word_exposures": result["word_exposures_per_pass"],
                          "panel_windows": panel["selected_windows"], "panel_loss_tokens": panel["loss_tokens"]}), flush=True)
    except BaseException as exc:
        (OUT / f"failure-{time.time_ns()}.json").write_text(json.dumps({"utc": utc(), "type": type(exc).__name__, "error": str(exc)}, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    build()

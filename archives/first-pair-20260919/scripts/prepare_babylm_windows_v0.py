"""Build a conservative word-accounted window INDEX, never new training tokens.

No tokenizer training, model, dev/test input, or GPU use. Physical source lines
are not document boundaries. Explicit headers and files are reset boundaries;
hidden upstream sampling joins remain unknown. Only cut records are encoded
again, using the fixed tokenizer, to recover verified character offsets.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "4")
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/babylm-2026-strict-small-raw-v0"
TOK = ROOT / "data/babylm-2026-tokenizer-16k-v0"
DEV_TOK = ROOT / "data/babylm-dev-token-ledger-v0"
OUT = ROOT / "data/babylm-2026-windows-v0"
HEADER = re.compile(r"^=\s*=\s*=.*=\s*=\s*=$")
MAX_LENGTH = 2048
WORD_LIMIT = 100_000_000
COLUMNS = ["source_index", "segment_index_in_source", "source_token_start",
           "source_token_end", "first_record_index", "last_record_index",
           "first_record_token_start", "last_record_token_end",
           "word_exposures", "input_tokens", "next_token_loss_positions"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def retained_edges(length, selected_blocks=64, block_size=4):
    # Current block IS eligible once its last token has become causal.
    return sum(min(visible // block_size, selected_blocks) * block_size
               + visible % block_size for visible in range(1, length + 1))


def make_windows(total_tokens, header_token_starts, max_length=MAX_LENGTH):
    assert total_tokens > 0 and max_length > 0
    resets = sorted(set([0, total_tokens] + list(header_token_starts)))
    assert resets[0] == 0 and resets[-1] == total_tokens
    for segment, (left, right) in enumerate(zip(resets, resets[1:])):
        assert 0 <= left < right <= total_tokens
        for start in range(left, right, max_length):
            yield segment, start, min(start + max_length, right)


def record_slice(offsets, start, end):
    assert 0 <= start < end <= int(offsets[-1])
    first = int(np.searchsorted(offsets, start, side="right")) - 1
    last = int(np.searchsorted(offsets, end - 1, side="right")) - 1
    return first, last, start - int(offsets[first]), end - int(offsets[last])


def word_token_intervals(text, encoding):
    """Map every whitespace word to all intersecting contiguous token indices.

    Offsets are character offsets supplied by the fixed tokenizer. Byte-level
    Unicode tokens may share offsets; they must count toward the same word.
    We reject unmapped words or nonmonotone offsets rather than estimating.
    """
    offsets = encoding.offsets
    assert len(offsets) == len(encoding.ids)
    for i, (start, end) in enumerate(offsets):
        assert 0 <= start <= end <= len(text)
        if i:
            assert offsets[i - 1][0] <= start and offsets[i - 1][1] <= end
    intervals = []
    cursor = 0
    for match in re.finditer(r"\S+", text):
        left, right = match.span()
        while cursor < len(offsets) and offsets[cursor][1] <= left:
            cursor += 1
        hit = cursor
        while hit < len(offsets) and offsets[hit][0] < right:
            hit += 1
        assert hit > cursor, f"Whitespace word has no token offset: {left}:{right}"
        # Reject internal empty/nonoverlapping tokens, which would invalidate
        # the compact interval representation (not observed in this tokenizer).
        assert all(offsets[t][1] > left and offsets[t][0] < right
                   for t in range(cursor, hit))
        intervals.append((cursor, hit))
    assert len(intervals) == len(text.split())
    return np.asarray(intervals, dtype=np.int64).reshape(-1, 2)


def partial_words(intervals, start, end):
    assert 0 <= start < end
    return int(np.count_nonzero((intervals[:, 0] < end) & (intervals[:, 1] > start)))


def iter_windows(manifest_path=OUT / "manifest.json", window_indices=None,
                 verify_hashes=True):
    """Yield readonly NumPy token views plus accounting; never train a model.

    Default order is the manifest's canonical file/segment/token order. Callers
    may supply a frozen index order, but must enforce the word budget in that
    order. Every item is a SINGLE reset segment; collating several items must
    preserve their attention, position, recurrent and convolution boundaries.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "window_index_word_accounting_and_coverage_verified"
    assert manifest["columns"] == COLUMNS
    if verify_hashes:
        for artifact in manifest["artifacts"]:
            path = manifest_path.parent / artifact["path"]
            assert path.parent == manifest_path.parent and sha(path) == artifact["sha256"]
    index = np.load(manifest_path.parent / "windows.u64.npy", mmap_mode="r")
    assert index.shape == (manifest["total_windows"], len(COLUMNS))
    if window_indices is None:
        window_indices = range(len(index))
    streams = {}
    for window_index in window_indices:
        assert isinstance(window_index, (int, np.integer))
        window_index = int(window_index)
        assert 0 <= window_index < len(index)
        row = [int(v) for v in index[window_index]]
        source, segment, start, end, first, last, left, right, words, n, loss = row
        if source not in streams:
            meta = manifest["source_summaries"][source]
            path = (ROOT / meta["token_ids_path_relative_to_project"]).resolve()
            assert any(path.is_relative_to(root.resolve()) for root in (TOK, DEV_TOK))
            assert path.name.endswith(".ids.u32")
            if verify_hashes:
                assert sha(path) == meta["token_ids_sha256"]
            streams[source] = np.memmap(path, dtype="<u4", mode="r")
        token_view = streams[source][start:end]
        assert len(token_view) == n and not token_view.flags.writeable
        yield {"window_index": window_index, "input_ids": token_view,
               "source_index": source, "segment_index_in_source": segment,
               "source_token_start": start, "source_token_end": end,
               "first_record_index": first, "last_record_index": last,
               "first_record_token_start": left, "last_record_token_end": right,
               "word_exposures": words, "input_tokens": n, "loss_tokens": loss,
               "single_segment": True, "reset_model_state_before": True}


def self_test(tokenizer):
    tests = []
    # Exposes splitting inside a word, leading/trailing whitespace, tabs,
    # newlines, and byte tokens with overlapping Unicode character offsets.
    for text in ["abcdefghijklmnopqrstuvwxyzzebra x\n", "  a\tb c  \r\n",
                 "é中🙂é café\n", "\n", "a b"]:
        enc = tokenizer.encode(text, add_special_tokens=False)
        intervals = word_token_intervals(text, enc)
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        for start in range(len(enc.ids)):
            for end in range(start + 1, len(enc.ids) + 1):
                expected = sum(any(enc.offsets[t][1] > a and enc.offsets[t][0] < b
                                   for t in range(start, end)) for a, b in spans)
                assert partial_words(intervals, start, end) == expected
        assert partial_words(intervals, 0, len(enc.ids)) == len(text.split())
    tests.append("every_subslice_matches_bruteforce_word_token_overlap_including_unicode")
    enc = tokenizer.encode("abcdefghijklmnopqrstuvwxyzzebra", add_special_tokens=False)
    assert len(enc.ids) > 1
    intervals = word_token_intervals("abcdefghijklmnopqrstuvwxyzzebra", enc)
    assert sum(partial_words(intervals, i, i + 1) for i in range(len(enc.ids))) > 1
    tests.append("word_cut_across_windows_counted_in_every_window")
    result = list(make_windows(13, [0, 3, 11], 4))
    assert result == [(0, 0, 3), (1, 3, 7), (1, 7, 11), (2, 11, 13)]
    assert sum(end - start for _, start, end in result) == 13
    off = np.array([0, 3, 8, 13], dtype=np.uint64)
    assert record_slice(off, 3, 8) == (1, 1, 0, 5)
    assert record_slice(off, 2, 9) == (0, 2, 2, 1)
    tests.append("known_boundaries_tail_coverage_and_exclusive_record_ends")
    for length in [1, 3, 4, 7, 8, 255, 256, 257, 259, 260, 2048]:
        brute = 0
        for query in range(length):
            blocks = [range(b * 4, b * 4 + 4) for b in range((query + 1) // 4)]
            chosen = blocks[:64]  # Top-k identity changes values, not support cardinality.
            positions = {p for block in chosen for p in block}
            positions.update(range(((query + 1) // 4) * 4, query + 1))
            assert all(p <= query for p in positions)
            brute += len(positions)
        assert retained_edges(length) == brute
    assert retained_edges(260) - retained_edges(259) == 256
    tests.append("density_enumeration_includes_current_completed_block_not_mandatory")
    return tests


def build():
    started, timer = utc(), time.perf_counter()
    manifest_path = TOK / "manifest.json"
    tok_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert tok_manifest["status"] == "tokenizer_and_unpacked_record_ledger_verified"
    assert tok_manifest["total_source_words"] == 10_000_000
    for artifact in tok_manifest["artifacts"]:
        path = TOK / artifact["path"]
        assert path.parent == TOK and sha(path) == artifact["sha256"]
    tokenizer = Tokenizer.from_file(str(TOK / "tokenizer.json"))
    tests = self_test(tokenizer)
    OUT.mkdir(exist_ok=True)
    target = OUT / "manifest.json"
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        assert previous["script_sha256"] == sha(Path(__file__))
        assert previous["tokenizer_manifest_sha256"] == sha(manifest_path)
        for artifact in previous["artifacts"]:
            assert sha(OUT / artifact["path"]) == artifact["sha256"]
        print(json.dumps({"status": "already_verified", "self_tests": tests}))
        return
    lock = OUT / "build.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"utc": started, "pid": os.getpid()}, stream)
    try:
        all_rows, source_summaries, boundary_rows = [], [], []
        length_histogram = Counter()
        total_reencoded, total_reencoded_tokens = 0, 0
        for source_index, source in enumerate(tok_manifest["files"]):
            raw = RAW / source["source"]
            assert raw.name.endswith(".train.txt") and raw.parent == RAW
            assert sha(raw) == source["source_sha256"]
            with raw.open("r", encoding="utf-8", newline="") as stream:
                texts = stream.readlines()
            offsets = np.load(TOK / source["offsets_file"], mmap_mode="r")
            words = np.load(TOK / source["words_file"], mmap_mode="r")
            ids = np.memmap(TOK / source["ids_file"], dtype="<u4", mode="r")
            assert len(texts) == len(words) == len(offsets) - 1
            assert np.all(np.diff(offsets) > 0) and int(offsets[-1]) == len(ids)
            assert all(len(text.split()) == int(n) for text, n in zip(texts, words))
            word_prefix = np.concatenate([np.array([0], dtype=np.uint64),
                                          np.cumsum(words, dtype=np.uint64)])
            header_records = [i for i, text in enumerate(texts) if HEADER.match(text.strip())]
            header_starts = [int(offsets[i]) for i in header_records]
            boundary_rows.append({"source_index": source_index, "source": raw.name,
                                  "header_record_indices_zero_based": header_records,
                                  "header_source_token_starts": header_starts})
            mapping_cache = {}

            def count_part(record, left, right):
                nonlocal total_reencoded, total_reencoded_tokens
                size = int(offsets[record + 1] - offsets[record])
                assert 0 <= left < right <= size
                if left == 0 and right == size:
                    return int(words[record])
                if record not in mapping_cache:
                    enc = tokenizer.encode(texts[record], add_special_tokens=False)
                    existing = ids[int(offsets[record]):int(offsets[record + 1])]
                    assert np.array_equal(np.asarray(enc.ids, dtype=np.uint32), existing)
                    assert tokenizer.decode(enc.ids, skip_special_tokens=False) == texts[record]
                    mapping_cache[record] = word_token_intervals(texts[record], enc)
                    total_reencoded += 1
                    total_reencoded_tokens += len(enc.ids)
                return partial_words(mapping_cache[record], left, right)

            source_rows = []
            expected_start = 0
            for segment, start, end in make_windows(len(ids), header_starts):
                assert start == expected_start
                expected_start = end
                first, last, left, right = record_slice(offsets, start, end)
                if first == last:
                    exposure = count_part(first, left, right)
                else:
                    exposure = count_part(first, left, int(offsets[first + 1] - offsets[first]))
                    exposure += int(word_prefix[last] - word_prefix[first + 1])
                    exposure += count_part(last, 0, right)
                n = end - start
                source_rows.append([source_index, segment, start, end, first, last,
                                    left, right, exposure, n, max(n - 1, 0)])
                length_histogram[n] += 1
            assert expected_start == len(ids)
            all_rows.extend(source_rows)
            exposure = sum(row[8] for row in source_rows)
            original_words = int(word_prefix[-1])
            assert exposure >= original_words
            # Independent boundary check: the excess equals each word touched
            # on both sides of every cut INSIDE a physical record.
            duplicated_at_cuts = 0
            for row in source_rows[:-1]:
                end, record = row[3], row[5]
                if end < int(offsets[record + 1]):
                    cut = end - int(offsets[record])
                    intervals = mapping_cache[record]
                    duplicated_at_cuts += int(np.count_nonzero(
                        (intervals[:, 0] < cut) & (intervals[:, 1] > cut)))
            assert exposure - original_words == duplicated_at_cuts
            source_summaries.append({
                "source_index": source_index, "source": raw.name,
                "source_sha256": source["source_sha256"],
                "token_ids_path_relative_to_project": str((TOK / source["ids_file"]).relative_to(ROOT)).replace("\\", "/"),
                "token_ids_sha256": sha(TOK / source["ids_file"]),
                "record_offsets_path_relative_to_project": str((TOK / source["offsets_file"]).relative_to(ROOT)).replace("\\", "/"),
                "source_records": len(texts), "explicit_header_records": len(header_records),
                "segments": len(set([0] + header_starts)), "windows": len(source_rows),
                "original_words": original_words, "window_word_exposures": exposure,
                "extra_word_exposures_due_to_cuts": duplicated_at_cuts,
                "input_tokens": len(ids), "next_token_loss_positions": len(ids) - len(source_rows),
                "offset_reencoded_records": len(mapping_cache),
            })
            print(json.dumps({"event": "source_windows_verified", **source_summaries[-1]}), flush=True)
        array = np.asarray(all_rows, dtype=np.uint64)
        assert array.shape == (len(all_rows), len(COLUMNS))
        total_tokens, exposure_words = int(array[:, 9].sum()), int(array[:, 8].sum())
        assert total_tokens == tok_manifest["total_text_tokens"]
        assert sum(x["original_words"] for x in source_summaries) == 10_000_000
        assert np.all(array[:, 3] - array[:, 2] == array[:, 9])
        assert np.all(array[:, 9] - 1 == array[:, 10])
        assert int(array[:, 10].sum()) == total_tokens - len(array)
        np.save(OUT / "windows.u64.npy", array)
        np.testing.assert_array_equal(np.load(OUT / "windows.u64.npy"), array)
        boundaries_path = OUT / "known-boundaries.json"
        boundaries_path.write_text(json.dumps(boundary_rows, indent=2) + "\n", encoding="utf-8")
        hist_path = OUT / "window-length-histogram.json"
        hist_path.write_text(json.dumps(dict(sorted(length_histogram.items())), indent=2) + "\n", encoding="utf-8")
        dense = sum(n * (n + 1) // 2 * count for n, count in length_histogram.items())
        kept = sum(retained_edges(n) * count for n, count in length_histogram.items())
        nontrivial_queries = sum(max(0, n - 259) * count for n, count in length_histogram.items())
        full_passes = WORD_LIMIT // exposure_words
        left = WORD_LIMIT - full_passes * exposure_words
        partial_pass_windows = 0
        for row in array:
            count = int(row[8])
            if count > left:
                break
            left -= count
            partial_pass_windows += 1
        budget_plan = {"word_limit": WORD_LIMIT, "full_canonical_passes": full_passes,
                       "extra_prefix_windows_in_next_pass": partial_pass_windows,
                       "accounted_word_exposures": WORD_LIMIT - left,
                       "unused_word_budget": left,
                       "scope": "Illustrative fixed-order maximal prefix; train sampler must enforce per-window budget for its frozen order; not executed training"}
        artifacts = [OUT / "windows.u64.npy", boundaries_path, hist_path]
        result = {
            "status": "window_index_word_accounting_and_coverage_verified",
            "started_utc": started, "completed_utc": utc(),
            "elapsed_seconds": time.perf_counter() - timer,
            "script_sha256": sha(Path(__file__)),
            "tokenizer_manifest_sha256": sha(manifest_path),
            "tokenizer_sha256": sha(TOK / "tokenizer.json"),
            "source_revision": tok_manifest["source_revision"],
            "columns": COLUMNS, "index_dtype": "uint64", "index_base": 0,
            "interval_convention": "token starts inclusive, token ends exclusive; first/last records inclusive",
            "max_window_tokens": MAX_LENGTH, "total_windows": len(array),
            "window_layout": {"single_segment_per_window": True,
                              "reset_before_every_window": ["position", "attention", "GDN_recurrent_state", "GDN_short_convolution_state"],
                              "reader": "scripts.prepare_babylm_windows_v0.iter_windows",
                              "input_ids_type": "readonly numpy uint32 view; caller chooses safe tensor dtype",
                              "cross_window_attention": False},
            "total_original_words": 10_000_000, "word_exposures_per_pass": exposure_words,
            "extra_word_exposures_per_pass": exposure_words - 10_000_000,
            "ten_passes_would_exceed_100M_by_words": 10 * exposure_words - WORD_LIMIT,
            "input_tokens_per_pass": total_tokens,
            "next_token_loss_positions_per_pass": int(array[:, 10].sum()),
            "tail_windows_retained": sum(count for n, count in length_histogram.items() if n < MAX_LENGTH),
            "tokens_dropped": 0, "tokens_duplicated": 0, "inserted_bos_eos": 0,
            "metadata_header_policy": "retain original tokens and words; reset before header; header belongs to following segment",
            "continuity": "reset at files and explicit headers; hidden upstream joins unknown; 2048 is an input length, not verified semantic context",
            "loss_policy": "input all window tokens; labels are in-window next tokens; final token has no next-token target; no cross-window prediction",
            "word_accounting": "Original whitespace word counted once per window whose tokens overlap it, using verified tokenizer character offsets only for cut records",
            "record_offset_reencodings": total_reencoded,
            "record_offset_reencoded_tokens": total_reencoded_tokens,
            "offset_reencoding_changes_tokens": False,
            "qsa64_block4_support_cardinality": {
                "dense_causal_edges_per_pass": dense, "retained_edges_per_pass": kept,
                "retained_edge_fraction": kept / dense,
                "queries_with_support_drop": nontrivial_queries,
                "fraction_queries_with_support_drop": nontrivial_queries / total_tokens,
                "formula": "sum_q [4*min(floor((q+1)/4),64) + ((q+1) mod 4)]; q is zero-based within window",
                "scope": "Exact support cardinality for input policy, independent of top-k identities; not measured speed, learned routing or quality"},
            "word_budget_plan": budget_plan,
            "self_tests": tests, "source_summaries": source_summaries,
            "scientific_model_training_runs": 0, "scientific_optimizer_updates": 0,
            "dev_or_test_read": False, "gpu_hours": 0,
            "artifacts": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in artifacts],
        }
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        samples = list(iter_windows(target, [0, len(array) // 2, len(array) - 1]))
        for item in samples:
            assert item["input_tokens"] == len(item["input_ids"])
            assert item["loss_tokens"] == max(0, item["input_tokens"] - 1)
            assert item["single_segment"] and item["reset_model_state_before"]
            assert item["word_exposures"] == int(array[item["window_index"], 8])
        result["self_tests"].append("readonly_iterator_first_middle_last_windows_match_index_and_budget")
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: result[k] for k in ["status", "total_windows", "word_exposures_per_pass",
                          "input_tokens_per_pass", "next_token_loss_positions_per_pass",
                          "qsa64_block4_support_cardinality", "word_budget_plan"]}), flush=True)
    except BaseException as exc:
        (OUT / f"failure-{time.time_ns()}.json").write_text(json.dumps(
            {"utc": utc(), "type": type(exc).__name__, "error": str(exc)}, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    build()

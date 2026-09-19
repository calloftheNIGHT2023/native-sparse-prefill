"""Train a train-only tokenizer and lossless record ledger, not packed LM data.

Physical lines are records, not asserted document boundaries. No model weights,
validation/test inputs, optimizer updates or GPU work are used by this script.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "4")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/babylm-2026-strict-small-raw-v0"
OUT = ROOT / "data/babylm-2026-tokenizer-16k-v0"
SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def records(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from f


def batches(path, size=2048):
    batch = []
    for line in records(path):
        batch.append(line)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def visible_edges(length, selected_blocks=64, block_size=4):
    # A block ending at this query is complete and may itself be dropped by top-k.
    # Only the incomplete tail (0..block_size-1 tokens) is always retained.
    return sum(min((t + 1) // block_size, selected_blocks) * block_size
               + (t + 1) % block_size for t in range(length))


def main():
    started = utc()
    begin = time.perf_counter()
    source = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    entries = sorted(source["files"], key=lambda e: e["path"])
    assert len(entries) == 6 and source["total_words_whitespace"] == 10_000_000
    for e in entries:
        p = RAW / e["path"]
        assert p.parent == RAW and p.name.endswith(".train.txt")
        assert sha(p) == e["sha256"], f"Source hash mismatch: {p.name}"
    OUT.mkdir(exist_ok=True)
    complete = OUT / "manifest.json"
    if complete.exists():
        old = json.loads(complete.read_text(encoding="utf-8"))
        assert old["source_manifest_sha256"] == sha(RAW / "manifest.json")
        assert old["status"] == "tokenizer_and_unpacked_record_ledger_verified"
        for artifact in old["artifacts"]:
            assert sha(OUT / artifact["path"]) == artifact["sha256"]
        print(json.dumps({"status": "already_verified", "manifest": str(complete)}))
        return
    lock = OUT / "build.lock"
    with lock.open("x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "utc": started}, f)
    try:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=16_384, min_frequency=2, show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=SPECIALS,
        )
        tokenizer.train_from_iterator(
            (line for e in entries for line in records(RAW / e["path"])), trainer=trainer,
        )
        tokenizer.save(str(OUT / "tokenizer.json"))
        vocab = tokenizer.get_vocab()
        assert tokenizer.get_vocab_size() == 16_384
        assert [vocab[t] for t in SPECIALS] == [0, 1, 2]
        reloaded = Tokenizer.from_file(str(OUT / "tokenizer.json"))
        checks = ["", " a  b\tC\n", "café 中文 🙂\r\n", "can't wouldn't—123.4",
                  "literal <|eos|> text"]
        for text in checks:
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            assert reloaded.encode(text, add_special_tokens=False).ids == ids
            assert tokenizer.decode(ids, skip_special_tokens=False) == text
        print(json.dumps({"event": "tokenizer_saved", "vocab_size": len(vocab),
                          "elapsed_seconds": time.perf_counter()-begin}), flush=True)
        rows = []
        hist = Counter()
        artifact_paths = [OUT / "tokenizer.json"]
        for entry in entries:
            path = RAW / entry["path"]
            stem = entry["path"].split(".")[0]
            token_path = OUT / f"{stem}.ids.u32"
            offsets, words, lengths = [0], [], []
            roundtrips, embedded_specials = 0, 0
            with token_path.open("wb") as f:
                for batch in batches(path):
                    encoded = tokenizer.encode_batch(batch, add_special_tokens=False)
                    for text, enc in zip(batch, encoded):
                        # All records round-trip, not just a favorable sample.
                        assert tokenizer.decode(enc.ids, skip_special_tokens=False) == text
                        roundtrips += 1
                        embedded_specials += sum(t in text for t in SPECIALS)
                        arr = np.asarray(enc.ids, dtype="<u4")
                        f.write(arr.tobytes())
                        offsets.append(offsets[-1] + len(enc.ids))
                        words.append(len(text.split()))
                        lengths.append(len(enc.ids))
            assert sum(words) == entry["word_count_whitespace"]
            assert embedded_specials == 0, "Literal special token in corpus requires explicit policy"
            # The binary stream is lossless serialization only, not a training sample stream.
            # Adjacent records must not acquire cross-record attention implicitly.
            offset_path, word_path = OUT / f"{stem}.offsets.npy", OUT / f"{stem}.words.npy"
            np.save(offset_path, np.asarray(offsets, dtype=np.uint64))
            np.save(word_path, np.asarray(words, dtype=np.uint32))
            assert token_path.stat().st_size == offsets[-1] * 4
            loaded_offsets = np.load(offset_path)
            loaded_words = np.load(word_path)
            assert np.array_equal(np.diff(loaded_offsets), lengths)
            assert int(loaded_words.sum()) == entry["word_count_whitespace"]
            hist.update(lengths)
            lens = np.asarray(lengths)
            row = {"source": path.name, "source_sha256": entry["sha256"],
                   "records": len(lengths), "whitespace_words": sum(words),
                   "text_tokens": offsets[-1], "zero_word_records": sum(w == 0 for w in words),
                   "record_token_p50": float(np.quantile(lens, .5)),
                   "record_token_p95": float(np.quantile(lens, .95)),
                   "record_token_p99": float(np.quantile(lens, .99)),
                   "record_token_max": int(lens.max()),
                   "records_over_2048_tokens": sum(n > 2048 for n in lengths),
                   "records_over_256_tokens": sum(n > 256 for n in lengths),
                   "roundtrip_checks": roundtrips,
                   "literal_special_occurrences": embedded_specials,
                   "ids_file": token_path.name, "offsets_file": offset_path.name,
                   "words_file": word_path.name}
            rows.append(row)
            artifact_paths.extend([token_path, offset_path, word_path])
            print(json.dumps({"event": "source_encoded", **row}), flush=True)
        assert sum(r["whitespace_words"] for r in rows) == 10_000_000
        dense_pairs = sum(n * (n+1) // 2 * count for n, count in hist.items())
        sparse_pairs = sum(visible_edges(n) * count for n, count in hist.items())
        hist_path = OUT / "record-token-length-histogram.json"
        hist_path.write_text(json.dumps(dict(sorted(hist.items())), indent=2)+'\n', encoding="utf-8")
        artifact_paths.append(hist_path)
        result = {"status": "tokenizer_and_unpacked_record_ledger_verified",
                  "started_utc": started, "completed_utc": utc(),
                  "elapsed_seconds": time.perf_counter()-begin,
                  "script_sha256": sha(Path(__file__)),
                  "source_dataset": source["dataset"], "source_revision": source["revision"],
                  "source_manifest_sha256": sha(RAW / "manifest.json"),
                  "tokenizer": {"type": "byte-level BPE", "vocab_size": len(vocab),
                                "normalizer": None, "add_prefix_space": False,
                                "min_frequency": 2, "special_tokens": dict(zip(SPECIALS, [0,1,2])),
                                "post_processor": None, "automatic_special_tokens": False,
                                "external_tokenizer_or_weights": False,
                                "training_sources": [e["path"] for e in entries]},
                  "runtime": {p: importlib.metadata.version(p) for p in ["tokenizers", "numpy"]},
                  "python": sys.version, "files": rows,
                  "total_source_words": 10_000_000,
                  "total_text_tokens": sum(r["text_tokens"] for r in rows),
                  "total_physical_line_records": sum(r["records"] for r in rows),
                  "tokens_include_original_whitespace": True,
                  "inserted_bos_eos_tokens": 0, "truncated_tokens": 0,
                  "packed_training_dataset_ready": False, "document_boundaries_verified": False,
                  "dev_or_test_read": False, "model_training_runs": 0,
                  "scientific_optimizer_updates": 0, "gpu_hours": 0,
                  "isolated_record_topk64_block4_diagnostic": {
                      "dense_causal_pairs": dense_pairs, "retained_pairs": sparse_pairs,
                      "retained_pair_fraction": sparse_pairs/dense_pairs,
                      "scope": "Exact combinatorial mask count assuming every physical line is isolated; not verified document policy, not runtime or quality evidence. All blocks ending <= query are top-k eligible; only the incomplete 0..3-token tail is mandatory. No BOS/EOS added."},
                  "artifacts": [{"path": x.name, "bytes": x.stat().st_size, "sha256": sha(x)}
                                for x in artifact_paths]}
        complete.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding="utf-8")
        print(json.dumps({k: result[k] for k in ["status", "total_source_words", "total_text_tokens",
                                               "total_physical_line_records", "elapsed_seconds",
                                               "isolated_record_topk64_block4_diagnostic"]}), flush=True)
    except BaseException as exc:
        (OUT / f"failure-{time.time_ns()}.json").write_text(
            json.dumps({"utc": utc(), "type": type(exc).__name__, "error": str(exc)}, indent=2)+'\n',
            encoding="utf-8")
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

"""Independently reconstruct pinned source bytes from the persisted token ledger."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/babylm-2026-tokenizer-16k-v0"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    begin = time.perf_counter()
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "tokenizer_and_unpacked_record_ledger_verified"
    for a in manifest["artifacts"]:
        assert sha(OUT / a["path"]) == a["sha256"]
    tokenizer = Tokenizer.from_file(str(OUT / "tokenizer.json"))
    rows = []
    for r in manifest["files"]:
        ids = np.memmap(OUT / r["ids_file"], dtype="<u4", mode="r")
        offsets = np.load(OUT / r["offsets_file"])
        words = np.load(OUT / r["words_file"])
        assert len(offsets) == len(words)+1 == r["records"]+1
        assert int(offsets[0]) == 0 and int(offsets[-1]) == len(ids)
        assert np.all(offsets[1:] >= offsets[:-1])
        assert int(ids.max()) < manifest["tokenizer"]["vocab_size"]
        h = hashlib.sha256()
        decoded_word_count = 0
        for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
            text = tokenizer.decode(ids[int(a):int(b)].tolist(), skip_special_tokens=False)
            h.update(text.encode("utf-8"))
            n = len(text.split())
            assert n == int(words[i])
            decoded_word_count += n
        assert h.hexdigest() == r["source_sha256"]
        assert decoded_word_count == r["whitespace_words"]
        rows.append({"source": r["source"], "decoded_source_sha256": h.hexdigest(),
                     "records_checked": len(words), "decoded_words": decoded_word_count,
                     "persisted_text_tokens": len(ids), "pass": True})
    # Independent explicit set enumeration for the mask-count formula used in preparation.
    pairs = ROOT / "scripts/prepare_babylm_tokenizer_v0.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("prep", pairs)
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    mask_cases = 0
    for n in range(1, 65):
        for block in (1, 2, 4):
            for k in (0, 1, 2, 64):
                count = 0
                for t in range(n):
                    complete_blocks = list(range((t + 1) // block))
                    chosen = complete_blocks[:k]
                    visible = {j for b in chosen for j in range(b*block, (b+1)*block)}
                    visible.update(range(((t+1)//block)*block, t+1))
                    assert all(j <= t for j in visible)
                    count += len(visible)
                assert count == prep.visible_edges(n, k, block)
                mask_cases += 1
    result = {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.perf_counter()-begin,
              "ledger_manifest_sha256": sha(OUT / "manifest.json"),
              "script_sha256": sha(Path(__file__)), "files": rows,
              "all_saved_source_bytes_reconstructed": True,
              "total_records_checked": sum(r["records_checked"] for r in rows),
              "total_decoded_words": sum(r["decoded_words"] for r in rows),
              "mask_formula_enumeration_cases": mask_cases,
              "scientific_training_updates": 0, "gpu_work": False,
              "scope": "Verifies tokenizer serialization and corpus accounting only. Does not verify document continuity, split independence, model learning, or training speed."}
    result_path = ROOT / "results/babylm-tokenizer-ledger-validation-v0.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

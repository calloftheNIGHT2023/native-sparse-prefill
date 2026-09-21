"""Pin official BabyLM dev and encode with the already frozen train tokenizer.

Never trains a tokenizer/model, opens the final test split, or changes training
streams. Line-level exact overlap and explicit marker overlap are descriptive
audits only: no removal, source reconstruction, or model-based subset selection.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('RAYON_NUM_THREADS', '4')
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
REVISION = '169f42e32d0aaf65ec6b91d55bafad27a3afc729'
DATASET = 'BabyLM-community/BabyLM-dev'
RAW = ROOT / 'data/babylm-dev-raw-v0'
OUT = ROOT / 'data/babylm-dev-token-ledger-v0'
PROVENANCE = ROOT / 'literature/babylm-dev-preparation-2026-09-17'
TRAIN_RAW = ROOT / 'data/babylm-2026-strict-small-raw-v0'
TOKENIZER_DIR = ROOT / 'data/babylm-2026-tokenizer-16k-v0'
TOKENIZER_SHA = '230b9d6993dcaf32f40cec2c44ac79d7d713213616ba2d8e45ad4a96e5a9dbe6'
EXPECTED_NAMES = {f'{stem}.dev' for stem in ('bnc_spoken', 'childes', 'gutenberg', 'open_subtitles', 'simple_wiki', 'switchboard')}
MARKER = re.compile(r'^=\s*=\s*=.*=\s*=\s*=$')
SPECIALS = ('<|pad|>', '<|bos|>', '<|eos|>')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def lines(path):
    # Same physical-line serialization and original newline preservation as train.
    with path.open('r', encoding='utf-8', newline='') as stream:
        yield from stream


def download(entry):
    name = entry['path']
    assert name in EXPECTED_NAMES and Path(name).name == name
    destination = RAW / name
    url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}'
    started = utc()
    if destination.exists():
        blob = destination.read_bytes()
        cached = True
    else:
        with urllib.request.urlopen(url, timeout=90) as response:
            blob = response.read(int(entry['size']) + 1)
        cached = False
    assert len(blob) == entry['size'], f'Byte size mismatch: {name}'
    digest = hashlib.sha256(blob).hexdigest()
    if 'lfs' in entry:
        assert digest == entry['lfs']['oid'], f'LFS SHA256 mismatch: {name}'
        validation = 'LFS SHA256'
    else:
        git_digest = hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest()
        assert git_digest == entry['oid'], f'Git blob SHA1 mismatch: {name}'
        validation = 'Git blob SHA1 plus local SHA256'
    blob.decode('utf-8', errors='strict')
    if not cached:
        part = destination.with_suffix(destination.suffix + '.partial')
        part.write_bytes(blob)
        part.replace(destination)
    return {'path': name, 'url': url, 'bytes': len(blob), 'sha256': digest,
            'official_git_oid': entry['oid'], 'official_lfs_oid': entry.get('lfs', {}).get('oid'),
            'validation': validation, 'download_started_utc': started,
            'verified_utc': utc(), 'used_verified_existing_file': cached}


def overlap_summary(dev, train):
    common = dev.keys() & train.keys()
    long_common = {text for text in common if len(text.split()) >= 20}
    return {'dev_nonempty_line_occurrences': sum(dev.values()),
            'dev_unique_nonempty_lines': len(dev),
            'exact_shared_unique_lines': len(common),
            'dev_occurrences_matching_train': sum(dev[text] for text in common),
            'train_occurrences_matching_dev': sum(train[text] for text in common),
            'dev_line_occurrences_ge20_words': sum(count for text, count in dev.items() if len(text.split()) >= 20),
            'exact_shared_unique_lines_ge20_words': len(long_common),
            'dev_occurrences_matching_train_ge20_words': sum(dev[text] for text in long_common),
            'train_occurrences_matching_dev_ge20_words': sum(train[text] for text in long_common),
            'dev_word_occurrences_in_exact_matching_ge20_lines': sum(dev[text] * len(text.split()) for text in long_common)}


def corpus_counter(path):
    return Counter(text for line in lines(path) if (text := line.rstrip('\r\n')).strip())


def duplicate_audit(train_manifest, source_rows):
    all_train = Counter()
    train_by_source = {}
    train_markers = {}
    for entry in train_manifest['files']:
        path = TRAIN_RAW / entry['path']
        assert sha(path) == entry['sha256']
        source = entry['path'].split('.')[0]
        counter = corpus_counter(path)
        train_by_source[source] = counter
        all_train.update(counter)
        train_markers[source] = Counter({text.strip(): count for text, count in counter.items() if MARKER.fullmatch(text.strip())})
    all_dev = Counter()
    per_source, marker_rows = [], []
    for entry in source_rows:
        source = entry['path'].split('.')[0]
        dev = corpus_counter(RAW / entry['path'])
        all_dev.update(dev)
        per_source.append({'source': source, 'against_matching_train_source': overlap_summary(dev, train_by_source[source]),
                           'against_all_train_sources': overlap_summary(dev, all_train)})
        markers = Counter({text.strip(): count for text, count in dev.items() if MARKER.fullmatch(text.strip())})
        shared = sorted(markers.keys() & train_markers[source].keys())
        marker_rows.append({'source': source, 'train_marker_occurrences': sum(train_markers[source].values()),
                            'dev_marker_occurrences': sum(markers.values()),
                            'train_unique_marker_labels': len(train_markers[source]),
                            'dev_unique_marker_labels': len(markers),
                            'shared_exact_marker_labels': shared,
                            'shared_unique_marker_count': len(shared),
                            'dev_marker_occurrences_matching_train': sum(markers[label] for label in shared)})
    result = {'created_utc': utc(), 'train_revision': train_manifest['revision'], 'dev_revision': REVISION,
              'exact_line_definition': 'Remove trailing CR/LF only; preserve case, Unicode and all other whitespace; exclude whitespace-only lines.',
              'deduplicated': False, 'model_based_selection': False, 'final_test_read': False,
              'global': overlap_summary(all_dev, all_train), 'per_source': per_source,
              'explicit_marker_overlap': marker_rows,
              'interpretation': 'Exact short phrases are not automatically leakage. >=20-word matches are separately reported. Shared source identifiers are not proof of adjacent passages or full-document overlap. No exact/near/document deduplication has been performed.'}
    write_json(OUT / 'train-dev-overlap.json', result)
    return result


def main():
    started, clock_start = utc(), time.perf_counter()
    metadata_path = ROOT / 'literature/babylm-boundary-audit-2026-09-17/dev-metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    assert metadata['id'] == DATASET and metadata['sha'] == REVISION
    tokenizer_path = TOKENIZER_DIR / 'tokenizer.json'
    assert sha(tokenizer_path) == TOKENIZER_SHA
    tokenizer_manifest_path = TOKENIZER_DIR / 'manifest.json'
    tokenizer_manifest = json.loads(tokenizer_manifest_path.read_text(encoding='utf-8'))
    assert tokenizer_manifest['status'] == 'tokenizer_and_unpacked_record_ledger_verified'
    train_manifest_path = TRAIN_RAW / 'manifest.json'
    train_manifest = json.loads(train_manifest_path.read_text(encoding='utf-8'))
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    tree_path = PROVENANCE / 'dev-tree.json'
    tree_url = f'https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}?recursive=true'
    if tree_path.exists():
        tree = json.loads(tree_path.read_text(encoding='utf-8'))
    else:
        with urllib.request.urlopen(tree_url, timeout=60) as response:
            tree = json.load(response)
        write_json(tree_path, tree)
    entries = sorted((entry for entry in tree if entry['type'] == 'file' and entry['path'].endswith('.dev')), key=lambda entry: entry['path'])
    assert {entry['path'] for entry in entries} == EXPECTED_NAMES
    total_bytes = sum(entry['size'] for entry in entries)
    provenance = {'created_utc': utc(), 'metadata_input': str(metadata_path.relative_to(ROOT)),
                  'metadata_sha256': sha(metadata_path), 'tree_url': tree_url,
                  'tree_sha256': sha(tree_path), 'revision': REVISION, 'total_dev_bytes': total_bytes,
                  'maximum_download_bytes': 100_000_000, 'script_sha256': sha(Path(__file__))}
    write_json(PROVENANCE / 'source-manifest.json', provenance)
    if total_bytes > 100_000_000:
        print(json.dumps({'status': 'metadata_only_download_cap_exceeded', **provenance}))
        return
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    complete = OUT / 'manifest.json'
    if complete.exists():
        previous = json.loads(complete.read_text(encoding='utf-8'))
        assert previous['status'] == 'frozen_tokenizer_dev_record_ledger_verified'
        assert previous['dev_revision'] == REVISION and previous['tokenizer_sha256'] == TOKENIZER_SHA
        assert previous['train_source_manifest_sha256'] == sha(train_manifest_path)
        for item in previous['artifacts']:
            assert sha(OUT / item['path']) == item['sha256']
        raw_manifest = json.loads((RAW / 'manifest.json').read_text(encoding='utf-8'))
        for item in raw_manifest['files']:
            assert sha(RAW / item['path']) == item['sha256']
        print(json.dumps({'status': 'already_verified', 'manifest': str(complete)}))
        return
    lock = OUT / 'build.lock'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump({'started_utc': started, 'pid': os.getpid()}, stream)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            raw_rows = list(pool.map(download, entries))
        raw_manifest = {'status': 'official_dev_raw_verified', 'created_utc': utc(), 'dataset': DATASET,
                        'revision': REVISION, 'total_bytes': total_bytes, 'files': raw_rows,
                        'official_tree_sha256': sha(tree_path), 'final_test_read': False}
        write_json(RAW / 'manifest.json', raw_manifest)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        assert tokenizer.get_vocab_size() == 16384
        rows, artifacts = [], []
        for entry in raw_rows:
            source = entry['path'].split('.')[0]
            token_path = OUT / f'{source}.ids.u32'
            offset_path, word_path = OUT / f'{source}.offsets.npy', OUT / f'{source}.words.npy'
            offsets, words, lengths = [0], [], []
            roundtrips = special_occurrences = 0
            def encode(batch, stream):
                nonlocal roundtrips, special_occurrences
                for text, encoding in zip(batch, tokenizer.encode_batch(batch, add_special_tokens=False)):
                    assert tokenizer.decode(encoding.ids, skip_special_tokens=False) == text
                    roundtrips += 1
                    special_occurrences += sum(token in text for token in SPECIALS)
                    encoded = np.asarray(encoding.ids, dtype='<u4')
                    assert len(encoded) == 0 or int(encoded.max()) < 16384
                    stream.write(encoded.tobytes())
                    offsets.append(offsets[-1] + len(encoded))
                    words.append(len(text.split()))
                    lengths.append(len(encoded))
            with token_path.open('wb') as stream:
                batch = []
                for text in lines(RAW / entry['path']):
                    batch.append(text)
                    if len(batch) == 2048:
                        encode(batch, stream); batch = []
                if batch:
                    encode(batch, stream)
            assert special_occurrences == 0, 'Literal special token requires an explicit frozen policy'
            np.save(offset_path, np.asarray(offsets, dtype=np.uint64))
            np.save(word_path, np.asarray(words, dtype=np.uint32))
            assert token_path.stat().st_size == offsets[-1] * 4
            assert np.array_equal(np.diff(np.load(offset_path)), lengths)
            assert int(np.load(word_path).sum()) == sum(words)
            row = {'source': entry['path'], 'source_sha256': entry['sha256'], 'records': len(lengths),
                   'whitespace_words': sum(words), 'text_tokens': offsets[-1],
                   'zero_word_records': sum(value == 0 for value in words), 'roundtrip_checks': roundtrips,
                   'record_token_p50': float(np.quantile(lengths, .5)),
                   'record_token_p95': float(np.quantile(lengths, .95)),
                   'record_token_max': max(lengths), 'literal_special_occurrences': special_occurrences,
                   'ids_file': token_path.name, 'offsets_file': offset_path.name, 'words_file': word_path.name}
            rows.append(row)
            artifacts.extend((token_path, offset_path, word_path))
            print(json.dumps({'event': 'dev_source_encoded', **row}), flush=True)
        duplicate = duplicate_audit(train_manifest, raw_rows)
        artifacts.append(OUT / 'train-dev-overlap.json')
        assert sha(tokenizer_path) == TOKENIZER_SHA
        result = {'status': 'frozen_tokenizer_dev_record_ledger_verified', 'started_utc': started,
                  'completed_utc': utc(), 'elapsed_seconds': time.perf_counter() - clock_start,
                  'script_sha256': sha(Path(__file__)), 'dev_dataset': DATASET, 'dev_revision': REVISION,
                  'raw_manifest_sha256': sha(RAW / 'manifest.json'),
                  'train_source_manifest_sha256': sha(train_manifest_path),
                  'tokenizer_path_relative_to_project': str(tokenizer_path.relative_to(ROOT)).replace('\\', '/'),
                  'tokenizer_sha256': TOKENIZER_SHA, 'tokenizer_manifest_sha256': sha(tokenizer_manifest_path),
                  'tokenizer_retrained': False, 'vocab_size': 16384, 'files': rows,
                  'total_source_words': sum(item['whitespace_words'] for item in rows),
                  'total_text_tokens': sum(item['text_tokens'] for item in rows),
                  'total_physical_line_records': sum(item['records'] for item in rows),
                  'tokens_include_original_whitespace': True, 'inserted_bos_eos_tokens': 0,
                  'truncated_tokens': 0, 'deduplicated': False, 'final_test_read': False,
                  'training_stream_modified': False, 'model_forward_calls': 0,
                  'scientific_optimizer_updates': 0, 'gpu_hours': 0,
                  'dev_subset_selected': False, 'document_boundaries_verified': False,
                  'loss_ready_window_index_created': False,
                  'overlap_summary': duplicate['global'],
                  'runtime': {name: importlib.metadata.version(name) for name in ('tokenizers', 'numpy')},
                  'artifacts': [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in artifacts],
                  'scope': 'Frozen raw official dev and lossless line ledger only. Matching training window/reset/loss policy must be separately applied before NLL. Shared source metadata and exact phrases are disclosed; no independence or deduplication claim.'}
        write_json(complete, result)
        print(json.dumps({'event': 'dev_preparation_complete', 'manifest': str(complete),
                          'total_words': result['total_source_words'], 'total_tokens': result['total_text_tokens'],
                          'overlap': duplicate['global']}), flush=True)
    except Exception as error:
        failure = {'utc': utc(), 'type': type(error).__name__, 'message': str(error),
                   'scientific_optimizer_updates': 0, 'final_test_read': False}
        with (OUT / 'failures.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(failure) + '\n')
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

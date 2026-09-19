"""Create a new, connection-redacted research snapshot. No network/model calls.

Original experiment files are never modified. Run from the local research root.
The output directory must not exist. See EXPORT_MANIFEST.json for byte hashes.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path


TEXT_SUFFIXES = {'.py', '.json', '.jsonl', '.md', '.txt', '.ps1', '.sh',
                 '.yaml', '.yml', '.toml', '.patch', '.diff', '.csv', '.tsv'}
SECRET = re.compile(r'(?<![A-Za-z0-9_-])(?:gh[pousr]_[A-Za-z0-9_]{20,}|'
                    r'github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|'
                    r'sk-[A-Za-z0-9_-]{24,}|rpa_[A-Za-z0-9_-]{20,})|'
                    r'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----')
IP = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    replacements = {}
    # Inspect metadata only, never key files or credential stores.
    connection_files = list((root / 'logs').glob('cloud-connection*.json'))
    connection_files += list((root / 'configs').glob('*.json'))
    def collect(value):
        if isinstance(value, dict):
            for key, val in value.items():
                if isinstance(val, str) and key in {'pod_id', 'ssh_host', 'ssh_key_path', 'relay'}:
                    replacements[val] = 'REDACTED_' + key.upper()
                elif isinstance(val, (dict, list)):
                    collect(val)
        elif isinstance(value, list):
            for val in value:
                collect(val)
    for file in connection_files:
        try:
            collect(json.loads(file.read_text(encoding='utf-8')))
        except (ValueError, UnicodeError):
            pass
    replacements = dict(sorted(replacements.items(), key=lambda x: -len(x[0])))
    files, omitted = [], []

    def redact(text):
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = IP.sub('REDACTED_IPV4', text)
        text = re.sub(r'[A-Za-z0-9_-]+@ssh\.runpod\.io', 'REDACTED_SSH_RELAY', text)
        text = re.sub(r'(?i)([A-Z]:[\\/]+Users[\\/]+)[^\\/\s"\']+', r'\1USER', text)
        return text

    def emit(rel, original, source=None, compress=False, preserve=False):
        text = original.decode('utf-8-sig')
        if SECRET.search(text):
            raise ValueError('Credential signature detected in ' + str(source or rel))
        clean = text if preserve else redact(text)
        # Preserve bytes when unchanged, including original CRLF line endings.
        plain = original if clean == text else clean.encode('utf-8')
        payload = gzip.compress(plain, compresslevel=9, mtime=0) if compress else plain
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        files.append({'path': str(rel).replace('\\', '/'),
                      'source': source, 'source_bytes': len(original),
                      'source_sha256': sha(original), 'export_sha256': sha(payload),
                      'decompressed_sha256': sha(plain) if compress else None,
                      'export_bytes': len(payload), 'redacted': plain != original})

    def copy(file, rel=None, preserve=False):
        rel = rel or file.relative_to(root)
        emit(rel, file.read_bytes(), str(file.relative_to(root)).replace('\\', '/'), preserve=preserve)

    for directory in ['src', 'scripts', 'configs', 'tests', 'docs', 'experiments', 'patches', 'third_party']:
        for file in sorted((root / directory).rglob('*')):
            if not file.is_file() or any(p in {'.git', '__pycache__', '.pytest_cache'} for p in file.parts):
                continue
            if file.suffix not in TEXT_SUFFIXES and not file.name.lower().startswith(('license', 'copying')):
                continue
            text = file.read_text(encoding='utf-8-sig', errors='strict')
            if directory == 'scripts' and file.name != Path(__file__).name and ('ssh.runpod.io' in text or 'runpod_verifier_ed25519' in text or IP.search(text)):
                omitted.append({'path': str(file.relative_to(root)).replace('\\', '/'),
                                'reason': 'Local cloud operator helper with embedded connection settings'})
                continue
            copy(file, preserve=directory in {'src', 'tests', 'third_party'})

    for name in ['README.md', 'STATE.md', 'TIMELINE.md']:
        copy(root / name)
    for file in sorted(root.glob('requirements*.txt')):
        copy(file)
    pinned = root / 'literature/babylm-causal-adapter-2026-09-17'
    for file in sorted(pinned.rglob('*')):
        if file.is_file() and file.suffix in TEXT_SUFFIXES:
            copy(file, preserve=True)

    # Only source manifests and the synthetic scoring fixtures, never corpora.
    for folder in ['babylm-2026-strict-small-raw-v0', 'babylm-2026-tokenizer-16k-v0',
                   'babylm-2026-windows-v0', 'babylm-dev-raw-v0',
                   'babylm-dev-token-ledger-v0', 'babylm-dev-windows-v0']:
        file = root / 'data' / folder / 'manifest.json'
        if file.exists():
            copy(file, Path('evidence/data-manifests') / folder / 'manifest.json')
    for file in sorted((root / 'data/babylm-task-engineering-fixtures-v0').glob('*')):
        if file.is_file() and file.suffix in {'.json', '.jsonl'}:
            copy(file)

    source = root / 'logs/manual-status-20260919T165216Z'
    curves, endpoints = [], {}
    for mode in ['dense', 'sparse']:
        run = source / mode
        raw = (run / 'events.jsonl').read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        summary = json.loads((run / 'summary.json').read_bytes())
        emit(Path('evidence/babylm-first-pair') / mode / 'events.redacted.jsonl.gz', raw,
             str((run / 'events.jsonl').relative_to(root)).replace('\\', '/'), compress=True)
        for name in ['summary.json', 'protocol.json', 'stage.json']:
            copy(run / name, Path('evidence/babylm-first-pair') / mode / name)
        evals = [r for r in rows if r['type'] == 'evaluation']
        for event in evals:
            c, total = event['counts'], event['metrics']['total']
            curves.append({'mode': mode, 'utc': event['utc'], 'updates': c['updates'],
                           'word_exposures': c['word_exposures'], 'input_tokens': c['input_tokens'],
                           'panel_nll': total['nll'], 'panel_loss_tokens': total['loss_tokens'],
                           'panel_windows': total['windows']})
        endpoints[mode] = {'completed_utc': summary['completed_utc'], 'counts': summary['counts'],
                           'eval_counts': summary['eval_counts'],
                           'run_wall_hours': summary['elapsed_wall_seconds'] / 3600,
                           'final_panel_nll': evals[-1]['metrics']['total']['nll'],
                           'checkpoint_sha256': summary['checkpoint_sha256'],
                           'final_model_sha256': summary['snapshot_state']['final_receipts'][-1]['model_sha256']}
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
    writer.writeheader(); writer.writerows(curves)
    emit('evidence/babylm-first-pair/panel-learning-curves.csv', stream.getvalue().encode())
    emit('evidence/babylm-first-pair/endpoints.json', (json.dumps(endpoints, indent=2) + '\n').encode())
    copy(source / 'completion-audit.json', Path('evidence/babylm-first-pair/completion-audit.redacted.json'))

    ignore = '''# Runtime data, credentials, environments and model weights
.venv*/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.env
.env.*
*.pem
*.key
*.pt
*.pth
*.safetensors
*.npy
*.npz
*.u32
logs/
results/
exports/
provenance/
data/*
!data/babylm-task-engineering-fixtures-v0/
'''
    emit('.gitignore', ignore.encode())
    emit('.gitattributes', b'* -text\n# Preserve scientific-source bytes and archived evidence hashes.\n')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Private GitHub research snapshot; no credentials, raw corpus or model weights.',
                'redaction_notice': 'Connection metadata removed in exported copies only. Embedded historical hashes refer to original local files. Scientific source bytes preserved; redacted protocols are archival evidence, not directly deployable cloud authorization.',
                'files': files, 'omitted_operator_helpers': omitted,
                'excluded_roots': ['.venv*', 'logs', 'results', 'exports', 'provenance', 'raw/tokenized datasets'],
                'selected_evidence_added_separately': True}
    (out / 'EXPORT_MANIFEST.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode())
    print(json.dumps({'output': str(out), 'files': len(files), 'bytes': sum(f['export_bytes'] for f in files),
                      'redacted_files': sum(f['redacted'] for f in files), 'omitted_helpers': len(omitted)}))


if __name__ == '__main__':
    main()

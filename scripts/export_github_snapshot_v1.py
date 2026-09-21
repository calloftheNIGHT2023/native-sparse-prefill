"""Create a fresh public-safe code/evidence snapshot, with historical sources.

No network, git writes, model calls, credentials, corpus text or weights.
Exported protocols are redacted records, not deployment authorizations.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from export_github_snapshot_v0 import SECRET, IP, TEXT_SUFFIXES

TERMINAL = 'logs/babylm-dw-final-text-backup-20260921/20260921T211814.452452Z/manifest.json'
COST_TERMINAL = 'logs/babylm-dw-cost-profile-backup-20260921/20260921T234120.236686Z/manifest.json'
REPORT_DIRS = ('babylm-dw-seed-full-dev-audit-20260921', 'babylm-stage-w-complete-20260921',
               'babylm-dw-seed-training-audit-20260921')
SELF = 'scripts/export_github_snapshot_v1.py'
VERIFY = 'scripts/verify_github_evidence_v1.py'
ARCHIVE = 'archives/first-pair-20260919'
RELAYS = re.compile(r'[A-Za-z0-9_-]+@ssh\.runpod\.io')
KEYPATH = re.compile(r'(?:[A-Za-z]:[\\/]|~[\\/]|/)[^\s\"\'<>]*[\\/]\.ssh[\\/][^\s\"\'<>]+')
SENSITIVE_KEYS = {'pod_id', 'ssh_host', 'ssh_key_path', 'key_path', 'identity_file',
                  'relay', 'ssh_target', 'relay_target', 'ssh_command', 'public_ip'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--include-cost-profile', action='store_true')
    args = parser.parse_args()
    root, out, old = args.source.resolve(), args.output.resolve(), args.historical.resolve()
    if out.exists(): raise ValueError('Fresh export required')
    replacements = {}
    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in SENSITIVE_KEYS and isinstance(item, str) and len(item) >= 5 and not item.startswith('REDACTED'):
                    replacements[item] = 'REDACTED_CONNECTION_METADATA'
                elif isinstance(item, (list, dict)): collect(item)
        elif isinstance(value, list):
            for item in value: collect(item)
    # Metadata only. Never open any referenced private-key or environment file.
    for path in list((root / 'logs').glob('cloud-connection*.json')) + list((root / 'configs').rglob('*.json')):
        try: collect(load(path))
        except (ValueError, UnicodeError): pass
    replacements = dict(sorted(replacements.items(), key=lambda item: -len(item[0])))
    def sensitive(text):
        return bool(SECRET.search(text) or RELAYS.search(text) or KEYPATH.search(text)
                    or any(value in text for value in replacements))
    def redact(text):
        if SECRET.search(text): raise ValueError('Credential signature: export refused')
        for value, replacement in replacements.items(): text = text.replace(value, replacement)
        text = RELAYS.sub('REDACTED_SSH_RELAY', text)
        text = KEYPATH.sub('REDACTED_KEY_PATH', text)
        text = IP.sub('REDACTED_IPV4', text)
        return re.sub(r'(?i)([A-Z]:[\\/]+Users[\\/]+)[^\\/\s\"\']+', r'\1USER', text)
    entries, omitted = [], []
    out.mkdir(parents=True, exist_ok=False)
    def emit(relative, original, source=None, compress=False, preserve=False, expected=None):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe export path')
        if expected is not None and sha(original) != expected: raise ValueError('Original hash mismatch: ' + str(source))
        text = original.decode('utf-8-sig')
        if SECRET.search(text): raise ValueError('Credential signature in ' + str(source or relative))
        if preserve:
            if sensitive(text): raise ValueError('Preserved file contains connection metadata: ' + str(source or relative))
            plain = original
        else:
            clean = redact(text); plain = original if clean == text else clean.encode('utf-8')
        if sensitive(plain.decode('utf-8-sig')): raise ValueError('Redaction incomplete: ' + str(relative))
        payload = gzip.compress(plain, compresslevel=9, mtime=0) if compress else plain
        target = out / relative; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: stream.write(payload)
        entry = {'path': relative.as_posix(), 'source': source, 'source_sha256': sha(original),
                 'source_bytes': len(original), 'export_sha256': sha(payload), 'export_bytes': len(payload),
                 'decompressed_sha256': sha(plain) if compress else None,
                 'redacted': plain != original, 'byte_preserved': plain == original}
        entries.append(entry)
        return entry['path']
    def copy(source, relative=None, **kwargs):
        source = Path(source)
        return emit(relative or source.relative_to(root), source.read_bytes(),
                    source.relative_to(root).as_posix(), **kwargs)

    # Preserve the historical snapshot root, including its own source manifest.
    # Its unchanged v0 verifier resolves its historical files under this root.
    previous = load(old / 'EXPORT_MANIFEST.json')
    for entry in previous['files']:
        source = old / entry['path']; raw = source.read_bytes()
        if sha(raw) != entry['export_sha256']: raise ValueError('Historical file changed: ' + entry['path'])
        plain = gzip.decompress(raw) if entry.get('decompressed_sha256') else raw
        if entry.get('decompressed_sha256') and sha(plain) != entry['decompressed_sha256']: raise ValueError('Historical gzip changed')
        if sensitive(plain.decode('utf-8-sig')): raise ValueError('Historical archive requires explicit redaction review: ' + entry['path'])
        target = out / ARCHIVE / entry['path']; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
        entries.append({'path': (Path(ARCHIVE) / entry['path']).as_posix(),
                        'source': 'historical-snapshot/' + entry['path'], 'source_sha256': sha(raw),
                        'source_bytes': len(raw), 'export_sha256': sha(raw), 'export_bytes': len(raw),
                        'decompressed_sha256': entry.get('decompressed_sha256'), 'redacted': False, 'byte_preserved': True})
    emit(Path(ARCHIVE) / 'EXPORT_MANIFEST.json', (old / 'EXPORT_MANIFEST.json').read_bytes(),
         'historical-snapshot/EXPORT_MANIFEST.json', preserve=True)

    current_master = load(root / 'configs/babylm-dw-seed-confirmation-20260921-v1/master.json')
    required_scientific = current_master['source_sha256']
    for name, digest in required_scientific.items():
        if sha((root / name).read_bytes()) != digest: raise ValueError('Current scientific source changed: ' + name)
    for directory in ('src', 'scripts', 'configs', 'tests', 'docs', 'experiments', 'patches', 'third_party'):
        for path in sorted((root / directory).rglob('*')):
            if not path.is_file() or any(x in {'.git', '__pycache__', '.pytest_cache'} for x in path.parts): continue
            if path.suffix not in TEXT_SUFFIXES and not path.name.lower().startswith(('license', 'copying')): continue
            rel = path.relative_to(root).as_posix(); text = path.read_text(encoding='utf-8-sig')
            if SECRET.search(text): raise ValueError('Credential signature in ' + rel)
            preserve = directory in {'src', 'scripts', 'tests', 'third_party', 'patches'}
            operator = (sensitive(text) or RELAYS.search(text) or KEYPATH.search(text)
                        or ('ssh.runpod.io' in text or 'runpod_verifier_ed25519' in text)
                        or bool(IP.search(text)))
            if directory == 'scripts' and rel not in {SELF, VERIFY, 'scripts/export_github_snapshot_v0.py'} and operator:
                if rel in required_scientific: raise ValueError('Required scientific source cannot be omitted: ' + rel)
                omitted.append({'path': rel, 'reason': 'Operator helper contains concrete connection metadata or connection setup strings'})
                continue
            if preserve and sensitive(text): raise ValueError('Cannot redact scientific/code bytes: ' + rel)
            copy(path, preserve=preserve)
    for name in ('STATE.md', 'TIMELINE.md'):
        copy(root / name)
    for path in root.glob('requirements*.txt'): copy(path, preserve=True)
    pinned = root / 'literature/babylm-causal-adapter-2026-09-17'
    for path in sorted(pinned.rglob('*')):
        if path.is_file() and path.suffix in TEXT_SUFFIXES: copy(path, preserve=True)
    for folder in ('babylm-2026-strict-small-raw-v0', 'babylm-2026-tokenizer-16k-v0', 'babylm-2026-windows-v0',
                   'babylm-dev-raw-v0', 'babylm-dev-token-ledger-v0', 'babylm-dev-windows-v0'):
        source = root / 'data' / folder / 'manifest.json'
        if source.exists(): copy(source, Path('evidence/data-manifests') / folder / 'manifest.json')
    for source in (root / 'data/babylm-task-engineering-fixtures-v0').glob('*'):
        if source.is_file() and source.suffix in {'.json', '.jsonl'}: copy(source)

    index = {'schema_version': 1, 'historical_root': ARCHIVE, 'terminal_manifest_source': TERMINAL,
             'terminal_manifest_sha256': sha((root / TERMINAL).read_bytes()), 'terminal': {}, 'seed1_raw': {}, 'reports': {}}
    terminal = load(root / TERMINAL)
    if terminal['status'] != 'complete_all_terminal_text_sha_verified' or len(terminal['receipts']) != 45: raise ValueError('Terminal allowlist differs')
    copy(root / TERMINAL, 'evidence/seed-20260921/terminal-manifest.redacted.json')
    for receipt in terminal['receipts']:
        source = root / receipt['local_path']; raw = source.read_bytes()
        if len(raw) != receipt['size_bytes']: raise ValueError('Terminal original size differs')
        relative = Path('evidence/seed-20260921/raw') / receipt['remote_relative_path']
        compress = source.suffix == '.jsonl'
        if compress: relative = Path(relative.as_posix() + '.gz')
        target = emit(relative, raw, receipt['local_path'], compress=compress, expected=receipt['sha256'])
        index['terminal'][receipt['remote_relative_path']] = target
    for folder in REPORT_DIRS:
        for source in sorted((root / 'results' / folder).iterdir()):
            if source.is_file() and source.suffix in {'.json', '.md', '.py'}:
                target = copy(source, Path('evidence/reports') / folder / source.name)
                index['reports'][folder + '/' + source.name] = target
    cost_summary = None
    if args.include_cost_profile:
        cost_manifest = load(root / COST_TERMINAL)
        if len(cost_manifest['files']) != 6 or cost_manifest['result']['status'] != 'complete_read_only_cost_diagnostic':
            raise ValueError('Cost diagnostic terminal evidence incomplete')
        index['cost'] = {}
        index['cost_manifest'] = copy(root / COST_TERMINAL, 'evidence/cost-profile/terminal-manifest.redacted.json')
        for item in cost_manifest['files']:
            source = root / item['local_path']; original = source.read_bytes()
            if len(original) != item['bytes']: raise ValueError('Cost evidence size differs')
            rel = Path('evidence/cost-profile/raw') / item['path']
            compressed = source.suffix == '.jsonl'
            if compressed: rel = Path(rel.as_posix() + '.gz')
            index['cost'][item['path']] = emit(rel, original, item['local_path'], compress=compressed, expected=item['sha256'])
            if item['path'].endswith('/summary.json'): cost_summary = json.loads(original)
        report_folder = 'babylm-dw-cost-profile-audit-20260921'
        audit = load(root / 'results' / report_folder / 'audit.json')
        if not str(audit.get('status', '')).startswith('passed') or audit.get('checks_failed', 0) or audit.get('failures'):
            raise ValueError('Cost audit has not passed')
        for source in sorted((root / 'results' / report_folder).iterdir()):
            if source.is_file() and source.suffix in {'.json', '.md', '.py'}:
                index['reports'][report_folder + '/' + source.name] = copy(source, Path('evidence/reports') / report_folder / source.name)
    old_quality = load(root / 'results/babylm-stage-w-complete-20260921/metrics-audit.json')
    selectors = {
        'D': 'full-dev-dense/windows.jsonl',
        'W_prefix': '/babylm-w-arm-full-dev-20260921-v0/windows.jsonl',
        'W_remainder': '/babylm-w-arm-remainder-20260921-v0/evaluation/windows.jsonl',
    }
    for label, suffix in selectors.items():
        matches = [(p, meta) for p, meta in old_quality['input_files'].items() if p.endswith(suffix)]
        if len(matches) != 1: raise ValueError('First seed raw evidence ambiguous: ' + label)
        source, meta = matches[0]
        index['seed1_raw'][label] = copy(root / source, 'evidence/seed-20260917/' + label + '.windows.jsonl.gz',
                                        compress=True, expected=meta['sha256'])

    quality = load(root / 'results/babylm-dw-seed-full-dev-audit-20260921/metrics-audit.json')
    table = quality['two_seed_descriptive']['paired_results']
    lines = ['# Native sparse prefill 研究快照', '',
             '当前结果：从随机初始化、全参数、第一步固定局部稀疏注意力开始，完成两个配对初始化种子的 BabyLM 一轮训练。'
             'W 在两个种子的完整开发集上都低于密集 D 的 NLL；PPL 分别为 50.07 对 51.33、50.56 对 54.40。'
             '这是小型混合模型上固定局部基线的质量证据，不是新路由方法、完整 Qwen 架构复现或论文创新已成立。', '']
    if cost_summary is not None:
        dshare = cost_summary['arms']['D']['pooled_instrumented_forward_share']['gdn_mixer'] * 100
        wshare = cost_summary['arms']['W']['pooled_instrumented_forward_share']['gdn_mixer'] * 100
        lines += [f"另已完成约 {cost_summary['elapsed_wall_seconds']:.0f} 秒的小型工程诊断：在固定的八窗口样本和共享 GPU 上，"
                  f"GDN 占仪器化前向流耗时约 {dshare:.1f}%（D）与 {wshare:.1f}%（W）。"
                  '这只是定位开销：不代表全训练前向加反向占比，也不能据此宣称端到端加速或成本降低。'
                  '诊断共 36 次前向、32 次反向、0 次参数更新，与科学训练分开记账。', '']
    lines += [
             'Current work: random-initialized, full-parameter BabyLM pretraining in a small 95.391M-parameter hybrid model. '
             'D uses dense global attention; W uses the fixed most-recent 64 complete blocks (4 tokens per block) plus the original causal tail. '
             'W has no learned indexer or auxiliary routing objective.', '',
             '## Verified development-set results', '',
             'Each arm trained for one epoch: 10,001,709 word exposures, 16,325,414 input tokens, 1,413 updates. '
             'Evaluation covers all 18,792 fixed dev windows and 17,418,742 target tokens. Lower is better.', '',
             '| Backbone initialization seed | D NLL | W NLL | D PPL | W PPL | W minus D NLL |',
             '|---|---:|---:|---:|---:|---:|']
    for pair in table:
        d, w = pair['D'], pair['W']
        lines.append(f"| {pair['backbone_seed']} | {d['nll']:.9f} | {w['nll']:.9f} | {d['ppl']:.6f} | {w['ppl']:.6f} | {w['nll']-d['nll']:.9f} |")
    lines += ['', 'The fixed-local baseline has lower dev NLL in both matched initializations. '
              'Data order is held fixed; this is not two fully independent dataset/optimizer sweeps. '
              'The dev set was already used during development. Two seeds do not establish statistical equivalence, '
              'broad generalization, a novel method, or paper-level sufficiency.', '',
              '**No acceleration or cost reduction is established.** Both implementations allocate full attention-score reference tensors. '
              'Hardware migration and shared-GPU timing are not matched speed evidence. '
              'The latest host price is unknown and monetary cost remains null. '
              'Failed earlier work and the interrupted initial three-hour allocation remain in the reports.', '',
              '## Inspect and verify', '',
              'Run `python scripts/verify_github_evidence_v1.py` from this snapshot. It uses only the Python standard library: '
              'hash checks, decompression, original raw-counter and token-weighted metric recomputation, and the unchanged historical verifier. '
              'It performs no model calls, downloads, training, or network operations.', '',
              '- `evidence/seed-20260921/`: 45 hash-verified terminal text artifacts; raw events and dev windows are gzipped.',
              '- `evidence/seed-20260917/`: first-seed D dev rows and the disjoint W prefix/remainder rows.',
              '- `evidence/reports/`: detailed training, quality, execution, and cost audits with their limitations.',
              '- `archives/first-pair-20260919/`: prior published first-pair evidence with its original sources and unchanged verifier. '
              'This earlier ten-pass experiment is historical, not the current one-epoch result.',
              '- `EXPORT_MANIFEST.json`: original-byte SHA, exported-byte SHA, and redaction/compression accounting.', '',
              '## Reproduction boundary', '',
              'This is an auditable source-and-text-evidence package, **not a complete runnable reproduction bundle**. '
              'Raw/tokenized BabyLM corpora, model weights, environment binaries, private keys, API tokens, and live connection settings are excluded. '
              'Scientific source bytes are preserved. Exported protocols and metadata may be redacted; embedded source/original-file hashes '
              'refer to pre-redaction evidence and must not be treated as hashes of the exported JSON. '
              'Cloud launch authorization and connection files are not provided. The public verifier never loads model tensors and does not establish numerical replay.', '',
              'The small hybrid-model results do not reproduce or establish claims about the full Qwen architecture. '
              'Earlier LoRA/CPT experiments are historical and do not answer random-initialized pretraining.', '']
    emit('README.md', '\n'.join(lines).encode())
    emit('EVIDENCE_INDEX.json', (json.dumps(index, indent=2) + '\n').encode())
    emit('.gitignore', (old / '.gitignore').read_bytes(), 'historical-snapshot/.gitignore', preserve=True)
    emit('.gitattributes', b'* -text\n# Preserve original scientific bytes and evidence hashes.\n')
    manifest = {'schema_version': 2, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Public code and selected text evidence snapshot; no publication or novelty claim.',
                'files': entries, 'omitted_operator_helpers': omitted,
                'required_current_scientific_source_sha256': required_scientific,
                'historical_archive_byte_preserved': True, 'selected_terminal_artifacts': 45,
                'selected_cost_artifacts': 6 if args.include_cost_profile else 0,
                'excluded': ['private keys/API tokens/environment credentials', 'concrete cloud connection settings',
                             'raw/tokenized corpora', 'model weights/checkpoints', 'whole logs/results trees',
                             'Python/CUDA environments', 'git repository metadata'],
                'redaction_notice': 'Embedded original hashes refer to source evidence, not redacted export bytes. Both hashes are separately recorded.',
                'secret_scan': {'credential_signatures': 0, 'known_connection_values': 0, 'concrete_ssh_relays': 0,
                                'private_key_paths': 0, 'scanned_files': len(entries),
                                'limitations': 'Pattern and metadata-value scan; not a guarantee against every possible unknown secret.'}}
    manifest_text = json.dumps(manifest, indent=2) + '\n'
    if sensitive(manifest_text): raise ValueError('Manifest itself contains connection metadata')
    (out / 'EXPORT_MANIFEST.json').write_text(manifest_text, encoding='utf-8')
    print(json.dumps({'output': str(out), 'files': len(entries), 'bytes': sum(e['export_bytes'] for e in entries),
                      'omitted_operator_helpers': len(omitted), 'redacted_files': sum(e['redacted'] for e in entries),
                      'manifest_sha256': sha((out / 'EXPORT_MANIFEST.json').read_bytes()), 'model_calls': 0}))


if __name__ == '__main__':
    main()

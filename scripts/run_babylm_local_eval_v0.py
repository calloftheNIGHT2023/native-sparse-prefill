"""Read-only W fixed-local final1413 full-dev evaluation via frozen scorer.

Legacy mode=dense is optimizer/factory API compatibility only. Persistent W
checkpoint identity, explicit condition and the W training receipt are required.
Root binds the unique final model SHA before launch; no checkpoint selection.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_babylm_local_train_v0 as common
require, resolve, sha, load, write, canonical = common.require, common.resolve, common.sha, common.load, common.write, common.canonical


def validate(protocol):
    require(protocol['schema_version'] == 1 and protocol['condition'] == common.CONDITION
            and protocol['engine_compat_mode'] == protocol['mode'] == 'dense', 'Explicit W compatibility/scientific identity required')
    require(protocol['scope'] == 'scientific_evaluation' and protocol['device'] == 'cuda' and protocol['dtype'] == 'float32', 'Frozen CUDA FP32 scientific evaluation required')
    require(protocol.get('window_indices') is None and protocol['checkpoint_format'] == 'model_only', 'Only unique final model, complete dev evaluation is allowed')
    require(protocol.get('enable_routing_diagnostics', False) is False and protocol.get('routing_policy', 'learned') == 'learned', 'No routing intervention allowed for W')
    require(protocol.get('actual_policy') == 'local', 'Actual W policy must be explicit; legacy learned label is compatibility only')
    require(protocol.get('position_diagnostics') is True and protocol.get('torch_num_threads') == 1
            and 0 < protocol['max_wall_seconds'] <= 3300, 'Original positional evaluation/runtime and bounded wall required')
    common.verify_sources({'source_sha256': protocol['expected_source_hashes']})
    ap = resolve(protocol['training_audit_path'])
    require(sha(ap) == protocol['training_audit_sha256'], 'Frozen W training audit differs')
    audit = load(ap)
    require(audit['status'] == 'complete_local_epoch_audited' and audit['condition'] == common.CONDITION
            and audit['engine_compat_mode'] == 'dense', 'W training is incomplete or mislabelled')
    require(all(audit['counts'][k] == v for k, v in common.EXPECTED.items()), 'W one-epoch training counts differ')
    cp = resolve(protocol['checkpoint_path'])
    require(cp == resolve(audit['final_model_checkpoint_path']) and cp.name == common.FINAL_MODEL
            and protocol['checkpoint_sha256'] == audit['final_model_checkpoint_sha256'] == sha(cp), 'Only audited W final1413 checkpoint may be evaluated')
    require(protocol['checkpoint_protocol_sha256'] == audit['protocol_sha256'], 'W training protocol identity differs')
    for name in common.REQUIRED:
        require(protocol['expected_source_hashes'][name] == audit['source_sha256'][name], 'W training/eval source differs: ' + name)
    return audit


@contextmanager
def injected_factory(evaluator):
    from src.babylm_hybrid.local_attention_v0 import build_local_model
    original = evaluator.build_model
    evaluator.build_model = build_local_model
    try: yield
    finally: evaluator.build_model = original


def checkpoint_identity(protocol):
    import torch
    saved = torch.load(resolve(protocol['checkpoint_path']), map_location='cpu', weights_only=True)
    require(saved['protocol']['scientific_condition'] == common.CONDITION
            and saved['protocol']['engine_compatibility_mode'] == 'dense'
            and saved['protocol_sha256'] == protocol['checkpoint_protocol_sha256'], 'Checkpoint lacks W protocol identity')
    require(saved['point']['updates'] == 1413 and saved['resume_supported'] is False, 'Expected immutable final1413 model-only evidence')
    require('local_attention_contract_v0' in saved['model_state'] and not any('.indexer.' in n for n in saved['model_state']), 'W state identity/indexer contract differs')
    for name in common.REQUIRED:
        matches = [v for k, v in saved['source_hashes'].items() if k.replace('\\', '/').endswith(name)]
        require(matches and all(v == protocol['expected_source_hashes'][name] for v in matches), 'Checkpoint misses W source provenance: ' + name)
    del saved


def execute(path, expected_sha):
    path = resolve(path); require(sha(path) == expected_sha, 'Frozen W eval protocol differs')
    protocol = load(path); require(protocol.get('launch_allowed') is True, 'W evaluation launch not frozen')
    training_audit = validate(protocol)
    output = resolve(protocol['output_dir']); require(output.is_relative_to(ROOT / 'results') and not output.exists(), 'Fresh W eval results required')
    receipt_path = output.parent / (output.name + '-local-evaluation-audit.json')
    require(not receipt_path.exists(), 'Existing W evaluation audit cannot be overwritten')
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'status': 'running', 'condition': common.CONDITION, 'engine_compat_mode': 'dense',
               'actual_policy': 'local', 'legacy_row_routing_policy_warning': 'Rows say learned due to frozen scorer API; factory and persistent checkpoint marker enforce fixed-local W without indexer.',
               'protocol_sha256': expected_sha, 'training_protocol_sha256': training_audit['protocol_sha256'],
               'started_utc': datetime.now(timezone.utc).isoformat(), 'mode_warning': 'Frozen scorer mode=dense is compatibility only; factory builds fixed-local W.',
               'scientific_forward_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0}
    write(receipt_path, receipt)
    try:
        checkpoint_identity(protocol)
        from scripts import run_babylm_checkpoint_eval_v0 as evaluator
        with injected_factory(evaluator): summary = evaluator.run_evaluation(protocol)
        receipt['counts'] = summary['counts']; receipt['scientific_forward_calls'] = summary['counts']['forward_calls']
        require(summary['status'] == 'evaluation_complete' and not summary['partial_metrics_only'], 'W full-dev evaluation incomplete')
        require(summary['counts']['forward_calls'] == summary['counts']['forward_attempts'] == summary['counts']['committed_windows'] == 18792
                and summary['optimizer_updates'] == summary['backward_calls'] == 0, 'W full-dev model-call counts differ')
        require(summary['observed_window_indices'] == list(range(18792)), 'W full-dev order/coverage differs')
        require(summary['total']['loss_tokens'] == 17418742 and summary['total']['input_tokens'] == 17437534, 'W frozen dev denominator differs')
        receipt.update(status='complete_local_full_dev_audited', total=summary['total'],
                       summary_sha256=sha(output / 'summary.json'), windows_jsonl_sha256=sha(output / 'windows.jsonl'),
                       checkpoint_sha256=protocol['checkpoint_sha256'], source_sha256=protocol['expected_source_hashes'],
                       scientific_claim='W baseline scoring only; no equivalence or speed claim.')
    except BaseException as e:
        receipt.update(status='local_evaluation_failed', error_type=type(e).__name__, error=str(e), traceback=traceback.format_exc())
        if (output / 'summary.json').exists():
            s = load(output / 'summary.json'); receipt.update(counts=s.get('counts'), scientific_forward_calls=s.get('counts', {}).get('forward_calls', 0))
    receipt['completed_utc'] = datetime.now(timezone.utc).isoformat(); write(receipt_path, receipt)
    print(json.dumps({k: receipt.get(k) for k in ['status', 'condition', 'counts', 'total']}, allow_nan=False))
    return 0 if receipt['status'] == 'complete_local_full_dev_audited' else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True); p.add_argument('--protocol-sha256'); p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.execute:
        require(bool(args.protocol_sha256), 'Explicit frozen protocol SHA required')
        return execute(args.protocol, args.protocol_sha256)
    validate(load(resolve(args.protocol)))
    print(json.dumps({'status': 'static_validation_passed', 'condition': common.CONDITION, 'model_calls': 0}))
    return 0


if __name__ == '__main__': raise SystemExit(main())

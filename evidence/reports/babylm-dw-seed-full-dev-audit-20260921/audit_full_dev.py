"""Independent local new-seed D/W full-dev audit. No models/torch/cloud.

python audit_full_dev.py --manifest <immutable.json> --manifest-sha256 <SHA>
The input collector pins text artifacts; this audit does NOT claim that model
weight files are locally backed up or numerically reloaded.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'results/babylm-dw-seed-confirmation-20260921-v1/'
INIT_SHA = '3f26ce064cc74c3f9544ec8e2fd3ef6a8489096a37bb1ce3c51701d07cb4792c'
DEV_SHA = 'baa53c08d1c26e6dda2f238d2221a1d2cfef4c754ec46327765987f423519619'
MODEL_NAME = 'model-u00001413-w000010001709-i000016325414-l000016302816.pt'
FIELDS = ('window_index', 'source_index', 'source', 'segment_index_in_source',
          'source_token_start', 'source_token_end', 'word_exposures', 'input_tokens', 'loss_tokens')
BINS = ((1, 256), (257, 512), (513, 1024), (1025, 2048))
PINS = {
 'results/babylm-stage-c2-final-20260921/recompute_metrics.py': 'a21f86dd5e459ce8cff99a84cdb63db848c2bc9f659fc059c2d5e6e3b904ad4b',
 'results/babylm-dw-seed-training-audit-20260921/D-audit.json': 'd7fbe9006dc20dbd11845bacb9ae9856f4d5d0e5a7320a7755d1703ed6acd36a',
 'results/babylm-dw-seed-training-audit-20260921/W-audit.json': '2c40ee84acc1a22a762c4660156587e1b465ed805cdcdde0a46ca3a807879e8d',
 'results/babylm-dw-seed-training-audit-20260921/pair-training-audit.json': 'dfb289cfbfc6196e234f46fdc2209507999672b8ae74ea4f54310452060533ad',
 'results/babylm-stage-w-complete-20260921/metrics-audit.json': 'eb2bce7cfacbe5077d23bb7b6d7d851c73b2c34333e9b8d17ad2e87a180ecc12',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def load(path):
    def bad(value): raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=bad)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def logical(path):
    value = str(path).replace('\\', '/')
    marker = '/babylm-optimization-20260920-v0/'
    return value.split(marker, 1)[1] if marker in value else value


class Audit:
    def __init__(self):
        self.report = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                       'checks': 0, 'failures': [], 'input_files': {}, 'analysis_counts': {
                           'model_calls': 0, 'forward_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0,
                           'cuda_calls': 0, 'remote_calls': 0, 'checkpoint_loads': 0}}

    def check(self, condition, message):
        self.report['checks'] += 1
        if not condition:
            self.report['failures'].append(message)
            raise ValueError(message)

    def file(self, path, expected=None):
        path = Path(path).resolve()
        self.check(path.is_relative_to(ROOT), 'File outside project')
        digest = sha(path)
        if expected is not None: self.check(digest == expected, 'SHA differs: ' + str(path))
        self.report['input_files'][path.relative_to(ROOT).as_posix()] = {'sha256': digest, 'size_bytes': path.stat().st_size}
        return path

    def collection(self, path, digest):
        manifest = load(self.file(path, digest))
        self.check(bool(manifest.get('receipts')), 'No final text receipts')
        self.check(manifest.get('status') == 'complete_all_terminal_text_sha_verified', 'Terminal text collector is not complete')
        files = {}
        for receipt in manifest['receipts']:
            self.check(receipt.get('status') in ('downloaded_sha_verified', 'reused_sha_verified'), 'File receipt is not SHA verified')
            name = receipt['remote_relative_path']
            self.check(name not in files and not name.startswith('/') and '..' not in Path(name).parts,
                       'Invalid/duplicate remote relative path')
            local = self.file(ROOT / receipt['local_path'], receipt['sha256'])
            self.check(local.stat().st_size == receipt['size_bytes'], 'Collected file size differs')
            files[name] = local
        self.report['collection_status'] = manifest.get('status')
        self.report['collection_scope'] = 'SHA-verified text artifacts; no assertion of complete local weight backup'
        return files


def read_layout(a):
    path = a.file(ROOT / 'data/babylm-dev-windows-v0/manifest.json', DEV_SHA)
    manifest = load(path)
    a.check(manifest['total_windows'] == 18792, 'Dev total windows differs')
    for artifact in manifest['artifacts']:
        a.file(path.parent / artifact['path'], artifact['sha256'])
    for source in manifest['source_summaries']:
        a.file(ROOT / source['token_ids_path_relative_to_project'], source['token_ids_sha256'])
    index = np.load(path.parent / 'windows.u64.npy', mmap_mode='r', allow_pickle=False)
    a.check(index.shape == (18792, len(manifest['columns'])), 'Dev index shape differs')
    return manifest, index, {key: i for i, key in enumerate(manifest['columns'])}


def validate_layout(a, rows, layout, arm, cfg):
    manifest, index, columns = layout
    sources = {item['source_index']: item['source'] for item in manifest['source_summaries']}
    layers = cfg['layer_types'].count('global'); B = cfg['block_size']; K = cfg['selected_complete_blocks']
    support = {n: layers * sum(q + 1 - max(0, (q + 1) // B - K) * B for q in range(n))
               for n in {r['input_tokens'] for r in rows}}
    a.check([r['window_index'] for r in rows] == list(range(18792)), arm + ':exact IDs/order')
    for row in rows:
        source = index[row['window_index']]
        for name in ('source_index', 'segment_index_in_source', 'source_token_start', 'source_token_end', 'word_exposures', 'input_tokens'):
            a.check(type(row[name]) is int and row[name] == int(source[columns[name]]), arm + ':original window metadata ' + name)
        targets, n = row['loss_tokens'], row['input_tokens']
        a.check(targets == int(source[columns['next_token_loss_positions']]) == n - 1, arm + ':target ledger')
        a.check(row['source'] == sources[row['source_index']], arm + ':source name')
        a.check(row['grad_enabled'] is False and row['model_training'] is False and row['forward_calls'] == 1, arm + ':evaluation mode')
        if targets:
            a.check(math.isclose(row['nll'] * targets, row['nll_sum'], rel_tol=0, abs_tol=1e-6), arm + ':row LM reduction')
            a.check(math.isclose(math.exp(row['nll']), row['ppl'], rel_tol=1e-12), arm + ':row PPL')
        else:
            a.check(row['nll'] is row['ppl'] is None and row['nll_sum'] == 0, arm + ':zero target')
        parts = row['query_history_bins']
        a.check(set(parts) == {f'{lo}-{hi}' for lo, hi in BINS}, arm + ':fixed bins')
        for lo, hi in BINS:
            part = parts[f'{lo}-{hi}']
            a.check(part['loss_tokens'] == max(0, min(targets, hi) - lo + 1), arm + ':position denominator')
            a.check(math.isfinite(part['nll_sum']), arm + ':position finite')
            a.check(part['loss_tokens'] > 0 or part['nll_sum'] == 0, arm + ':empty position')
        a.check(math.isclose(math.fsum(x['nll_sum'] for x in parts.values()), row['nll_sum'], rel_tol=1e-5, abs_tol=1e-5), arm + ':position scalar reduction')
        a.check(row['window_length_bin'] == next(f'{lo}-{hi}' for lo, hi in BINS if lo <= n <= hi), arm + ':length bin')
        attention = row['attention_counts']; dense = layers * n * (n + 1) // 2
        a.check(attention['logical_dense_causal_pairs'] == dense and attention['logical_kept_pairs'] == (dense if arm == 'D' else support[n]), arm + ':attention support')
        a.check(attention['indexer_score_elements'] == attention['indexer_zero_score_visible_query_count'] == 0, arm + ':no indexer')


def bind_arm(a, arm, files, master, priors):
    def get(suffix):
        name = PREFIX + arm + '/' + suffix
        a.check(name in files, 'Missing complete final artifact: ' + name)
        return files[name]
    train = load(get('train/run/protocol.json')); trsummary = load(get('train/run/summary.json'))
    traudit = load(get('train/audit.json')); protocol = load(get('bound-eval.json'))
    bound = load(get('checkpoint-binding.json')); summary = load(get('eval/summary.json'))
    evaudit = load(get('eval-audit.json')); worker = load(get('eval-worker.json'))
    prior = priors[arm]['epoch_and_tail_audit']
    for name, digest in prior['input_file_sha256'].items():
        a.check(sha(get('train/run/' + name)) == digest, arm + ':previous independently audited training bytes unchanged ' + name)
    a.check(worker['status'] == 'complete' and evaudit['status'] == 'complete_seed_full_dev_audited', arm + ':terminal worker/audit')
    a.check(traudit['status'] == 'complete_seed_epoch_audited' and traudit['arm'] == arm, arm + ':training terminal')
    a.check(protocol == load(get('eval/protocol.json')), arm + ':actual eval protocol equals bound protocol')
    a.check(protocol['condition'] == ('D_dense' if arm == 'D' else 'W_fixed_local')
            and protocol['actual_policy'] == ('dense' if arm == 'D' else 'local'), arm + ':scientific condition')
    a.check(protocol['mode'] == protocol['engine_compat_mode'] == 'dense', arm + ':legacy compatibility mode')
    a.check(protocol['backbone_seed'] == train['backbone_seed'] == master['backbone_seed'] == 20260921, arm + ':new seed')
    a.check(train['data_order_seed'] == 20260919 and train['max_epochs'] == 1, arm + ':training schedule identity')
    a.check(protocol['window_indices'] is None and protocol['scope'] == 'scientific_evaluation' and protocol['dtype'] == 'float32', arm + ':full FP32 request')
    a.check(protocol['checkpoint_format'] == 'model_only' and Path(protocol['checkpoint_path']).name == MODEL_NAME, arm + ':only final1413 model')
    a.check(sha(get('train/audit.json')) == protocol['training_audit_sha256'] == bound['training_audit_sha256'], arm + ':training audit binding')
    a.check(canonical(train) == protocol['checkpoint_protocol_sha256'] == traudit['protocol_sha256'] == summary['checkpoint']['training_protocol_sha256'], arm + ':training protocol chain')
    model_sha = protocol['checkpoint_sha256']
    a.check(model_sha == bound['checkpoint_sha256'] == traudit['final_model_checkpoint_sha256'] == summary['checkpoint']['sha256'] == evaudit['checkpoint_sha256'], arm + ':checkpoint SHA provenance chain')
    a.check(logical(protocol['checkpoint_path']) == logical(bound['checkpoint_path']) == logical(traudit['final_model_checkpoint_path']) == logical(summary['checkpoint']['path']), arm + ':checkpoint path chain')
    a.check(summary['checkpoint']['point']['updates'] == 1413 and summary['checkpoint']['point']['word_exposures'] == 10001709, arm + ':checkpoint point')
    a.check(traudit['initial_parameter_hashes_sha256'] == master['expected_initial_parameter_hashes_sha256'] == INIT_SHA
            and canonical(trsummary['initial_parameter_hashes']) == INIT_SHA, arm + ':shared initial backbone')
    a.check(canonical(protocol) == summary['protocol_sha256'] == evaudit['protocol_sha256'], arm + ':evaluation canonical identity')
    a.check(summary['source_hashes'] == protocol['expected_source_hashes'] == master['source_sha256'], arm + ':source pins')
    a.check(summary['transformers_gdn_source']['sha256'] == protocol['transformers_gdn_source_sha256'], arm + ':GDN source')
    a.check(summary['runtime'] == protocol['expected_runtime'], arm + ':runtime exact')
    a.check(summary['data_fingerprint']['manifest_sha256'] == protocol['dev_manifest_sha256'] == DEV_SHA, arm + ':dev manifest')
    a.check(sha(get('eval/windows.jsonl')) == summary['windows_jsonl_sha256'] == evaudit['windows_sha256'], arm + ':raw-window hash chain')
    a.check(sha(get('eval/summary.json')) == evaudit['summary_sha256'], arm + ':eval summary SHA')
    template = load(a.file(ROOT / master['eval_template'], master['eval_template_sha256']))
    expected = dict(template)
    expected.update(condition=protocol['condition'], actual_policy=protocol['actual_policy'], engine_compat_mode='dense',
                    backbone_seed=20260921, checkpoint_path=protocol['checkpoint_path'], checkpoint_sha256=model_sha,
                    checkpoint_protocol_sha256=canonical(train), output_dir=PREFIX + arm + '/eval',
                    training_audit_path=PREFIX + arm + '/train/audit.json', training_audit_sha256=sha(get('train/audit.json')), launch_allowed=True)
    a.check(protocol == expected, arm + ':only prescribed final binding changed template')
    a.check(bound['master_protocol_sha256'] == canonical(master) and bound['template_file_sha256'] == master['eval_template_sha256'], arm + ':binding master/template identity')
    final = trsummary['snapshot_state']['final_receipts']
    a.check(len(final) == 1 and final[0]['model_sha256'] == model_sha and final[0]['point']['updates'] == 1413, arm + ':unique final snapshot receipt')
    final_logical = PREFIX + arm + '/train/run/' + final[0]['path']
    a.check(final_logical in files, arm + ':final snapshot receipt text available')
    a.check(sha(files[final_logical]) == final[0]['sha256'] == traudit['final_model_receipt_sha256'], arm + ':final model receipt SHA')
    final_doc = load(files[final_logical])
    a.check(final_doc['model_sha256'] == model_sha and final_doc['counts'] == trsummary['counts']
            and final_doc['protocol_sha256'] == canonical(train), arm + ':final receipt content')
    return protocol, summary, get('eval/windows.jsonl'), get('eval/summary.json')


def run(a, manifest, digest):
    for path, pin in PINS.items(): a.file(ROOT / path, pin)
    helper_path = ROOT / 'results/babylm-stage-c2-final-20260921/recompute_metrics.py'
    spec = importlib.util.spec_from_file_location('frozen_metrics_helpers', helper_path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    files = a.collection(manifest, digest)
    a.check(PREFIX + 'master.json' in files and PREFIX + 'stage.json' in files, 'Final queue master/stage absent')
    master = load(files[PREFIX + 'master.json']); stage = load(files[PREFIX + 'stage.json'])
    a.check(stage['status'] == 'complete_pending_independent_local_audit', 'Queue has not completed all prescribed stages')
    a.check([x['stage'] for x in stage['stages']] == ['train_D', 'train_W', 'eval_D', 'eval_W']
            and all(x['status'] == 'complete' and x['returncode'] == 0 for x in stage['stages']), 'Terminal stage sequence differs')
    for name, pin in master['source_sha256'].items(): a.file(ROOT / name, pin)
    priors = {arm: load(ROOT / f'results/babylm-dw-seed-training-audit-20260921/{arm}-audit.json') for arm in ('D', 'W')}
    layout = read_layout(a); values = {}; rows = {}; summaries = {}; protocols = {}; aggregator = helper.Audit()
    for arm in ('D', 'W'):
        protocols[arm], summaries[arm], wp, sp = bind_arm(a, arm, files, master, priors)
        values[arm], rows[arm], _ = aggregator.dev(arm, wp, sp)
        a.check(not aggregator.report['failures'], arm + ':independent frozen aggregate checks failed: ' + str(aggregator.report['failures'][:5]))
        validate_layout(a, rows[arm], layout, arm, protocols[arm]['model_config'])
    a.check(protocols['D']['model_config'] == protocols['W']['model_config'], 'Model configs differ')
    a.check(summaries['D']['runtime'] == summaries['W']['runtime'], 'D/W scoring runtimes differ')
    a.check(values['D']['normalized_data_fingerprint'] == values['W']['normalized_data_fingerprint'], 'D/W data fingerprint differs')
    for d, w in zip(rows['D'], rows['W']):
        a.check(all(d[k] == w[k] for k in FIELDS), 'D/W original row identity differs')
        a.check({k: v['loss_tokens'] for k, v in d['query_history_bins'].items()} == {k: v['loss_tokens'] for k, v in w['query_history_bins'].items()}, 'Paired position denominators differ')
    difference = {'total': helper.compare(values['W']['total'], values['D']['total'])}
    for key in ('per_source', 'position'):
        a.check(set(values['D'][key]) == set(values['W'][key]), 'Paired aggregation groups differ')
        difference[key] = {name: helper.compare(values['W'][key][name], values['D'][key][name]) for name in values['D'][key]}
    difference['per_source_position'] = {source: {name: helper.compare(cell, values['D']['per_source_position'][source][name])
                                                for name, cell in bins.items()} for source, bins in values['W']['per_source_position'].items()}
    old = load(ROOT / 'results/babylm-stage-w-complete-20260921/metrics-audit.json')
    a.check(old['status'] == 'passed_complete_two_segment_W_union_audit' and not old['failures'], 'Old seed audit not passed')
    oldD, oldW = old['full_dev']['D']['total'], old['full_dev']['W']['total']
    seeds = [{'backbone_seed': 20260917, 'D': oldD, 'W': oldW, 'W_minus_D': helper.compare(oldW, oldD)},
             {'backbone_seed': 20260921, 'D': values['D']['total'], 'W': values['W']['total'], 'W_minus_D': difference['total']}]
    a.report.update(status='passed_independent_new_seed_D_W_full_dev_audit', full_dev=values, W_minus_D=difference,
                    backbone_seed=20260921, shared_initial_parameter_hashes_sha256=INIT_SHA,
                    exact_window_metadata_and_position_denominators_verified=True,
                    frozen_helper_checks=aggregator.report['checks'], frozen_helper_failures=aggregator.report['failures'],
                    checkpoint_provenance_sha256={arm: protocols[arm]['checkpoint_sha256'] for arm in ('D', 'W')},
                    weight_files_locally_audited=False, weight_checkpoint_loads=0,
                    two_seed_descriptive={'paired_results': seeds, 'mean_paired_nll_difference': math.fsum(x['W_minus_D']['nll_difference'] for x in seeds) / 2,
                                          'inference_scope': 'Two matched seeds on already used dev; no inferential test or equivalence declaration.'},
                    physical_counts_new_full_dev={'forward_attempts': 37584, 'forward_calls': 37584, 'committed_windows': 37584, 'backward_calls': 0, 'optimizer_updates': 0},
                    unknown_hourly_rate_usd=None,
                    limitations=['This text/raw-metrics audit does not establish complete local checkpoint backup or tensor replay.',
                                 'Two matched seeds on the same existing dev do not establish statistical equivalence or broad generalization.',
                                 'Fixed local W has no learned indexer; evidence does not validate a new routing method or novelty.',
                                 'Both use full-score reference implementations; no end-to-end cost/speed advantage follows.',
                                 'Training updates, panels, old failed work, migration replay and full-dev forwards remain separate ledgers.'])
    a.check('torch' not in sys.modules, 'Audit unexpectedly imported torch')


def report_text(report):
    values = report['full_dev']; diff = report['W_minus_D']['total']
    lines = ['# 新种子 D/W 完整开发集独立审计', '', '逐窗原始记录审计通过。每组18792窗，17418742监督token，0反向/0更新。', '',
             '| 指标 | D | W |', '|---|---:|---:|',
             f"| NLL | {values['D']['total']['nll']:.12f} | {values['W']['total']['nll']:.12f} |",
             f"| PPL | {values['D']['total']['ppl']:.12f} | {values['W']['total']['ppl']:.12f} |", '',
             f"W−D NLL={diff['nll_difference']:.12f}，W相对PPL变化={100*diff['ppl_relative_difference']:.4f}%。", '',
             '| 来源 | D PPL | W PPL | W−D NLL |', '|---|---:|---:|---:|']
    for name, d in values['D']['per_source'].items():
        w = values['W']['per_source'][name]
        lines.append(f"| {name} | {d['ppl']:.6f} | {w['ppl']:.6f} | {w['nll']-d['nll']:.6f} |")
    lines += ['', '| 主干seed | D PPL | W PPL | W−D NLL |', '|---|---:|---:|---:|']
    for seed in report['two_seed_descriptive']['paired_results']:
        lines.append(f"| {seed['backbone_seed']} | {seed['D']['ppl']:.6f} | {seed['W']['ppl']:.6f} | {seed['W_minus_D']['nll_difference']:.6f} |")
    lines += ['', '以上是两个配对种子的描述性结果，不是统计等价/广泛泛化、方法新颖性或端到端加速结论。',
              '本审计仅验证原始评分与元数据链；未加载权重，不能据此宣称权重本地完整备份。费用未知仍为空。',
              '所有原始SHA、token与位置分母、来源×位置差值及限制见 metrics-audit.json。审计新增模型/远端调用为0。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', default=str(Path(__file__).with_name('metrics-audit.json')))
    args = parser.parse_args(); output = Path(args.output).resolve()
    require(output.parent == Path(__file__).parent and not output.exists(), 'Fresh audit output in this result directory required')
    a = Audit(); a.report['script_sha256'] = sha(__file__)
    try: run(a, (ROOT / args.manifest).resolve(), args.manifest_sha256)
    except BaseException as error:
        a.report.update(status='failed_independent_full_dev_audit', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    a.report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(a.report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    success = a.report['status'] == 'passed_independent_new_seed_D_W_full_dev_audit'
    if success:
        text = output.with_suffix('.md'); require(not text.exists(), 'Report output exists'); text.write_text(report_text(a.report), encoding='utf-8')
    print(json.dumps({'status': a.report['status'], 'checks': a.report['checks'], 'failure': a.report.get('error'),
                      'full_dev': {arm: value['total'] for arm, value in a.report.get('full_dev', {}).items()},
                      'audit_sha256': sha(output)}, ensure_ascii=False))
    return 0 if success else 2


if __name__ == '__main__': raise SystemExit(main())

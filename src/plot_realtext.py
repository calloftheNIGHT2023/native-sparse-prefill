"""Export a scientific figure from saved diagnostic results; no new experiments."""
import argparse
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    diagnostics = json.loads((args.runs / 'summary.json').read_text(encoding='utf-8'))
    nll = json.loads((args.nll / 'summary.json').read_text(encoding='utf-8'))
    methods = ['selected_only', 'outside_probe', 'dense_warmup', 'full_supervision', 'selected_budget_match']
    labels = ['Selected', '+ probes', 'Warm-up', 'Full target', 'More steps']
    colors = ['#546a7b', '#d55e00', '#8a89a6', '#009e73', '#0072b2']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for col, layer in enumerate([1, 4]):
        rows = {r['method']: r for r in diagnostics['summary'] if r['layer'] == layer}
        axes[0, col].bar(np.arange(5), [rows[m]['mean']['relative_output_error'] for m in methods],
                         yerr=[rows[m]['sample_std']['relative_output_error'] for m in methods], color=colors, capsize=3)
        means, stds = [], []
        for method in methods:
            values = [r['mean_delta_vs_dense'] for r in nll['results'] if r['layer'] == layer and r['method'] == method]
            means.append(statistics.mean(values))
            stds.append(statistics.stdev(values))
        axes[1, col].bar(np.arange(5), means, yerr=stds, color=colors, capsize=3)
        axes[0, col].set_title(f'Layer {layer + 1} (index {layer})')
        for row in range(2):
            axes[row, col].set_xticks(np.arange(5), labels, rotation=18)
            axes[row, col].grid(axis='y', alpha=.2)
            axes[row, col].set_axisbelow(True)
    axes[0, 0].set_ylabel('Relative attention-output error\n(lower is better)')
    axes[1, 0].set_ylabel('Next-token NLL increase over dense\n(nats; lower is better)')
    fig.suptitle('Pythia 70M / WikiText: frozen-backbone diagnostic', fontsize=15)
    fig.supxlabel('Mean ± sample SD over 3 indexer seeds; 16 reused test paragraphs. No backbone training or speed benchmark.', fontsize=9)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / 'realtext-diagnostic.png', dpi=180)
    fig.savefig(args.output / 'realtext-diagnostic.svg')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--nll', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

"""CLI for a frozen bounded D/E baseline protocol; no launch overrides."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from babylm_hybrid.training import train_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--mode', choices=('dense', 'sparse'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--stop-after-updates', type=int)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if protocol.get('device') == 'cuda':
        # Match the separate numerical gate in every fresh child process.
        # These settings do not carry over from that child's successful exit.
        import torch
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    result = train_run(protocol, args.mode, args.output_dir, args.resume_checkpoint, args.stop_after_updates)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

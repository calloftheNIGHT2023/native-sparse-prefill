# Backward review kit

This kit is preparation, not a validated fix. It has no model weights and needs no training.

Use the verified Python 3.11 / torch 2.8.0+cu128 / Triton 3.4 environment and original FlashMoBA wheel from the previous stage. Capture GPU name and extension SHA. The old compiled wheel was tested on RTX6000Ada; a new architecture must pass its own correctness gates.

Original check from project root:

    .venv-flashmoba-v0/bin/python scripts/verify_long_backward_candidate.py --output results/flashmoba-long-backward-original-v0 --label original --repeats 16

For candidate validation, copy the pinned FlashMoBA source including its CUTLASS submodule into a separate directory, verify and apply the candidate patch there, then build using the recorded CUDA12.9.1 toolchain and existing environment. Do not install over the original wheel. Run the candidate from a fresh process with its directory on PYTHONPATH and a different output directory. Confirm result.json points at the candidate .so and has a different SHA; otherwise the comparison is invalid.

Compare actual K4 masks first. Mask mismatch invalidates the conditional FP64 comparison; regenerate the reference with actual mask. Predeclared checks: reference relative L2 <= .05 on selected rows, deterministic Q/K/V identical across16 repeats, plus the existing standard correctness suite. No automatic long training follows these diagnostics.

Source and original wheel/toolchain restore details remain in docs/flashmoba-cloud-restore.md and previous artifacts. The kit alone is not a standalone environment installer.

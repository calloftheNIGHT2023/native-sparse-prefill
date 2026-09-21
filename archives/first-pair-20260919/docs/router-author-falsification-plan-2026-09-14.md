# Native top-8 training deficit: falsification plan v0

Frozen before new updates, 2026-09-14 UTC. User requests continuation until a defensible rejection or paper-supporting result. This plan tests the current candidate, not a promise of acceptance.

## Claim and decision

The observed 19-epoch gap is a valid finite-budget observation. The stronger candidate interpretation is that hard top-8 training has a persistent learning obstruction on author MQAR, rather than simply requiring a longer ordinary optimization run. We have not established this interpretation.

Stage A: resume the existing exact-QK top-8 checkpoint at epoch 19 to epoch 64 with unchanged model, mask, training data, fixed order, batch 256, AdamW, learning rate schedule, optimizer moments, and saved RNG. No scheduler restart, dense steps, auxiliary objective, or new learned router. Save every epoch and milestones 24/32/48/64. Primary development success: at least 99% accuracy at the final epoch. Earlier crossings and instability are descriptive.

If Stage A passes, the interpretation of persistent failure on this run is rejected. Confirm the ordinary native training result with initialization/training RNG seed 124, and a paired dense control, both run for 64 epochs under the same schedule. A success falsifies inability on this task; it does not prove equivalence of compute efficiency or all seeds. If confirmation fails, report the seed sensitivity and do not erase the first counterexample.

If Stage A fails, test the author-provided alternate learning rate 0.002154434690031882 from the original seed-123 initialization for 64 epochs, before inventing a new optimization mechanism. Failure under these finite settings cannot prove universal impossibility. Any further intervention must register a distinct prediction and compare against primary literature first.

Evaluation is separate from training: after the relevant fits finish, generate a fresh clean set (seed 2026091701, 1024 examples), source-value swaps (one supervised answer per row), and filler-noise intervention (seed 2026091702). Exclude row overlap with train, development and all three previous exposed clean test sets. Evaluate frozen final checkpoints without test-based checkpoint selection. Replay predictions on CPU, verify label construction and immutable checkpoint hashes. If a subsequent intervention is designed using these results, this test is then exposed and cannot serve as its new test.

## Accounting and correctness

An epoch is 391 main updates, 25,600,000 input tokens and 1,600,000 supervised answers. Stage A adds 17,595 updates and 1,152,000,000 input tokens; do not count its old 7,429-update prefix again. No indexer updates. Each independent 64-epoch fit adds 25,024 updates and 1,638,400,000 tokens. Record disposable resume-equivalence test updates separately from scientific fits.

Verify serialized model/optimizer/scheduler/RNG continuation against uninterrupted updates, including dropout. All new result directories must be absolute and newly created. Protect parent files by SHA. Snapshot source and configuration. Log per-step loss, learning rate, time and per-epoch development metrics; retain optimizer/RNG checkpoints, failure tracebacks and UTC timestamps.

Each fit has a 1,800-second wall cap. At most three scientific fits in this plan; at most 90 minutes active compute on the existing A40, estimated well below $2 at the previously seen $0.49/hour (not a verified current invoice). Keep the $500 overall ceiling and original $30 first-stage envelope; actual prior bill remains unknown. No new Pod, purchase, or scaling. GPU idle is not billing stopped.

## Novelty gate

Read primary papers, particularly SSA (arXiv:2511.20102), The emergence of sparse attention (2505.17863), and Transformers Learn Faster with Semantic Focus (2506.14095). Distinguish hard enforced sparsity from learned concentration under dense attention. A generic sparse-gradient/warmup/plateau observation is not a new contribution. If the apparent obstacle is resolved by ordinary training and generic explanation overlaps existing work, close this candidate rather than relabeling a routine result as a paper.

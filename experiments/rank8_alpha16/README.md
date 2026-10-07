# Rank 8 / alpha 16 comparison

Run from the original development project using the existing myenv environment.
Training reuses exactly the original 640 pairs and frozen feature caches.
The primary comparison is epoch 3 / step 1920; epoch 6 / step 3840 is secondary.
Both ranks share three lowlight and three deblur images after decoded-RGB overlap checks.

- `protocol.json`: frozen settings, source hashes, and baseline metadata.
- `configs/`: exact training command, frozen train config and clean evaluation manifest.
- `weights/`: this rank-8 run only.
- `source_images/`: physically copied input and target pairs.
- `results/rank*_epoch*/`: generated images, per-image CSV and aggregate metrics.
- `reports/comparison.json` and `comparison.md`: paired comparison and limitations.
- `logs/`: timestamped fixed-name logs; an intentional rerun overwrites only its own log.
- `status.json`: pipeline phase, step count, PID and errors.

Inspect progress: `python code/rank_ablation.py status` from this experiment directory.
The published final project and selected rank-16 adapter are not replaced by this experiment.

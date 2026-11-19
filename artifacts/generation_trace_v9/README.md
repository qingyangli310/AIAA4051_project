# Best v9 Generator checkpoint

This archive is the completed v9 Trace-only Flan-T5-base run `853633` from
the school HPC. Dev selection chose epoch 3. The model weights in
`checkpoint-final` and `epoch_03/checkpoint` have the same SHA-256, so the
archive stores one copy of the best model.

- Dev100 local diagnostic success: **66/100**.
- Dev100 exact match to the single reference: **60/100**.
- Training: 2,000 frozen Generation examples, 100 fixed Dev examples,
  seed 42, 3 epochs, 750 steps.
- Model weight file: `checkpoint-final/model.safetensors`, 990,345,064 bytes,
  SHA-256 `bbf45876009d69b0907a36066ae55404496f34455175f18c3ae934df49aaf9`.
- `dataset_ids.json` is the original file exported by the run. The
  `data/transformed_*.jsonl` files preserve the exact model input/target
  text used for training and Dev, without adding any test examples.
- `results/generation_trace_v9/run_report.json` records model asset hashes,
  training parameters, split provenance and final metrics.

The weight file uses **Git LFS**. After cloning this branch, run `git lfs pull`
before loading `checkpoint-final` with Transformers. The checkpoint includes
the tokenizer and model configuration. The stored `training_args.bin` is a
PyTorch serialization artifact; loading the model for inference does not
require that file.

The 66/100 result is a local movement diagnostic on Dev, not the official
member A executor score or a held-out test result. ID inference and the
official executor evaluation should be added as separate result artifacts
after they complete.

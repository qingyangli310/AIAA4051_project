# Selected Flan-T5-base Repair checkpoint

This archive contains the selected two-epoch Repair checkpoint trained from
`google/flan-t5-base` revision
`7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` on
`data/repair_high_quality_v1`.

- Seed: 42
- Train / Dev: 1,000 / 90
- Optimizer: Adafactor
- Learning rate: 2e-4
- Per-device batch size: 1
- Gradient accumulation: 4
- Gradient checkpointing: enabled
- Load dtype and training precision: BF16
- Completed steps: 500
- Development Repair success: 68/90
- Development parse count: 84/90
- Development exact match: 67/90
- Mean optimality: 1.0

The base model file downloaded for the run had SHA-256
`1dfb70afdcedceb9f9fae2f9b68e004ad934361fb35b9b2bd50b45ea90790fc8`.
The selected checkpoint `model.safetensors` has SHA-256
`5ef885e4e3f1c88c1dc4c1a8ccda09f0f31563f3c54c5de329b6b068cf29968f`.

The weight file is stored with Git LFS. Run `git lfs pull` after cloning
before loading this checkpoint.

This is Development-only evidence. The checkpoint has not yet been evaluated
on frozen ID/OOD data or in the complete same-family pipeline.

# Verification trace v1 checkpoint

This is the selected Verification checkpoint trained on the frozen 1,000-example
`verification_trace_v1` training set. It was initialized from the frozen
Generator v9 Flan-T5-base checkpoint, then fine-tuned on CUDA for exactly three
epochs. Only the 50-example Verification Dev split was used for model comparison;
the frozen ID/OOD test splits were not opened.

## Reproducibility record

- Base family: `google/flan-t5-base`
- Base revision: `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`
- Starting Generator v9 weights SHA-256: `bbf45876009d69b0907a36066ae55404496f34455175f18c3ae934df49aafaf9`
- Final weights SHA-256: `c125484941f4744540cfd250ade8ca1d921f0efb8d6a88fa22a03cc083a474f1`
- Train data SHA-256: `8637b26069579c5946b21762edc76f64648a1703ec78a8df0065efef4fe46997`
- Dev data SHA-256: `29411101876b3a257e3456aff393d14011bd6473132e2a70b2aa4dff0bbf2ecc`
- Seed: 42
- Optimizer: AdamW
- Learning rate: `2e-4`
- Effective batch size: 4 (`batch_size=1`, `gradient_accumulation=4`)
- Precision: BF16 load and training
- Gradient checkpointing: enabled
- Epochs: 3 (750 optimizer steps)
- CUDA training time: 431.3 seconds

The model emits an execution trace followed by an explicitly labelled
`Verdict:`. Runtime parsing exposes only the strict Project 6 verdict contract:
`valid` or `invalid | step=<index> | reason=<error_type>`.

See `results/verification_high_quality_v1/` for the preserved Dev predictions,
normalized scoring, baseline comparison, and full run report.

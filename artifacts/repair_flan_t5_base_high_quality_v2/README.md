# Flan-T5-base Repair high-quality v2 checkpoint

This is the selected three-epoch Repair checkpoint trained from the pinned
original `google/flan-t5-base` snapshot, not from the v1 Repair checkpoint.

- Base revision: `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`
- Base weight SHA-256: `1dfb70afdcedceb9f9fae2f9b68e004ad934361fb35b9b2bd50b45ea90790fc8`
- Checkpoint weight SHA-256: `c097e2031d30e540e4549c076ce1bf39b475bfa4b4873c518db794c1ad0351d5`
- Train data SHA-256: `a0d826e31a04eea953890d87a7850fa7eafc1dca83777f2b368bbd1414f79069`
- Dev data SHA-256: `f1b9744c3f57ff6fb7c72cf3c42379e628d3f24906d2596f1e22808def8008a0`
- Seed: 42
- Epochs: 3

The model weight is tracked through Git LFS. Verify it before use:

```bash
sha256sum artifacts/repair_flan_t5_base_high_quality_v2/checkpoint-final/model.safetensors
```

Development results and provenance are in
`results/repair_flan_t5_base_high_quality_v2/`.

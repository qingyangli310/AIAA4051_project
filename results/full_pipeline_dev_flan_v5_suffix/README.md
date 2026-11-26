# Repair v5 suffix: Dev-only ablation

Flan-T5-base was trained for three epochs on 500 feedback-suffix Train rows,
seed 42. The independent no-feedback branch used its existing v4 checkpoint
trained on 500 rows. The Generator, Verifier, and Dev100 input match the v4
comparison. This run is exploratory; no frozen ID/OOD Test was run for v5.

| Condition | End-to-end Dev100 success | Repair success among 34 failures |
| --- | ---: | ---: |
| Direct | 66/100 | — |
| Independent no-feedback v4 | 68/100 | 2/34 |
| Self feedback v5 | 65/100 | 0/34 |
| Correct executor feedback v5 | 69/100 | 3/34 |
| Shuffled executor feedback v5 | 66/100 | 0/34 |

All 100 rows completed with zero runtime errors. The suffix syntax parsed on
all 34 natural failures, but only three reached the goal. Independent v5 Dev50
had 49/50 parses and 10/50 successful repairs. Thus the shorter output fixed
much of the format issue but did not learn reliable route completion. v5 is
not selected; v4 remains the Dev-selected Repair checkpoint. The previously
recorded formal v4 ID/OOD results remain unchanged.

The v5 checkpoint remains on the GPU host at
`/data/liqingyang/project6_repair_v4_20261126/results/flan_repair_v5_suffix/checkpoint-final`.
Machine-readable reports and per-example records are archived here.

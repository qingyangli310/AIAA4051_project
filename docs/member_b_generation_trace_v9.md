# Member B: Generation Trace-only experiment

The earlier grid-state model emitted both `Trace:` and `Actions:`. The action
line sometimes differed from the trace. This experiment makes the model emit
only `Trace:` and derives the exported action list from that one output.

The implementation is in [`generation_trace_v9/`](../generation_trace_v9/).
It preserves the existing Train2000 and Dev100 environment split, model,
seed and training schedule. The only required local code dependency from the
previous Generation package is
[`generation_models_v5/model_assets.py`](../generation_models_v5/model_assets.py),
which is included in this branch. The prepared Train2000 and Dev100 data,
offline model folder and its manifest are required on the HPC filesystem;
they are not uploaded with this branch. See the package README for exact paths
and submission commands.

The downloaded result artifacts and detailed comparison are in
[`results/generation_trace_v9/`](../results/generation_trace_v9/).
On the fixed Dev100, the local movement check found 66 successful v9 paths,
compared with 48 from v7 `Actions` and 47 from v8 hard-example weighting.
The v9 exact-reference-match count was 60; valid alternate paths can differ
from the one reference. Of the 45 long-path examples, 20 succeeded; of the
10 required-detour examples, one succeeded. All 34 remaining v9 failures
collided with obstacles.

These results have **not** yet been evaluated by member A's official executor.
The downloaded v9 `run_report.json` was also unavailable, so the archived
record does not claim its model asset hashes, selected checkpoint path or GPU
identity. Add that report and the official executor evaluation when available.

"""Compute exact McNemar comparisons from audited per-example results."""
import json
from math import comb

from audit_final_pipeline_results import audit

COMPARISONS = (("no_feedback", "executor_feedback"),
               ("no_feedback", "self_feedback"),
               ("executor_feedback", "shuffled_executor_feedback"))


def compare(rows, a, b):
    successes = [(row["branches"][a]["final_result"]["success"],
                  row["branches"][b]["final_result"]["success"]) for row in rows]
    both = successes.count((True, True))
    neither = successes.count((False, False))
    a_only = successes.count((True, False))
    b_only = successes.count((False, True))
    discordant = a_only + b_only
    return {
        "both_success": both, "both_fail": neither, "a_only": a_only, "b_only": b_only,
        "a_success_rate": (both + a_only) / len(rows),
        "b_success_rate": (both + b_only) / len(rows),
        "b_minus_a": (b_only - a_only) / len(rows),
        "mcnemar_exact_p": min(1, 2 * sum(comb(discordant, k)
                              for k in range(min(a_only, b_only) + 1)) / 2**discordant)
        if discordant else None,
    }


if __name__ == "__main__":
    records, audit_summary = audit()
    print(json.dumps({
        split: {"comparisons": {f"{a} vs {b}": compare(rows, a, b) for a, b in COMPARISONS},
                "shuffle_changed": audit_summary[split]["shuffle_changed"],
                "interpretation": "Exact two-sided tests on discordant pairs; unadjusted, "
                "requires independent pairs. No discordances gives no informative test. "
                "Zero changed feedback gives no feedback-content contrast."}
        for split, rows in records.items()
    }, indent=2))

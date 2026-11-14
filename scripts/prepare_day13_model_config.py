"""Validate a manually confirmed decision and write a pinned smoke configuration.

No network access, model downloads or revision discovery. Supply full commit SHAs.
"""

import argparse
import json
from pathlib import Path
import re


def validate_decision(decision):
    if decision.get("status") != "confirmed":
        raise ValueError("Obtain owner confirmation, then set status to confirmed")
    for name in ("confirmed_by", "confirmed_at", "confirmation_evidence"):
        if not isinstance(decision.get(name), str) or not decision[name].strip():
            raise ValueError(f"Missing {name}")
    if decision.get("checkpoint") not in {"google-t5/t5-small", "google/flan-t5-small"}:
        raise ValueError("Select the confirmed T5-small or Flan-T5-small checkpoint")
    for name in ("revision", "tokenizer_revision"):
        if not isinstance(decision.get(name), str) or not re.fullmatch(r"[a-fA-F0-9]{40}", decision[name]):
            raise ValueError(f"{name} must be a full 40-character commit SHA, not main")
    if decision.get("tokenizer") != decision["checkpoint"] or decision["tokenizer_revision"] != decision["revision"]:
        raise ValueError("This backend loads tokenizer and model from the same repository and revision")
    settings = decision["proposed_settings"]
    for name in ("max_input_length", "max_new_tokens"):
        if type(settings.get(name)) is not int or settings[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(settings.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    if settings.get("do_sample") is not False or type(settings.get("num_beams")) is not int or settings["num_beams"] != 1:
        raise ValueError("Current backend uses do_sample=false and num_beams=1")
    if not isinstance(settings.get("device"), str) or not settings["device"]:
        raise ValueError("device must be a nonempty string")
    return {
        "checkpoint": decision["checkpoint"], "revision": decision["revision"],
        **{key: settings[key] for key in ("device", "seed", "max_input_length", "max_new_tokens")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("configs/day13_model_frozen.json"))
    args = parser.parse_args()
    config = validate_decision(json.loads(args.decision.read_text(encoding="utf-8-sig")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    print(f"Pinned config created: {args.output}; revisions supplied by user, not remotely verified.")


if __name__ == "__main__":
    main()

"""Standard-library-only, offline verification of portable pretrained model folders."""
import hashlib
import json
from pathlib import Path
import re

REPOS = {"t5-base": "google-t5/t5-base", "flan-t5-base": "google/flan-t5-base"}
REQUIRED = {"config.json", "model.safetensors", "tokenizer.json", "spiece.model"}
ALLOWED = REQUIRED | {"tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "generation_config.json"}


def hashes(path):
    path = Path(path)
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha256.update(chunk)
            blob.update(chunk)
    return {"size": size, "sha256": sha256.hexdigest(), "blob_id": blob.hexdigest()}


def validate_plan(plan, key):
    if plan["repo_id"] != REPOS[key] or not re.fullmatch(r"[0-9a-f]{40}", plan["revision"]):
        raise ValueError("wrong model repository or revision")
    if not REQUIRED <= set(plan["files"]) <= ALLOWED:
        raise ValueError("missing required model files or unexpected filenames")
    for name, info in plan["files"].items():
        if not isinstance(info["size"], int) or info["size"] <= 0:
            raise ValueError(f"invalid file size: {name}")
        expected = info.get("sha256") or info.get("blob_id")
        if not expected or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", expected):
            raise ValueError(f"missing upstream checksum: {name}")


def matches(actual, expected):
    return actual["size"] == expected["size"] and all(
        actual[key] == expected[key] for key in ("sha256", "blob_id") if expected.get(key))


def verify_model(directory, key):
    directory = Path(directory)
    manifest_path = directory / "model_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}. Finish downloading and upload the entire model folder.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_plan(manifest, key)
    for name, expected in manifest["files"].items():
        if not matches(hashes(directory / name), expected):
            raise ValueError(f"model file damaged/incomplete: {directory / name}")
    return manifest

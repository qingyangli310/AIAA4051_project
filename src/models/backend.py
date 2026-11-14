"""Lazy seq2seq inference; importing this module does not load/download models."""

from dataclasses import dataclass
import time


@dataclass(frozen=True)
class ModelConfig:
    checkpoint: str = "google-t5/t5-small"
    revision: str = "main"
    device: str = "cpu"
    seed: int = 42
    max_input_length: int = 512
    max_new_tokens: int = 64

    def __post_init__(self):
        if self.max_input_length <= 0 or self.max_new_tokens <= 0:
            raise ValueError("token limits must be positive")


def output_length_metadata(count, last_token, eos_token_id, limit):
    eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
    return {
        "output_tokens_including_eos": count,
        "ended_with_eos": count > 0 and last_token in eos_ids,
        # EOS can be forced at the limit: these are independent observations.
        "reached_output_limit": count >= limit,
    }


class Seq2SeqBackend:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.load_seconds = None

    def load(self):
        if self.model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

        set_seed(self.config.seed)
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.checkpoint, revision=self.config.revision
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.checkpoint, revision=self.config.revision
        )
        model.to(self.config.device)
        model.eval()
        self.tokenizer, self.model = tokenizer, model
        self.load_seconds = time.perf_counter() - started

    def generate_text(self, prompt):
        import torch

        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_length = inputs["input_ids"].shape[1]
        if input_length > self.config.max_input_length:
            raise ValueError(
                f"Input has {input_length} tokens; limit is {self.config.max_input_length}. "
                "Input was not truncated."
            )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        using_cuda = self.model.device.type == "cuda"
        if using_cuda:
            torch.cuda.synchronize(self.model.device)
            torch.cuda.reset_peak_memory_stats(self.model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                num_beams=1,
                early_stopping=False,
            )
        if using_cuda:
            torch.cuda.synchronize(self.model.device)
        elapsed = time.perf_counter() - started
        ids = outputs[0, 1:]  # Exclude the decoder start token.
        metadata = {
            "checkpoint": self.config.checkpoint,
            "requested_revision": self.config.revision,
            "resolved_model_revision": getattr(self.model.config, "_commit_hash", None),
            "device": str(self.model.device),
            "seed": self.config.seed,
            "input_tokens": input_length,
            "input_truncated": False,
            "max_input_length": self.config.max_input_length,
            "decoding": {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": False, "num_beams": 1, "early_stopping": False,
                "forced_eos_token_id": self.model.generation_config.forced_eos_token_id,
            },
            "load_seconds": self.load_seconds,
            "inference_seconds": elapsed,
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated(self.model.device) if using_cuda else None
            ),
            **output_length_metadata(
                ids.numel(), ids[-1].item() if ids.numel() else None,
                self.model.generation_config.eos_token_id, self.config.max_new_tokens,
            ),
        }
        return {
            "raw_output": self.tokenizer.decode(outputs[0], skip_special_tokens=True),
            "metadata": metadata,
        }

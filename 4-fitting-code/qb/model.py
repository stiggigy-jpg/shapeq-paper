"""CAUSAL-LM harness: walking linear layers, collecting inputs, perplexity, safe weight swap."""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import Linear

_threads_set = False


def load_model(model_id: str, dtype=torch.float32):
    """Model in eval mode, no gradients, CPU only. Returns (model, config dict)."""
    global _threads_set
    if not _threads_set:
        # QB_TORCH_THREADS — explicit intra-op thread cap for runs within a RAM/CPU budget
        # (set_num_threads overrides OMP_NUM_THREADS at runtime, the env var alone does not help)
        torch.set_num_threads(int(os.environ.get("QB_TORCH_THREADS")
                                 or max(1, (os.cpu_count() or 4) - 2)))
        _threads_set = True
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    config = {
        "vocab": int(model.config.vocab_size),
        "layers": int(model.config.num_hidden_layers),
        "hidden": int(model.config.hidden_size),
    }
    return model, config


def linear_layers(model, include_lm_head: bool = False) -> list[str]:
    names = [n for n, m in model.named_modules() if isinstance(m, Linear)]
    return names if include_lm_head else [n for n in names if "lm_head" not in n]


def collect_inputs(model, names: list[str], chunks: np.ndarray,
                   max_rows: int = 2048) -> dict[str, np.ndarray]:
    """Inputs of the given layers (rows, C) in a single model run; rows <= max_rows per layer."""
    modules = dict(model.named_modules())
    store: dict[str, list[torch.Tensor]] = {n: [] for n in names}

    def hook_for(name: str):
        def hook(module, inputs):
            got = sum(t.shape[0] for t in store[name])
            if got >= max_rows:
                return
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            store[name].append(x[: max_rows - got])
        return hook

    handles = [modules[n].register_forward_pre_hook(hook_for(n)) for n in names]
    try:
        with torch.no_grad():
            for i in range(chunks.shape[0]):
                model(torch.as_tensor(chunks[i:i + 1]))
                if all(sum(t.shape[0] for t in store[n]) >= max_rows for n in names):
                    break
    finally:
        for h in handles:
            h.remove()

    return {n: torch.cat(store[n], dim=0)[:max_rows].numpy() for n in names}


def perplexity(model, chunks: np.ndarray, batch: int = 4) -> float:
    """Sliding perplexity over chunks: NLL of the next token, without skipping batches."""
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, chunks.shape[0], batch):
            ids = torch.as_tensor(chunks[i:i + batch])
            logits = model(ids).logits[:, :-1, :]
            target = ids[:, 1:]
            # Row-wise CE: a float copy of the logits for the whole batch (batch×512×vocab 152k)
            # cost ~2.5 GB and hit a global OOM on Qwen; row-wise gives the same sum, ~0.6 GB
            nll = sum(float(torch.nn.functional.cross_entropy(
                logits[j].float(), target[j], reduction="sum"))
                for j in range(logits.shape[0]))
            total_nll += nll
            total_tokens += int(target.numel())
    return float(np.exp(total_nll / max(total_tokens, 1)))


@contextmanager
def replace_weight(model, name: str):
    """Temporarily replace a layer's weights; on exit restore the original bytes into the original tensor."""
    module = dict(model.named_modules())[name]
    original = module.weight.data.clone()
    try:
        yield module
    finally:
        module.weight.data = original

"""Residual-stream access for HF causal LMs: locate blocks, steer with
forward hooks, capture activations at per-row positions.

Site convention: site 0 is the embedding output; site k >= 1 is the output of
transformer block k-1. This matches HuggingFace ``output_hidden_states``:
``hidden_states[k]`` is exactly site k, so there are ``n_layers + 1`` sites.
"""

import torch


def resolve_blocks(model):
    """Return (embed_module, [block, ...]) for supported HF causal LMs.

    Handles the common layouts: ``model.model.layers`` (OLMo/Llama-family),
    ``model.gpt_neox.layers`` (Pythia), ``model.transformer.h`` (GPT-2).
    """
    for path in ("model", "gpt_neox", "transformer"):
        inner = getattr(model, path, None)
        if inner is None:
            continue
        layers = getattr(inner, "layers", None)
        if layers is None:
            layers = getattr(inner, "h", None)
        embed = (getattr(inner, "embed_tokens", None)
                 or getattr(inner, "embed_in", None)
                 or getattr(inner, "wte", None))
        if layers is not None and embed is not None:
            return embed, list(layers)
    raise ValueError(f"cannot locate transformer blocks on {type(model).__name__}")


def n_sites(model):
    """Number of residual-stream sites (embedding output + one per block)."""
    _, blocks = resolve_blocks(model)
    return len(blocks) + 1


def _site_modules(model):
    embed, blocks = resolve_blocks(model)
    modules = {0: embed}
    for i, block in enumerate(blocks):
        modules[i + 1] = block
    return modules


def _replace_hidden(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden,) + tuple(output[1:])
    return new_hidden


class SteeringHooks:
    """Add ``scale * vec`` to the residual stream at each configured site, at
    every position of every row. Use as a context manager; hooks are removed
    on exit so the model is restored exactly.

    site_vectors: ``{site: (vec [D], scale float)}``. ``vec`` is used as given
    (normalize upstream if a unit direction is intended).
    """

    def __init__(self, model, site_vectors):
        self.model = model
        self.site_vectors = dict(site_vectors)
        self._handles = []

    def __enter__(self):
        modules = _site_modules(self.model)
        unknown = sorted(set(self.site_vectors) - set(modules))
        if unknown:
            raise ValueError(f"sites {unknown} out of range (max {max(modules)})")

        def make_hook(vec, scale):
            def hook(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                shift = (scale * vec).to(device=hidden.device, dtype=hidden.dtype)
                return _replace_hidden(output, hidden + shift)
            return hook

        for site, (vec, scale) in self.site_vectors.items():
            self._handles.append(
                modules[site].register_forward_hook(make_hook(vec, scale)))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False


@torch.no_grad()
def capture_at_offsets(model, input_ids, attention_mask, offsets, device="cpu"):
    """One forward pass; gather the residual stream at per-row offsets.

    ``offsets`` are relative to each row's content length: offset -1 is the
    last content token, -2 the one before it. Returns
    ``{offset: float32 CPU tensor [n_sites, B, D]}``.
    """
    out = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        use_cache=False,
        output_hidden_states=True,
    )
    lengths = attention_mask.to(out.hidden_states[0].device).sum(dim=1)
    batch_idx = torch.arange(input_ids.shape[0], device=out.hidden_states[0].device)
    result = {}
    for off in offsets:
        pos = lengths + off
        if (pos < 0).any():
            raise ValueError(f"offset {off} out of range for a row of length "
                             f"{int(lengths.min())}")
        rows = [h[batch_idx, pos].float().cpu() for h in out.hidden_states]
        result[off] = torch.stack(rows)
    return result


def project(acts, direction):
    """Scalar projection of ``acts [..., D]`` onto the unit vector of ``direction``."""
    d = direction.float()
    d = d / d.norm()
    return acts.float() @ d

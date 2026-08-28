"""
patch.py — causal concept swaps in the residual stream.

The scoreboard test. Reading a concept out of the workspace only shows
correlation. To show the workspace is *used*, we edit it: remove the source
concept direction and add the target with matched magnitude, then check whether
the model's downstream computation follows the edit (Soccer->Rugby, spider->ant,
France->China). This mirrors the paper's intervention experiments.

Implementation: at a set of layers/positions we add alpha * (u_tgt - u_src) to
the residual stream, where u_* are unit unembedding directions and alpha scales
to the local activation norm. This is a difference-vector steering edit (cf.
ActAdd / steering vectors) grounded in the lens's own concept directions.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from contextlib import contextmanager


def _layer_modules(model):
    """Return the ordered list of transformer block modules, arch-agnostic."""
    for path in [("model", "layers"), ("transformer", "h"),
                 ("gpt_neox", "layers"), ("model", "decoder", "layers")]:
        obj = model
        ok = True
        for a in path:
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok:
            return list(obj)
    raise RuntimeError("could not locate transformer blocks")


@contextmanager
def concept_swap(lens, src: str, tgt: str, layers, positions, alpha: float = 0.3):
    """
    Context manager that installs forward hooks nudging the residual stream from
    the source concept toward the target at the given block layers/positions.
    `alpha` is the perturbation size as a FRACTION of the local activation norm
    (so it is comparable across layers whose norms differ by orders of
    magnitude). alpha~0.2-0.5 is a meaningful-but-non-destructive edit; alpha>=1
    tends to wipe out the distribution. `layers` are 0-indexed block indices.
    """
    blocks = _layer_modules(lens.model)
    u_src = lens.token_direction(src).to(lens.device)
    u_tgt = lens.token_direction(tgt).to(lens.device)
    delta_unit = (u_tgt - u_src)
    delta_unit = delta_unit / (delta_unit.norm() + 1e-8)
    handles = []

    def mk_hook():
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            for p in positions:
                if p < h.shape[1]:
                    scale = h[0, p].norm()
                    h[0, p] = h[0, p] + alpha * scale * delta_unit.to(h.dtype)
            return out
        return hook

    try:
        for L in layers:
            handles.append(blocks[L].register_forward_hook(mk_hook()))
        yield
    finally:
        for hnd in handles:
            hnd.remove()


@contextmanager
def concept_swap_repe(lens, vecs_by_layer: dict, positions, scale: float = 1.0):
    """
    The CORRECT causal edit: steer along input-side concept directions
    (difference-of-means activation vectors from lens.concept_vector), which the
    network actually reads. `vecs_by_layer` maps block-index -> raw delta vector.
    Unlike concept_swap (unembedding rows), this reliably flips downstream answers.
    """
    blocks = _layer_modules(lens.model)
    handles = []

    def mk(L):
        v = vecs_by_layer[L]
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            for p in positions:
                if p < h.shape[1]:
                    h[0, p] = h[0, p] + scale * v.to(h.dtype)
            return out
        return hook

    try:
        for L in vecs_by_layer:
            handles.append(blocks[L].register_forward_hook(mk(L)))
        yield
    finally:
        for hnd in handles:
            hnd.remove()


@torch.no_grad()
def swap_effect_repe(lens, prompt: str, pos_prompts: list[str], neg_prompts: list[str],
                     candidates: list[str], layers=None, positions=None,
                     scale: float = 1.0) -> dict:
    """
    Measure the causal effect of swapping neg-concept -> pos-concept in the
    workspace using input-side difference-of-means directions. pos/neg_prompts
    are small contrast sets eliciting each concept (e.g. China vs France).
    """
    enc = lens.tok(prompt, return_tensors="pt").to(lens.device)
    seq = enc["input_ids"].shape[1]
    positions = positions if positions is not None else list(range(seq))
    if layers is None:
        layers = list(range(lens.n_layers // 4, lens.n_layers // 4 + lens.n_layers // 3))
    vecs = {L: lens.concept_vector(pos_prompts, neg_prompts, L) for L in layers}
    base = answer_logprobs(lens, prompt, candidates)
    with concept_swap_repe(lens, vecs, positions, scale=scale):
        patched = answer_logprobs(lens, prompt, candidates)
    return {"baseline": base, "patched": patched, "layers": layers, "scale": scale}


@torch.no_grad()
def answer_logprobs(lens, prompt: str, candidates: list[str]) -> dict[str, float]:
    """Next-token log-prob of each candidate answer word after `prompt`."""
    enc = lens.tok(prompt, return_tensors="pt").to(lens.device)
    logits = lens.model(**enc).logits[0, -1]
    lp = F.log_softmax(logits.float(), dim=-1)
    out = {}
    for c in candidates:
        ids = lens.tok.encode(c, add_special_tokens=False)
        out[c] = lp[ids[0]].item() if ids else float("nan")
    return out


@torch.no_grad()
def swap_effect(lens, prompt: str, src: str, tgt: str, candidates: list[str],
                layers=None, positions=None, alpha: float = 0.3) -> dict:
    """
    Measure the causal effect of swapping src->tgt in the workspace on the
    model's answer. Returns baseline vs patched answer log-probs.
    positions default to the whole prompt; layers default to the middle third.
    """
    enc = lens.tok(prompt, return_tensors="pt").to(lens.device)
    seq = enc["input_ids"].shape[1]
    positions = positions if positions is not None else list(range(seq))
    if layers is None:
        lo, hi = lens.n_layers // 3, 2 * lens.n_layers // 3
        layers = list(range(lo, hi))
    base = answer_logprobs(lens, prompt, candidates)
    with concept_swap(lens, src, tgt, layers, positions, alpha=alpha):
        enc2 = lens.tok(prompt, return_tensors="pt").to(lens.device)
        logits = lens.model(**enc2).logits[0, -1]
        lp = F.log_softmax(logits.float(), dim=-1)
        patched = {}
        for c in candidates:
            ids = lens.tok.encode(c, add_special_tokens=False)
            patched[c] = lp[ids[0]].item() if ids else float("nan")
    return {"baseline": base, "patched": patched, "layers": layers, "alpha": alpha}

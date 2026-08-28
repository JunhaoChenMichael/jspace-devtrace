"""
metrics.py — J-space metrics for comparing checkpoints along a training trajectory.

These are the scalars we track across base -> SFT -> RL(GRPO/DPO) -> multimodal
to make "the workspace evolves during post-training" a quantitative claim rather
than an anecdote. Each metric uses the SAME lens on every checkpoint, so a fixed
lens bias cancels in the contrast.

Metrics
-------
perspective_index(prompt, reaction_words):
    At READING positions (while the model consumes a user message), how much
    workspace mass sits on "reaction/stance" tokens (WARNING, dangerous, unsafe,
    refuse) vs at writing time. Anthropic report base models activate reaction
    words only when WRITING, post-trained models already when READING. This
    metric localizes that shift and should jump at the SFT stage.

reasoning_trace_strength(prompt, hop_word, answer_pair):
    On a multi-hop prompt whose bridge concept never appears in the text
    (e.g. "how many legs does the web-spinning animal have?" -> bridge 'spider'),
    (a) peak workspace probability of the bridge word across layers, and
    (b) its causal weight = how much swapping it flips the answer. Hypothesis:
    RL-for-reasoning increases both.

workspace_capacity(items):
    Ask the model to silently hold k items; count how many are simultaneously
    decodable in the workspace above threshold. A capacity number to compare.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from patch import swap_effect


@torch.no_grad()
def _mass_on(lens, text, words, positions, method="logit"):
    """Total workspace probability on a set of words, averaged over layers/positions."""
    hs, ids = lens.hidden_states(text)
    word_ids = []
    for w in words:
        e = lens.tok.encode(w, add_special_tokens=False)
        if e:
            word_ids.append(e[0])
    word_ids = torch.tensor(word_ids, device=lens.device)
    layers = list(range(1, lens.n_layers + 1))
    vals = []
    for p in positions:
        for L in layers:
            if method == "future" and L in lens.future_lens:
                logits = lens.futurelens(hs[L, p], L)
            else:
                logits = lens.logitlens(hs[L, p])
            probs = F.softmax(logits.float(), dim=-1)
            vals.append(probs[word_ids].sum().item())
    return sum(vals) / max(len(vals), 1)


@torch.no_grad()
def perspective_index(lens, user_message: str, reaction_words: list[str],
                      method="logit") -> dict:
    """
    Reaction-word workspace mass while READING the user message, relative to a
    neutral baseline of reading a benign message. Higher => the model forms its
    own stance during reading (post-trained "perspective").
    """
    enc = lens.tok(user_message, return_tensors="pt").to(lens.device)
    read_positions = list(range(enc["input_ids"].shape[1]))
    reading = _mass_on(lens, user_message, reaction_words, read_positions, method)
    return {"reading_reaction_mass": reading, "n_positions": len(read_positions)}


@torch.no_grad()
def reasoning_trace_strength(lens, prompt: str, hop_word: str,
                             answer_src: str, answer_tgt: str,
                             hop_alt: str, method="logit") -> dict:
    """
    (a) peak workspace prob of the (silent) bridge concept across layers/positions
    (b) causal weight: does swapping hop_word->hop_alt flip answer_src->answer_tgt
    """
    hs, ids = lens.hidden_states(prompt)
    hop_ids = lens.tok.encode(hop_word, add_special_tokens=False)
    hop_id = hop_ids[0] if hop_ids else None
    peak = 0.0
    if hop_id is not None:
        for p in range(len(ids)):
            for L in range(1, lens.n_layers + 1):
                if method == "future" and L in lens.future_lens:
                    logits = lens.futurelens(hs[L, p], L)
                else:
                    logits = lens.logitlens(hs[L, p])
                prob = F.softmax(logits.float(), dim=-1)[hop_id].item()
                peak = max(peak, prob)
    eff = swap_effect(lens, prompt, hop_word, hop_alt, [answer_src, answer_tgt])
    d_src = eff["patched"][answer_src] - eff["baseline"][answer_src]
    d_tgt = eff["patched"][answer_tgt] - eff["baseline"][answer_tgt]
    causal_weight = d_tgt - d_src  # >0 means swap pushed answer src->tgt
    return {"bridge_peak_prob": round(peak, 5),
            "causal_weight": round(causal_weight, 4),
            "detail": eff}


@torch.no_grad()
def workspace_capacity(lens, hold_prompt: str, items: list[str], method="logit",
                       thresh: float = 1e-3) -> dict:
    """Count how many held items are simultaneously decodable in the workspace."""
    hs, ids = lens.hidden_states(hold_prompt)
    last = len(ids) - 1
    decodable = 0
    per_item = {}
    for w in items:
        e = lens.tok.encode(w, add_special_tokens=False)
        if not e:
            per_item[w] = 0.0
            continue
        wid = e[0]
        best = 0.0
        for L in range(1, lens.n_layers + 1):
            if method == "future" and L in lens.future_lens:
                logits = lens.futurelens(hs[L, last], L)
            else:
                logits = lens.logitlens(hs[L, last])
            best = max(best, F.softmax(logits.float(), dim=-1)[wid].item())
        per_item[w] = round(best, 5)
        if best >= thresh:
            decodable += 1
    return {"n_decodable": decodable, "n_items": len(items), "per_item": per_item}

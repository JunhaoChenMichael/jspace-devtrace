"""Test the CORRECT input-side concept direction (difference-of-means) for a
causal France->China swap, vs the failed unembedding-row baseline."""
from jlens import WorkspaceLens, pick_device
from patch import _layer_modules, answer_logprobs
from contextlib import contextmanager
import torch

lens = WorkspaceLens("Qwen/Qwen2.5-0.5B-Instruct")
prompt = "Question: What is the capital of France? Answer: The capital is"
cands = ["Paris", "Beijing"]

pos = ["China", "I visited China last year.", "China is a country in Asia.",
       "The Great Wall is in China.", "Beijing is the capital of China."]
neg = ["France", "I visited France last year.", "France is a country in Europe.",
       "The Eiffel Tower is in France.", "Paris is the capital of France."]

base = answer_logprobs(lens, prompt, cands)
print("baseline:", {k: round(v, 2) for k, v in base.items()},
      " gap(B-P):", round(base["Beijing"] - base["Paris"], 2))

@contextmanager
def steer(vecs_by_layer, positions, scale):
    blocks = _layer_modules(lens.model)
    handles = []
    def mk(L):
        v = vecs_by_layer[L]
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            for p in positions:
                if p < h.shape[1]:
                    h[0, p] = h[0, p] + scale * v.to(h.dtype)
            return o
        return hook
    try:
        for L in vecs_by_layer:
            handles.append(blocks[L].register_forward_hook(mk(L)))
        yield
    finally:
        for hnd in handles: hnd.remove()

ids = lens.tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
subj = [7]  # 'France'
allpos = list(range(len(ids)))

for band in [range(6, 14), range(8, 16)]:
    vecs = {L: lens.concept_vector(pos, neg, L) for L in band}
    for scale in [1.0, 2.0, 4.0]:
        for tag, P in [("subj", subj), ("all", allpos)]:
            with steer(vecs, P, scale):
                lp = answer_logprobs(lens, prompt, cands)
            print(f"band={list(band)[0]}-{list(band)[-1]} scale={scale} pos={tag:4s}  "
                  f"Paris={lp['Paris']:.2f} Beijing={lp['Beijing']:.2f}  "
                  f"gap={lp['Beijing']-lp['Paris']:.2f}"
                  + ("   <-- FLIP" if lp['Beijing'] > lp['Paris'] else ""))

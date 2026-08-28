"""Quick sweep to find a clean, non-destructive causal-swap config on a small model.
Tests France->China on a factual prompt across (alpha, layer-band) settings."""
import sys
from jlens import WorkspaceLens
from patch import concept_swap, answer_logprobs

lens = WorkspaceLens("Qwen/Qwen2.5-0.5B-Instruct")
prompt = "Question: What is the capital of France? Answer: The capital is"
cands = ["Paris", "Beijing"]
# subject 'France' token positions
ids = lens.tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
toks = [lens.tok.decode([i]) for i in ids]
subj_pos = [i for i, t in enumerate(toks) if "France" in t or "France".startswith(t.strip()) and t.strip()]
print("tokens:", toks)
print("subject positions:", subj_pos)

base = answer_logprobs(lens, prompt, cands)
print("baseline:", {k: round(v, 2) for k, v in base.items()})

n = lens.n_layers
bands = {"early": range(2, 8), "mid": range(8, 16), "late": range(14, 22)}
for alpha in [0.05, 0.1, 0.2]:
    for bname, band in bands.items():
        with concept_swap(lens, "France", "China", list(band),
                          subj_pos or list(range(len(ids))), alpha=alpha):
            lp = answer_logprobs(lens, prompt, cands)
        flip = lp["Beijing"] - lp["Paris"]
        base_flip = base["Beijing"] - base["Paris"]
        print(f"alpha={alpha} band={bname:5s}  Paris={lp['Paris']:.2f} Beijing={lp['Beijing']:.2f} "
              f"  (B-P: {base_flip:.2f} -> {flip:.2f})")

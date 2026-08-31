"""Isolate the cause: token set, or computation path?

Computes P(Yes) four ways on identical prompts:
  A probe math + probe tokens   (what the metacognitive track reports as V)
  B probe math + RL tokens
  C RL math    + RL tokens      (what the RL track reports as its admission score)
  D RL math    + probe tokens
If B == C the token set explains everything. If A == B and C == D the difference
is in the computation path instead.
"""
import sys, json
sys.path.insert(0, "src")
import torch, torch.nn.functional as F
from experiments.measure import yes_no_ids
from jlens import WorkspaceLens
from memory_rl.modeling import binary_action_logits, render_admission_prompt, yes_no_token_ids

model, rev, battery_path, n_ep = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
lens = WorkspaceLens(model, device="cuda", dtype=torch.bfloat16,
                     model_revision=rev, tokenizer_revision=rev)
tok = lens.tok
p_yes, p_no = yes_no_ids(lens)
rl_no, rl_yes = yes_no_token_ids(tok)
print("probe tokens yes/no:", p_yes, p_no)
print("rl    tokens yes/no:", rl_yes, rl_no)

def probe_math(prompt, yes_ids, no_ids):
    enc = tok(prompt, return_tensors="pt").to(lens.device)
    with torch.no_grad():
        logits = lens.model(**enc).logits[0, -1].float()
    p = F.softmax(logits, dim=-1)
    py = sum(p[i].item() for i in yes_ids); pn = sum(p[i].item() for i in no_ids)
    return py / (py + pn + 1e-9)

def rl_math(prompt, yes_ids, no_ids):
    with torch.no_grad():
        lg = binary_action_logits(lens.model, tok, [prompt], (no_ids, yes_ids), lens.device, 2048)
    return float(F.softmax(lg, dim=-1)[0, 1])

bat = json.load(open(battery_path))[:n_ep]
rows = []
for ep in bat:
    for it in ep["items"]:
        pr = render_admission_prompt(tok, ep["context"], it["concept"])
        rows.append((
            probe_math(pr, p_yes, p_no),   # A
            probe_math(pr, rl_yes, rl_no), # B
            rl_math(pr, rl_yes, rl_no),    # C
            rl_math(pr, p_yes, p_no),      # D
        ))
import statistics as st
names = ["A probe-math+probe-tok", "B probe-math+rl-tok", "C rl-math+rl-tok", "D rl-math+probe-tok"]
for i, n in enumerate(names):
    v = [r[i] for r in rows]
    print(f"{n:26s} mean={st.fmean(v):.6f} min={min(v):.3g} max={max(v):.3g} yes_rate={sum(x>=.5 for x in v)/len(v):.4f}")
print()
for i, j in ((0,1),(1,2),(0,2),(2,3)):
    d = max(abs(r[i]-r[j]) for r in rows)
    print(f"max|{names[i][0]} - {names[j][0]}| = {d:.6f}")

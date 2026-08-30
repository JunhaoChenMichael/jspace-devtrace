"""Independent V readout: no repo code, plain transformers, both scales.

Reimplements the verbal probe from scratch (own prompt assembly, own batching,
own AUC) so a bug in the repository's row plumbing cannot survive the check.
"""
import json, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

YES = [7414, 9454, 9693, 9834, 14004]
NO = [902, 2152, 2308, 2753, 8996]

def auc(scores, labels):
    p = sorted(zip(scores, labels)); r = [0.0]*len(p); i = 0
    while i < len(p):
        j = i
        while j+1 < len(p) and p[j+1][0] == p[i][0]: j += 1
        a = (i+1+j+1)/2
        for k in range(i, j+1): r[k] = a
        i = j+1
    pos = sum(x for _, x in p); neg = len(p)-pos
    return (sum(rk for rk, (_, x) in zip(r, p) if x) - pos*(pos+1)/2)/(pos*neg)

model_name, revision, battery_path, limit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    model_name, revision=revision, dtype=torch.bfloat16, device_map=None
).to("cuda").eval()

battery = json.load(open(battery_path))
episodes = battery[:limit] if limit else battery
scores, labels, shown = [], [], 0
with torch.no_grad():
    for ep in episodes:
        ctx = ep["context"]
        for cand in ep["items"]:
            q = (f"{ctx}\n\nBased only on the passage above, is the concept "
                 f"\"{cand['concept']}\" one of the most important things to remember in order to "
                 f"answer possible future questions? Answer with a single word: yes or no.")
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt").to("cuda")
            p = torch.softmax(model(**enc).logits[0, -1].float(), dim=-1)
            py = sum(p[i].item() for i in YES); pn = sum(p[i].item() for i in NO)
            scores.append(py/(py+pn+1e-9)); labels.append(int(cand["label"] == "load_bearing"))
            if shown < 5:
                print(f"   V={scores[-1]:.6f} {cand['label']:<14} {cand['concept'][:30]}"); shown += 1
print(f"MODEL {model_name} rev {revision[:12]}")
print(f"EPISODES {len(episodes)} CANDIDATES {len(scores)} POS {sum(labels)}")
print(f"INDEPENDENT_V_AUC {auc(scores, labels):.6f}")

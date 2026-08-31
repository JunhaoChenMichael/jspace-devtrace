"""Dump the actual forward-pass inputs and raw logits from both reporter paths."""
import sys, json
sys.path.insert(0, "src")
import torch, torch.nn.functional as F
from experiments.measure import yes_no_ids
from jlens import WorkspaceLens
from memory_rl.modeling import render_admission_prompt, tokenize_prompts, yes_no_token_ids

model, rev, battery = sys.argv[1], sys.argv[2], sys.argv[3]
lens = WorkspaceLens(model, device="cuda", dtype=torch.bfloat16,
                     model_revision=rev, tokenizer_revision=rev)
tok = lens.tok
p_yes, p_no = yes_no_ids(lens)
ep = json.load(open(battery))[0]
prompt = render_admission_prompt(tok, ep["context"], ep["items"][0]["concept"])

enc_a = tok(prompt, return_tensors="pt").to(lens.device)
enc_b = tokenize_prompts(tok, [prompt], lens.device, 2048)
print("probe input_ids shape:", tuple(enc_a["input_ids"].shape))
print("rl    input_ids shape:", tuple(enc_b["input_ids"].shape))
print("identical ids:", torch.equal(enc_a["input_ids"], enc_b["input_ids"]))
print("probe last 8 ids:", enc_a["input_ids"][0, -8:].tolist())
print("rl    last 8 ids:", enc_b["input_ids"][0, -8:].tolist())
print("probe keys:", sorted(enc_a.keys()), " rl keys:", sorted(enc_b.keys()))

with torch.no_grad():
    la = lens.model(**enc_a).logits[0, -1].float()
    lb = lens.model(**enc_b).logits[:, -1].float()[0]
print("\nmax|logits_a - logits_b| =", float((la - lb).abs().max()))
print("logit at yes ids (probe path):", [round(float(la[i]), 4) for i in p_yes])
print("logit at no  ids (probe path):", [round(float(la[i]), 4) for i in p_no])
pa = F.softmax(la, -1)
print("\nsoftmax-over-vocab  P(yes)=", float(sum(pa[i] for i in p_yes)),
      " P(no)=", float(sum(pa[i] for i in p_no)))
print("probe V =", float(sum(pa[i] for i in p_yes) / (sum(pa[i] for i in p_yes) + sum(pa[i] for i in p_no))))
ly = torch.logsumexp(la[p_yes], 0); ln = torch.logsumexp(la[p_no], 0)
print("logsumexp yes =", float(ly), " no =", float(ln),
      " -> P(Yes) =", float(torch.softmax(torch.stack([ln, ly]), 0)[1]))

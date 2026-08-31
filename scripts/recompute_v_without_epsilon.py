"""Recompute Decoupled V with the ratio the documentation claims.

verbal_salience returns py/(py+pn+1e-9). When py+pn << 1e-9 the epsilon
dominates and the result is proportional to the ABSOLUTE yes probability
rather than the yes-vs-no ratio. This recomputes both quantities on the same
candidates so the reported AUCs can be checked against the intended one.
"""
import json, sys
sys.path.insert(0, "src")
import torch, torch.nn.functional as F
from experiments.measure import yes_no_ids
from jlens import WorkspaceLens
from memory_rl.modeling import render_admission_prompt

model, rev, battery_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
lens = WorkspaceLens(model, device="cuda", dtype=torch.bfloat16,
                     model_revision=rev, tokenizer_revision=rev)
tok = lens.tok
yes_ids, no_ids = yes_no_ids(lens)

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

rows = []
battery = json.load(open(battery_path))
for e, ep in enumerate(battery):
    for c, it in enumerate(ep["items"]):
        prompt = render_admission_prompt(tok, ep["context"], it["concept"])
        enc = tok(prompt, return_tensors="pt").to(lens.device)
        with torch.no_grad():
            logits = lens.model(**enc).logits[0, -1].float()
        p = F.softmax(logits, dim=-1)
        py = float(sum(p[i] for i in yes_ids)); pn = float(sum(p[i] for i in no_ids))
        rows.append({
            "episode": e, "candidate_index": c, "label": it["label"],
            "p_yes_absolute": py, "p_no_absolute": pn,
            "yes_plus_no_mass": py + pn,
            "V_as_reported": py/(py+pn+1e-9),          # what the campaigns used
            "V_true_ratio": float(torch.softmax(torch.stack([
                torch.logsumexp(logits[no_ids], 0),
                torch.logsumexp(logits[yes_ids], 0)]), 0)[1]),
        })
    print(f"  {e+1}/{len(battery)}", flush=True)

lab = [int(r["label"] == "load_bearing") for r in rows]
summary = {
    "model": model, "revision": rev, "battery": battery_path,
    "n_candidates": len(rows),
    "auc_V_as_reported": auc([r["V_as_reported"] for r in rows], lab),
    "auc_V_true_ratio": auc([r["V_true_ratio"] for r in rows], lab),
    "auc_p_yes_absolute": auc([r["p_yes_absolute"] for r in rows], lab),
    "yes_plus_no_mass": {
        "min": min(r["yes_plus_no_mass"] for r in rows),
        "median": sorted(r["yes_plus_no_mass"] for r in rows)[len(rows)//2],
        "max": max(r["yes_plus_no_mass"] for r in rows),
        "fraction_below_1e-9": sum(r["yes_plus_no_mass"] < 1e-9 for r in rows)/len(rows),
    },
    "rows": rows,
}
with open(out_path, "x") as fh:
    json.dump(summary, fh, indent=2, sort_keys=True); fh.write("\n")
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))

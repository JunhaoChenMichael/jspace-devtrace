"""
likelihood_baseline.py — conditional-likelihood admission baseline: score each
candidate by the length-normalized log-probability of its tokens appended to
the context, log p(c|x)/|c|. Distinguishes the workspace readout from a plain
generation-probability proxy: the readout measures decodability of the
concept's first token from intermediate residual states, not the model's
propensity to say the phrase next.
"""
import os, sys, json, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jlens import WorkspaceLens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--battery", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lens = WorkspaceLens(args.model, dtype=torch.bfloat16)
    battery = json.load(open(args.battery))
    print(f"model={args.model} episodes={len(battery)}", flush=True)

    rows = []
    with torch.no_grad():
        for ei, ep in enumerate(battery):
            ctx_ids = lens.tok(ep["context"], return_tensors="pt").input_ids.to(lens.device)
            for it in ep["items"]:
                c_ids = lens.tok(" " + it["concept"], add_special_tokens=False,
                                 return_tensors="pt").input_ids.to(lens.device)
                full = torch.cat([ctx_ids, c_ids], dim=1)
                logits = lens.model(full).logits[0]
                lp = 0.0
                start = ctx_ids.shape[1]
                for j in range(c_ids.shape[1]):
                    lp += F.log_softmax(logits[start - 1 + j].float(), dim=-1)[
                        c_ids[0, j]].item()
                rows.append({"episode": ei, "concept": it["concept"],
                             "label": it["label"],
                             "LL": lp / c_ids.shape[1]})
            if (ei + 1) % 20 == 0:
                print(f"  {ei+1}/{len(battery)}", flush=True)

    from sklearn.metrics import roc_auc_score
    y = [1 if r["label"] == "load_bearing" else 0 for r in rows]
    print(f"LL AUC: {roc_auc_score(y, [r['LL'] for r in rows]):.3f}")
    json.dump(rows, open(args.out, "w"))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

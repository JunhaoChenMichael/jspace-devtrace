"""
vprobe_robust.py — reviewer-required robustness check: is V's anti-calibration
just one bad prompt? Three paraphrases of the importance probe + their ensemble.
"""
import os, sys, json, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jlens import WorkspaceLens
from measure import yes_no_ids
from experiments.measure import _yes_vs_no

PARAPHRASES = [
    ("P1", "{ctx}\n\nBased only on the passage above, is the concept \"{c}\" one of the "
           "most important things to remember in order to answer possible future "
           "questions? Answer with a single word: yes or no."),
    ("P2", "{ctx}\n\nYou can only keep a few notes about this passage. Should \"{c}\" "
           "be one of them? Answer with a single word: yes or no."),
    ("P3", "{ctx}\n\nRate the concept \"{c}\": will it be useful later for answering "
           "questions about what you just read? Answer with a single word: yes or no."),
]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--battery", default="data/benchmarks/battery_v4_final.json")
    ap.add_argument("--out", default="data/results/results_vrobust_v4_7B.json")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    lens = WorkspaceLens(args.model, dtype=getattr(torch, args.dtype))
    yes_ids, no_ids = yes_no_ids(lens)
    battery = json.load(open(args.battery))
    print(f"model={args.model} episodes={len(battery)}", flush=True)

    rows = []
    for ei, ep in enumerate(battery):
        for it in ep["items"]:
            row = {"episode": ei, "concept": it["concept"], "label": it["label"]}
            for tag, tmpl in PARAPHRASES:
                q = tmpl.format(ctx=ep["context"], c=it["concept"])
                msgs = [{"role": "user", "content": q}]
                prompt = lens.tok.apply_chat_template(msgs, tokenize=False,
                                                      add_generation_prompt=True)
                enc = lens.tok(prompt, return_tensors="pt").to(lens.device)
                logits = lens.model(**enc).logits[0, -1].float()
                row[f"V_{tag}"] = _yes_vs_no(logits, yes_ids, no_ids)
            row["V_ens"] = sum(row[f"V_{t}"] for t, _ in PARAPHRASES) / len(PARAPHRASES)
            rows.append(row)
        if (ei + 1) % 10 == 0:
            print(f"  {ei+1}/{len(battery)}", flush=True)

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"saved {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

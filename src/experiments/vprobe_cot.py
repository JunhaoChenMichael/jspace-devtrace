"""
vprobe_cot.py — deliberative (infer-then-answer) verbal probe: does letting
the reporter think before answering recover silently inferred content? The
Rehearsal result predicts generation-time computation can surface bridges;
this tests whether a chain-of-thought importance probe does.

V_cot per item = 1 if the model's final verdict is yes, else 0 (ties get half
credit in AUC as usual). Greedy decoding.
"""
import os, sys, json, argparse, re
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jlens import WorkspaceLens

PROMPT = ("{ctx}\n\nThink briefly (2-3 sentences) about what the passage "
          "implies beyond what it states, then decide: is the concept "
          "\"{c}\" one of the most important things to remember in order to "
          "answer possible future questions? End with exactly one line: "
          "FINAL: yes or FINAL: no.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--battery", default="data/benchmarks/battery_v4_final.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=120)
    args = ap.parse_args()

    lens = WorkspaceLens(args.model, dtype=torch.bfloat16)
    battery = json.load(open(args.battery))
    print(f"model={args.model} episodes={len(battery)}", flush=True)

    rows = []
    for ei, ep in enumerate(battery):
        for it in ep["items"]:
            q = PROMPT.format(ctx=ep["context"], c=it["concept"])
            msgs = [{"role": "user", "content": q}]
            prompt = lens.tok.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True)
            enc = lens.tok(prompt, return_tensors="pt").to(lens.device)
            with torch.no_grad():
                out = lens.model.generate(
                    **enc, max_new_tokens=args.max_new, do_sample=False,
                    pad_token_id=lens.tok.pad_token_id or lens.tok.eos_token_id)
            txt = lens.tok.decode(out[0, enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
            m = re.search(r"FINAL:\s*(yes|no)", txt, re.I)
            v = None if not m else (1.0 if m.group(1).lower() == "yes" else 0.0)
            if v is None:  # fallback: last yes/no in the text
                toks = re.findall(r"\b(yes|no)\b", txt, re.I)
                v = 1.0 if (toks and toks[-1].lower() == "yes") else 0.0
            rows.append({"episode": ei, "concept": it["concept"],
                         "label": it["label"], "V_cot": v})
        if (ei + 1) % 10 == 0:
            print(f"  {ei+1}/{len(battery)}", flush=True)

    from sklearn.metrics import roc_auc_score
    y = [1 if r["label"] == "load_bearing" else 0 for r in rows]
    s = [r["V_cot"] for r in rows]
    print(f"\nCoT verbal AUC: {roc_auc_score(y, s):.3f} "
          f"(yes-rate {sum(s)/len(s):.3f})")
    json.dump(rows, open(args.out, "w"))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

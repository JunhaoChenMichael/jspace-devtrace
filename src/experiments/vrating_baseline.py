"""
vrating_baseline.py — the canonical Generative-Agents-style 1--10 importance
rating (reviewer-required baseline): expected rating under the digit-token
distribution at the answer position.
"""
import os, sys, json, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jlens import WorkspaceLens

PROMPT = ("{ctx}\n\nOn a scale of 1 to 10, where 1 is purely mundane and 10 is "
          "extremely important, rate how important it is to remember the concept "
          "\"{c}\" in order to answer possible future questions about this "
          "passage. Answer with a single number.")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--battery", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    lens = WorkspaceLens(args.model, dtype=getattr(torch, args.dtype))
    digit_ids = {}
    for d in range(1, 10):
        for v in (str(d), " " + str(d)):
            ids = lens.tok.encode(v, add_special_tokens=False)
            if ids:
                digit_ids.setdefault(d, set()).add(ids[0])
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
            logits = lens.model(**enc).logits[0, -1].float()
            p = F.softmax(logits, dim=-1)
            num, den = 0.0, 0.0
            for d, ids in digit_ids.items():
                pd = sum(p[i].item() for i in ids)
                num += d * pd; den += pd
            rows.append({"episode": ei, "concept": it["concept"],
                         "label": it["label"],
                         "V_rating": num / (den + 1e-9)})
        if (ei + 1) % 10 == 0:
            print(f"  {ei+1}/{len(battery)}", flush=True)

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"saved {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

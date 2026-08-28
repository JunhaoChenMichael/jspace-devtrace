"""
finetune_metacog.py — METACOGNITIVE ALIGNMENT pilot (THEORY.md §4).

Hypothesis: the verbal importance reporter is miscalibrated because no
post-training signal ever tied it to the model's own internal availability.
Fix: distill the workspace readout INTO the verbal channel.

Training data: the exact verbal-importance prompt used by measure.py, with
yes/no targets derived from the model's OWN W_rr ranking (top-K items per
episode -> "yes"). Train batteries: v2f + v1f + v2_g2 (disjoint from eval).
Eval batteries: v4_final + v3d — measure V-AUC before/after with measure.py.

Success criterion (pre-registered): V-AUC on held-out v4/v3 rises toward the
W-AUC level without degrading W_rr itself or full-context QA accuracy.
"""
import os, sys, json, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPT = ("{ctx}\n\nBased only on the passage above, is the concept "
          "\"{c}\" one of the most important things to remember in order to "
          "answer possible future questions? Answer with a single word: yes or no.")


def build_dataset(pairs, top_k=2):
    """pairs: list of (results_file, battery_file). Yields (prompt, target)."""
    data = []
    for rfile, bfile in pairs:
        rows = json.load(open(rfile))
        battery = json.load(open(bfile))
        by_ep = {}
        for r in rows:
            by_ep.setdefault(r["episode"], []).append(r)
        for ei, items in by_ep.items():
            ctx = battery[ei]["context"]
            ranked = sorted(items, key=lambda x: x["W_rr"], reverse=True)
            for rank, it in enumerate(ranked):
                target = "yes" if rank < top_k else "no"
                data.append({"prompt": PROMPT.format(ctx=ctx, c=it["concept"]),
                             "target": target})
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B-Instruct")
    ap.add_argument("--out-dir", default="data/benchmarks/olmo1b-metacog")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--train-pairs", default=None,
                    help="comma list of results_file:battery_file; overrides the OLMo default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="CONTROL: permute yes/no targets across examples (frequency-matched), "
                         "so any held-out gain must come from probe-format de-biasing alone")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable()
    model.train()

    if args.train_pairs:
        pairs = [tuple(p.split(":")) for p in args.train_pairs.split(",")]
    else:
        pairs = [("data/results/results_v2f_olmo1b-rlvr.json", "data/benchmarks/battery_v2_final.json"),
                 ("data/results/results_v1f_olmo1b-rlvr.json", "data/benchmarks/battery_v1_final.json"),
                 ("data/results/results_v2g2_olmo1b-rlvr.json", "data/benchmarks/battery_v2_g2.json")]
    pairs = [(r, b) for r, b in pairs if os.path.exists(r)]
    data = build_dataset(pairs, top_k=args.top_k)
    if args.shuffle_labels:
        import random as _r
        targets = [d["target"] for d in data]
        _r.Random(0).shuffle(targets)
        for d, t in zip(data, targets):
            d["target"] = t
        print("CONTROL: labels shuffled (frequency-matched)", flush=True)
    ys = sum(1 for d in data if d["target"] == "yes")
    print(f"dataset: {len(data)} examples ({ys} yes / {len(data)-ys} no) "
          f"from {len(pairs)} batteries", flush=True)

    def encode(ex):
        msgs = [{"role": "user", "content": ex["prompt"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
        t_ids = tok(ex["target"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        ids = torch.cat([p_ids, t_ids])
        labels = torch.cat([torch.full_like(p_ids, -100), t_ids])
        return ids, labels

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    import random
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    steps_per_epoch = (len(data) + args.batch - 1) // args.batch
    sched = get_cosine_schedule_with_warmup(opt, 20, steps_per_epoch * args.epochs)

    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(data)
        for i in range(0, len(data), args.batch):
            batch = data[i:i + args.batch]
            enc = [encode(ex) for ex in batch]
            maxlen = max(len(e[0]) for e in enc)
            pad = tok.pad_token_id or tok.eos_token_id
            ids = torch.full((len(enc), maxlen), pad, dtype=torch.long)
            lbl = torch.full((len(enc), maxlen), -100, dtype=torch.long)
            att = torch.zeros((len(enc), maxlen), dtype=torch.long)
            for j, (x, y) in enumerate(enc):
                ids[j, :len(x)] = x; lbl[j, :len(y)] = y; att[j, :len(x)] = 1
            out = model(input_ids=ids.to(device), attention_mask=att.to(device),
                        labels=lbl.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                print(f"  epoch {epoch} step {step} loss {out.loss.item():.4f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"saved fine-tuned model -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

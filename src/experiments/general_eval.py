"""
general_eval.py — general-capability check for metacognitive-alignment
checkpoints: MMLU (0-shot, letter loglikelihood), ARC-Easy/Challenge
(length-normalized answer loglikelihood), GSM8K (5-shot, greedy, last-number
match on a fixed 500-item subsample). Streaming datasets: no disk cache.
Identical prompts/order across models so before/after deltas are paired.
"""
import os, re, json, argparse, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"


def load_model(path, adapter=None, dtype=torch.bfloat16):
    """Load a full checkpoint or a base checkpoint plus a LoRA adapter."""
    peft_model = None
    if adapter:
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise RuntimeError("--adapter evaluation requires PEFT") from exc
        config = PeftConfig.from_pretrained(adapter)
        configured_base = getattr(config, "base_model_name_or_path", None)
        if configured_base and str(configured_base).rstrip("/") != str(path).rstrip("/"):
            raise ValueError(
                f"adapter was trained from {configured_base!r}, not requested base {path!r}"
            )
        peft_model = PeftModel
    tok = AutoTokenizer.from_pretrained(adapter or path)
    mdl = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(DEV)
    if adapter:
        mdl = peft_model.from_pretrained(mdl, adapter)
    mdl.eval()
    return tok, mdl


@torch.no_grad()
def choice_logprob(tok, mdl, prompt, completion):
    """Sum logprob of completion tokens given prompt."""
    p_ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    c_ids = tok(completion, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
    full = torch.cat([p_ids, c_ids], dim=1)
    if full.shape[1] > 4096:
        full = full[:, -4096:]
        p_len = full.shape[1] - c_ids.shape[1]
    else:
        p_len = p_ids.shape[1]
    logits = mdl(full).logits[0]
    lp = 0.0
    for j in range(c_ids.shape[1]):
        lp += F.log_softmax(logits[p_len - 1 + j].float(), dim=-1)[c_ids[0, j]].item()
    return lp, c_ids.shape[1]


def eval_mmlu(tok, mdl, limit=None):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
    letters = ["A", "B", "C", "D"]
    n = correct = 0
    for ex in ds:
        q = ex["question"].strip()
        prompt = f"Question: {q}\n"
        for i, ch in enumerate(ex["choices"]):
            prompt += f"{letters[i]}. {ch}\n"
        prompt += "Answer:"
        scores = [choice_logprob(tok, mdl, prompt, f" {L}")[0] for L in letters]
        correct += int(max(range(4), key=lambda i: scores[i]) == ex["answer"])
        n += 1
        if n % 1000 == 0:
            print(f"    mmlu {n}: {correct/n:.4f}", flush=True)
        if limit and n >= limit:
            break
    return correct / n, n


def eval_arc(tok, mdl, config):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", config, split="test", streaming=True)
    n = correct = 0
    for ex in ds:
        prompt = f"Question: {ex['question'].strip()}\nAnswer:"
        best, best_i = -1e30, None
        for i, choice in enumerate(ex["choices"]["text"]):
            lp, ln = choice_logprob(tok, mdl, prompt, " " + choice)
            s = lp / max(ln, 1)
            if s > best:
                best, best_i = s, i
        correct += int(ex["choices"]["label"][best_i] == ex["answerKey"])
        n += 1
    return correct / n, n


GSM_SHOTS = None


def gsm_fewshot():
    global GSM_SHOTS
    if GSM_SHOTS is None:
        from datasets import load_dataset
        tr = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
        shots = []
        for ex in tr:
            shots.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}\n")
            if len(shots) == 5:
                break
        GSM_SHOTS = "\n".join(shots) + "\n"
    return GSM_SHOTS


def last_number(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace(",", ""))
    return m[-1].rstrip(".") if m else None


@torch.no_grad()
def eval_gsm8k(tok, mdl, limit=500):
    from datasets import load_dataset
    ds = list(load_dataset("openai/gsm8k", "main", split="test", streaming=True))
    random.Random(0).shuffle(ds)
    ds = ds[:limit]
    shots = gsm_fewshot()
    n = correct = 0
    for ex in ds:
        prompt = shots + f"Question: {ex['question']}\nAnswer:"
        enc = tok(prompt, return_tensors="pt").to(DEV)
        out = mdl.generate(**enc, max_new_tokens=256, do_sample=False,
                           temperature=None, top_p=None, top_k=None,
                           pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        gen = gen.split("Question:")[0]
        gold = last_number(ex["answer"].split("####")[-1])
        pred = last_number(gen)
        correct += int(pred is not None and pred == gold)
        n += 1
        if n % 100 == 0:
            print(f"    gsm8k {n}: {correct/n:.4f}", flush=True)
    return correct / n, n


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base/full Hugging Face checkpoint")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter directory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmlu-limit", type=int, default=0, help="0 = full test set")
    args = ap.parse_args()

    DEV = args.device
    tok, mdl = load_model(args.model, args.adapter, getattr(torch, args.dtype))
    res = {"model": args.model, "adapter": args.adapter, "tag": args.tag}
    shown = f"{args.model} + {args.adapter}" if args.adapter else args.model
    print(f"== {args.tag} ({shown})", flush=True)
    acc, n = eval_arc(tok, mdl, "ARC-Easy");      res["arc_easy"] = acc;      print(f"  arc_easy {acc:.4f} (n={n})", flush=True)
    acc, n = eval_arc(tok, mdl, "ARC-Challenge"); res["arc_challenge"] = acc; print(f"  arc_challenge {acc:.4f} (n={n})", flush=True)
    acc, n = eval_mmlu(tok, mdl, args.mmlu_limit or None); res["mmlu"] = acc; res["mmlu_n"] = n; print(f"  mmlu {acc:.4f} (n={n})", flush=True)
    acc, n = eval_gsm8k(tok, mdl);                res["gsm8k_500"] = acc;     print(f"  gsm8k {acc:.4f} (n={n})", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

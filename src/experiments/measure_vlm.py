"""
measure_vlm.py — workspace + verbal salience on a VISION-language model (TODO 7).

Reads the residual stream of Qwen2-VL at the END of an (image + context) encoding
and scores each concept by case-variant reciprocal rank under the logit lens —
the same W_rr as the text pilot, but the load-bearing concept can only have
entered through the IMAGE. V = chat yes/no importance probe with the image.
"""
import os, sys, json, argparse
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import yes_no_ids as _yes_no_ids  # reuse variants logic below
from experiments.measure import _yes_vs_no


def concept_token_ids(tok, concept):
    cands = []
    for variant in (" " + concept, " " + concept.capitalize(), concept, concept.capitalize()):
        ids = tok.encode(variant, add_special_tokens=False)
        if ids:
            cands.append(ids[0])
    return list(dict.fromkeys(cands))


def find_final_norm(model):
    import torch.nn as nn
    for path in [("model", "norm"), ("model", "language_model", "norm"),
                 ("language_model", "model", "norm")]:
        obj = model
        ok = True
        for a in path:
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.Module):
            return obj
    return nn.Identity()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--battery", default="data/benchmarks/battery_vlm.json")
    ap.add_argument("--out", default="data/results/results_vlm.json")
    ap.add_argument("--no-image", action="store_true",
                    help="ABLATION: same text, image withheld (bridge should vanish)")
    args = ap.parse_args()

    from transformers import AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "llava" in args.model.lower():
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16).to(device).eval()
        proc = AutoProcessor.from_pretrained(args.model)
    else:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16).to(device).eval()
        proc = AutoProcessor.from_pretrained(args.model, max_pixels=768 * 768)
    tok = proc.tokenizer
    norm = find_final_norm(model)
    unembed = model.get_output_embeddings()
    yes_ids, no_ids = _yes_no_ids(type("L", (), {"tok": tok})())

    battery = json.load(open(args.battery))
    print(f"model={args.model} episodes={len(battery)} image={not args.no_image} "
          f"device={device}", flush=True)

    def encode(ep, extra_text=None, gen_prompt=False):
        content = []
        img = None
        if not args.no_image:
            img = Image.open(ep["image"]).convert("RGB")
            content.append({"type": "image"})
        text = ep["context"] if extra_text is None else extra_text
        content.append({"type": "text", "text": text})
        msgs = [{"role": "user", "content": content}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen_prompt)
        kw = {"text": [prompt], "return_tensors": "pt"}
        if img is not None:
            kw["images"] = [img]
        return proc(**kw).to(device)

    rows = []
    with torch.no_grad():
        for ei, ep in enumerate(battery):
            inputs = encode(ep)
            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states  # tuple [1, seq, d]
            cands = {it["concept"]: concept_token_ids(tok, it["concept"])
                     for it in ep["items"]}
            rr = {c: 0.0 for c in cands}
            for L in range(1, len(hs)):
                h = hs[L][0, -1]
                logits = unembed(norm(h).to(unembed.weight.dtype)).float()
                for c, ids in cands.items():
                    for cid in ids:
                        rank = int((logits > logits[cid]).sum().item()) + 1
                        rr[c] = max(rr[c], 1.0 / rank)
            for it in ep["items"]:
                c = it["concept"]
                # V: yes/no importance probe, image attached
                q = (f"{ep['context']}\n\nBased on everything you saw and read, is the "
                     f"concept \"{c}\" one of the most important things to remember in "
                     f"order to answer possible future questions? Answer with a single "
                     f"word: yes or no.")
                vin = encode(ep, extra_text=q, gen_prompt=True)
                vlogits = model(**vin).logits[0, -1].float()
                rows.append({"episode": ei, "concept": c, "label": it["label"],
                             "W_rr": rr[c], "V": _yes_vs_no(vlogits, yes_ids, no_ids)})
            if (ei + 1) % 5 == 0:
                print(f"  {ei+1}/{len(battery)}", flush=True)

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"saved {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()

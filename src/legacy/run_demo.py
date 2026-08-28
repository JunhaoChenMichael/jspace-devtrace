"""
run_demo.py — end-to-end smoke test of the workspace lens on one small model.

  python run_demo.py --model Qwen/Qwen2.5-0.5B-Instruct

Shows (1) silent-word readouts across layers on a multi-hop prompt, and
(2) a causal concept swap that flips the answer.
"""
import argparse, json
from jlens import WorkspaceLens
from patch import swap_effect, swap_effect_repe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--future", action="store_true", help="also train+use future lens")
    args = ap.parse_args()

    lens = WorkspaceLens(args.model)
    print(f"loaded {args.model}: {lens.n_layers} layers, d={lens.d_model}, "
          f"vocab={lens.vocab}, device={lens.device}\n")

    # (1) silent words on a multi-hop prompt: bridge concept 'spider' should
    # surface internally though it is absent from prompt and answer.
    prompt = "Question: How many legs does the animal that spins a web have? Answer:"
    print(f"PROMPT: {prompt}\n--- silent words per layer (logit lens, last token) ---")
    for r in lens.silent_words(prompt, method="logit", topk=6):
        if r.layer % 2 == 0 or r.layer == lens.n_layers:
            print(f"  L{r.layer:2d}: " + ", ".join(
                f"{t}({p})" for t, p in zip(r.tokens, r.probs)))

    # (2) causal swap in the workspace: France -> China should move the answer
    # Paris -> Beijing. We contrast the WRONG output-side direction (unembedding
    # row: fails) against the CORRECT input-side direction (diff-of-means: flips).
    fact = "Question: What is the capital of France? Answer: The capital is"
    print(f"\nPROMPT: {fact}")
    print("--- causal swap  France -> China  (expect Paris down, Beijing up) ---")

    eff0 = swap_effect(lens, fact, "France", "China", ["Paris", "Beijing"])
    b, p = eff0["baseline"], eff0["patched"]
    print(f"  [baseline]                    Paris={b['Paris']:.2f}  Beijing={b['Beijing']:.2f}"
          f"  gap={b['Beijing']-b['Paris']:.2f}")
    print(f"  [unembedding steer (WRONG)]   Paris={p['Paris']:.2f}  Beijing={p['Beijing']:.2f}"
          f"  gap={p['Beijing']-p['Paris']:.2f}   (no flip: output direction)")

    pos = ["China.", "China is a country in Asia.", "The capital of China is Beijing.",
           "I visited China.", "The Great Wall is in China."]
    neg = ["France.", "France is a country in Europe.", "The capital of France is Paris.",
           "I visited France.", "The Eiffel Tower is in France."]
    eff = swap_effect_repe(lens, fact, pos, neg, ["Paris", "Beijing"], scale=1.0)
    q = eff["patched"]
    flip = "FLIP -> Beijing" if q["Beijing"] > q["Paris"] else "no flip"
    print(f"  [diff-of-means steer (RIGHT)] Paris={q['Paris']:.2f}  Beijing={q['Beijing']:.2f}"
          f"  gap={q['Beijing']-q['Paris']:.2f}   ({flip}: input direction)")

    if args.future:
        corpus = [
            "The capital of France is Paris, a city on the Seine.",
            "Water boils at one hundred degrees Celsius at sea level.",
            "A spider has eight legs and spins a web to catch insects.",
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "The mitochondria is the powerhouse of the cell in biology.",
        ] * 6
        print("\n[training future lens on a tiny corpus...]")
        lens.fit_future_lens(corpus, steps=150, verbose=False)
        print("--- silent words (future lens) ---")
        for r in lens.silent_words(prompt, method="future", topk=6):
            if r.layer % 2 == 0 or r.layer == lens.n_layers:
                print(f"  L{r.layer:2d}: " + ", ".join(
                    f"{t}({p})" for t, p in zip(r.tokens, r.probs)))


if __name__ == "__main__":
    main()

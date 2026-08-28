"""
embed_auc.py — predictive AUC of the embedding baseline (hostile-review gap:
"AUC(W) - AUC(bge-cosine) has never been computed").

Scores every item as cosine(concept, context) under bge-small-en-v1.5 and
reports AUC + paired diff vs W_rr from the corresponding results file.
"""
import os, sys, json, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import auc_ci, auc_diff_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="battery.json:results.json pairs")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.embed_model)
    mdl = AutoModel.from_pretrained(args.embed_model).to(device).eval()

    @torch.no_grad()
    def embed(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        return F.normalize(mdl(**enc).last_hidden_state[:, 0], dim=-1)

    for pair in args.pairs:
        bfile, rfile = pair.split(":")
        battery = json.load(open(bfile))
        rows = json.load(open(rfile))
        ctx_embs = {}
        for ei, ep in enumerate(battery):
            ctx_embs[ei] = embed([ep["context"]])
        E, W, lb = [], [], []
        for r in rows:
            ce = embed([r["concept"]])
            E.append(float((ce @ ctx_embs[r["episode"]].T).item()))
            W.append(r["W_rr"])
            lb.append(r["label"] == "load_bearing")
        a, lo, hi = auc_ci(E, lb)
        aw, lw, hw = auc_ci(W, lb)
        d, dlo, dhi, p = auc_diff_ci(W, E, lb)
        print(f"{os.path.basename(rfile)}:")
        print(f"  embedding AUC = {a:.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"  W_rr      AUC = {aw:.3f} [{lw:.3f}, {hw:.3f}]")
        print(f"  W - embed     = {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]  P(W>E) = {p:.2f}")


if __name__ == "__main__":
    main()

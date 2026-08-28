"""
regrade.py — grading-robustness check (audit item): recompute downstream
accuracies from stored per_episode answers under STRICT whole-word matching
(gold answer must appear as a whole word in the generated answer), and report
cells where the policy ordering changes vs the lenient substring grader.
"""
import sys, json, re, glob, os


def strict_grade(output, gold):
    o = re.sub(r"[^a-z0-9 ]", " ", output.lower())
    g = gold.strip().lower()
    return re.search(rf"\b{re.escape(g)}\b", o) is not None


def main():
    files = sys.argv[1:] or sorted(glob.glob("data/results/downstream_v*f_*.json"))
    for f in files:
        d = json.load(open(f))
        battery = json.load(open(d["battery"]))
        golds = {i: ep["answer"] for i, ep in enumerate(battery)}
        print(f"\n== {os.path.basename(f)}")
        for cond, recs in sorted(d["per_episode"].items()):
            lenient = sum(r["correct"] for r in recs) / len(recs)
            strict = sum(strict_grade(r["answer"], golds[r["episode"]]) for r in recs) / len(recs)
            flag = "  <-- changes" if abs(strict - lenient) > 0.05 else ""
            print(f"  {cond:16s} lenient={lenient:.3f} strict={strict:.3f}{flag}")


if __name__ == "__main__":
    main()

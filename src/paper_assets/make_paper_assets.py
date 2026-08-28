"""
make_paper_assets.py — generate all figures (PDF) and LaTeX tables for the paper.
Okabe-Ito colorblind-safe palette, fixed assignment; shape/linestyle redundancy
for grayscale printing; one axis per panel; direct labels where feasible.
Outputs: figures/*.pdf, paper/tables/*.tex
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle, FancyBboxPatch

ROOT_ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT_, 'src', 'analysis'))
from analyze import auc_ci, auc_diff_ci

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "data", "results")
BENCH = os.path.join(ROOT, "data", "benchmarks")
HERE = RESULTS  # legacy alias: all result files load relative to data/results
FIG = os.path.join(ROOT, "paper", "figures"); os.makedirs(FIG, exist_ok=True)
TAB = os.path.join(ROOT, "paper", "tables"); os.makedirs(TAB, exist_ok=True)

# Muted "Nature" palette, fixed semantics: W=steel blue, V=terracotta rose,
# embed=sage green, O=amber, P=lavender, S=pale blue, G=neutral gray.
# Identity is never color-alone (solid/dashed lines, markers, hatching also encode it).
C = {"W": "#6E8FB2", "V": "#C16E71", "E": "#7DA494", "O": "#EAB67A",
     "G": "#A6A6A6", "P": "#9F8DB8", "S": "#ABC8E5"}
CREF = "#9AA0A6"        # reference lines (chance / no-mem / full-context)
CBASE, CSHAM = "#C6C6C6", "#8E8E8E"   # causal: baseline (light) / sham (mid gray)
CACCENT = "#C16E71"     # emphasis marks (peak box) where no channel uses it


def _muted_diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "mutedRB", ["#C16E71", "#E7CFC9", "#F5F2EF", "#CBD6E4", "#6E8FB2"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load(path):
    for base in (RESULTS, BENCH):
        full = os.path.join(base, os.path.basename(path))
        if os.path.exists(full):
            return json.load(open(full))
    raise FileNotFoundError(path)


def auc_of(path, key):
    rows = load(path)
    lb = [r["label"] == "load_bearing" for r in rows]
    if not all(key in r for r in rows):
        return None
    return auc_ci([r[key] for r in rows], lb)


SIZES = ["0.5B", "1.5B", "3B", "7B"]
XPOS = [0.5, 1.5, 3.0, 7.0]
BATT = [("v1f", "results_v1f_{s}-Instruct.json", "Explicit"),
        ("v2f", "results_v2f_{s}-Instruct.json", "Evoked"),
        ("v4", "results_v4f_{s}-Instruct.json", "Decoupled"),
        ("v3", "results_v3f_{s}-Instruct.json", "Compositional")]
# embed AUC per battery (size-independent; from master_table run)
EMBED = {"v1f": 0.635, "v2f": 0.553, "v4": 0.471, "v3": 0.402}


def fig2_regime_map():
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.9), sharey=True)
    letters = ["(a)", "(b)", "(c)", "(d)"]
    for pi, (ax, (bv, tmpl, title)) in enumerate(zip(axes, BATT)):
        series = {}
        for key in ("W_rr", "V"):
            ys, los, his = [], [], []
            for sz in SIZES:
                r = auc_of(tmpl.format(s=sz), key)
                ys.append(r[0]); los.append(r[0] - r[1]); his.append(r[2] - r[0])
            series[key] = (ys, los, his)
        wy, vy = np.array(series["W_rr"][0]), np.array(series["V"][0])
        ax.fill_between(XPOS, wy, vy, where=wy >= vy, color=C["W"], alpha=0.10,
                        interpolate=True, lw=0)
        ax.fill_between(XPOS, wy, vy, where=vy > wy, color=C["V"], alpha=0.10,
                        interpolate=True, lw=0)
        ax.errorbar(XPOS, wy, yerr=series["W_rr"][1:], color=C["W"], marker="o",
                    ms=5, ls="-", lw=1.7, capsize=2.5, label="Workspace $W$")
        ax.errorbar(XPOS, vy, yerr=series["V"][1:], color=C["V"], marker="s",
                    ms=5, ls="--", lw=1.7, capsize=2.5, label="Verbal $V$")
        ax.axhline(EMBED[bv], color=C["E"], lw=1.3, ls=":", label="Embedding")
        ax.axhline(0.5, color=CREF, lw=0.8)
        if pi == 0:
            ax.text(0.52, 0.492, "chance", fontsize=6.8, color=CREF,
                    ha="left", va="top")
        ax.set_xscale("log"); ax.set_xticks(XPOS)
        ax.set_xticklabels(["0.5B", "1.5B", "3B", "7B"])
        ax.minorticks_off()
        ax.set_title(f"{letters[pi]} {title}", fontsize=9)
        ax.set_xlabel("Model size")
        ax.set_ylim(0.10, 0.82)
        ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    axes[0].set_ylabel("AUC (load-bearing vs. rest)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.10), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_regime_map.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig2 done")


def fig3_downstream():
    d05 = load("downstream_v2f_0.5B-Instruct.json")
    d7 = load("downstream_v4x_7B-Instruct.json")
    sig = {  # workspace-vs-verbal McNemar stars per (panel, k)
        (0, 2): "$p{=}.0015$", (0, 3): "$p{<}10^{-4}$", (1, 3): "$p{=}.0013^{\\dagger}$",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0))
    policies = [("workspace", C["W"], "Workspace"), ("verbal", C["V"], "Verbal"),
                ("embedding", C["E"], "Embedding"), ("oracle", C["G"], "Oracle")]
    for pi, (ax, d, title) in enumerate([
            (axes[0], d05, "(a) Qwen-0.5B, Evoked, 75 episodes"),
            (axes[1], d7, "(b) Qwen-7B, Decoupled, 68 episodes")]):
        ks = [1, 2, 3]
        width = 0.19
        for i, (pol, col, lab) in enumerate(policies):
            vals = [d["per_condition"].get(f"{pol}@{k}", np.nan) for k in ks]
            xs = [k + (i - 1.5) * width for k in ks]
            hatch = "///" if pol == "oracle" else None
            ax.bar(xs, vals, width * 0.9, color=col, label=lab, hatch=hatch,
                   edgecolor="white", linewidth=0.6)
            if pol in ("workspace", "verbal"):
                for x, v in zip(xs, vals):
                    ax.text(x, v + 0.012, f"{v:.2f}"[1:], ha="center",
                            fontsize=6.4, color=col)
        for k in ks:
            tag = sig.get((pi, k))
            if tag:
                y = max(d["per_condition"][f"workspace@{k}"],
                        d["per_condition"][f"verbal@{k}"]) + (0.085 if pi == 0 else 0.16)
                x0, x1 = k - 1.5 * width, k - 0.5 * width
                ax.plot([x0, x0, x1, x1], [y, y + 0.015, y + 0.015, y],
                        color=CREF, lw=0.9)
                ax.text((x0 + x1) / 2, y + 0.022, tag, ha="center", fontsize=6.2)
        ax.axhline(d["refs"]["no_memory"], color=CREF, lw=1, ls=":")
        ax.axhline(d["refs"]["full_context"], color=CREF, lw=1, ls="--")
        ax.text(0.52, d["refs"]["no_memory"] + 0.012, "no-memory", fontsize=6.6,
                ha="left", color=CREF)
        ax.text(0.52, d["refs"]["full_context"] + 0.012, "full-context",
                fontsize=6.6, ha="left", color=CREF)
        ax.set_xticks(ks); ax.set_xticklabels([f"$k{{=}}{k}$" for k in ks])
        ax.set_xlim(0.5, 3.62)
        ax.set_title(title, fontsize=9); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Memory budget")
    axes[0].set_ylabel("QA accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.09), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_downstream.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig3 done")


def fig1_hero():
    # Hero is hand-drawn by the authors; the .tex uses a placeholder via
    # \IfFileExists until figures/fig1_hero.pdf is supplied. Skip auto-gen so it
    # does not overwrite the author's version.
    print("fig1_hero: skipped (author-supplied)")
    return
    fig = plt.figure(figsize=(9.8, 3.0))
    # ---- left: pipeline ----
    axL = fig.add_axes([0.005, 0.04, 0.585, 0.92]); axL.axis("off")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1)

    def box(x, y, w, h, text, fc="#FFFFFF", ec="#555555", fs=8, lw=1.3):
        axL.add_patch(FancyBboxPatch((x, y), w, h,
                      boxstyle="round,pad=0.014,rounding_size=0.02",
                      fc=fc, ec=ec, lw=lw, mutation_aspect=0.6))
        axL.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, col="#555555", lw=1.6, ls="-"):
        axL.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                     linestyle=ls, shrinkA=2, shrinkB=2,
                                     mutation_scale=13))

    box(0.005, 0.32, 0.165, 0.36,
        "Context\n\u201cElena wandered\nthe Plaza\nMayor\u2026\u201d", fs=8)
    arrow(0.17, 0.58, 0.235, 0.72)
    arrow(0.17, 0.42, 0.235, 0.28)
    box(0.235, 0.60, 0.235, 0.30,
        "Workspace state\nJ-lens-style readout\n$W$(madrid) high $\\checkmark$",
        fc="#E8F1F9", ec=C["W"], lw=1.6)
    box(0.235, 0.10, 0.235, 0.30,
        "Verbal report\n\u201cis madrid important?\u201d\n$V$(madrid) low $\\times$",
        fc="#FBECE2", ec=C["V"], lw=1.6)
    axL.text(0.352, 0.955, "silently inferred, never written: \u201cmadrid\u201d",
             ha="center", fontsize=7.2, style="italic", color="#666666")
    arrow(0.47, 0.75, 0.545, 0.56, col=C["W"])
    arrow(0.47, 0.25, 0.545, 0.44, col=C["V"])
    box(0.545, 0.35, 0.155, 0.30, "Budget-$k$\nwrite gate", fs=8.5)
    arrow(0.70, 0.50, 0.755, 0.50)
    box(0.755, 0.32, 0.24, 0.36,
        "Later probe\n\u201cWhat language\nshould Elena use?\u201d\nanswerable only\nvia the bridge", fs=7.5)

    # ---- right: 2x2 regime heatmap (Qwen-3B; diverging, centered at 0.5) ----
    axR = fig.add_axes([0.665, 0.17, 0.30, 0.68])
    # rows: W (top), V (bottom); cols: explicit (v1), inferred (v4)
    data = np.array([[0.533, 0.594], [0.641, 0.254]])
    im = axR.imshow(data, cmap=_muted_diverging(), vmin=0.15, vmax=0.85, aspect="auto")
    for (i, j), val in np.ndenumerate(data):
        axR.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=13,
                 fontweight="bold",
                 color="white" if abs(val - 0.5) > 0.22 else "black")
    axR.set_xticks([0, 1])
    axR.set_xticklabels(["explicit", "inferred\n(silent bridge)"], fontsize=8.5)
    axR.set_yticks([0, 1])
    axR.set_yticklabels(["Workspace\n$W$", "Verbal\n$V$"], fontsize=8.5)
    axR.set_title("AUC by regime \u00d7 channel (Qwen-3B)", fontsize=9)
    axR.add_patch(Rectangle((0.5, 0.5), 1.0, 1.0, fill=False, ec="#111111", lw=2.4))
    axR.text(1, 1.30, "anti-calibrated", ha="center", fontsize=7.6,
             style="italic", color="white")
    axR.tick_params(length=0)
    for spine in axR.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=axR, fraction=0.045, pad=0.03, ticks=[0.25, 0.5, 0.75])
    cb.ax.tick_params(labelsize=7)
    cb.ax.axhline(0.5, color="k", lw=0.8)
    cb.set_label("AUC (0.5 = chance)", fontsize=7)
    fig.savefig(os.path.join(FIG, "fig1_hero.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig1 done")


# ---------------- LaTeX table generation ----------------

def fmt(a, lo=None, hi=None, bold=False):
    s = f"{a:.3f}" if a is not None else "---"
    if lo is not None:
        s += f" [{lo:.3f}, {hi:.3f}]"
    return f"\\textbf{{{s}}}" if bold else s


def write_tex(name, content):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(content)
    print(f"table {name} done")


def cluster_diff_ci(rows, n_boot=2000, seed=0):
    """Episode-cluster bootstrap CI for AUC(W_rr) - AUC(V)."""
    from sklearn.metrics import roc_auc_score
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["episode"], []).append(
            (1 if r["label"] == "load_bearing" else 0, r["W_rr"], r["V"]))
    eps = sorted(by_ep)
    def diff(chosen):
        y, w, v = [], [], []
        for e in chosen:
            for yy, ww, vv in by_ep[e]:
                y.append(yy); w.append(ww); v.append(vv)
        y = np.array(y)
        if not (0 < y.sum() < len(y)):
            return None
        return roc_auc_score(y, w) - roc_auc_score(y, v)
    pt = diff(eps)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        d = diff(rng.choice(eps, len(eps), replace=True))
        if d is not None:
            vals.append(d)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return pt, lo, hi


def tab1_master():
    """Main Table 1: 7B rows, 3 families x 4 batteries: W, V, W-V (episode-cluster CI), M_s."""
    fams = [("Qwen2.5-7B", "results_{v}_7B-Instruct.json"),
            ("Qwen3-8B", "results_{v}_qwen3-8B.json"),
            ("OLMo-2-7B", "results_{v}_olmo7b-Instruct.json"),
            ("Mistral-7B", "results_{v}_mistral7b.json")]
    battmap = {"v1f": "v1f", "v2f": "v2f", "v4": "v4f", "v3": "v3f"}
    names = {"v1f": "Explicit", "v2f": "Evoked", "v4": "Decoupled", "v3": "Compositional"}
    rows_tex = []
    for bv in ["v1f", "v2f", "v4", "v3"]:
        for fam, tmpl in fams:
            path = tmpl.format(v=battmap[bv])
            if not os.path.exists(os.path.join(HERE, path)):
                continue
            rows = load(path)
            lb = [r["label"] == "load_bearing" for r in rows]
            W = [r["W_rr"] for r in rows]; V = [r["V"] for r in rows]
            aw, wl, wh = auc_ci(W, lb)
            av, wl2, wh2 = auc_ci(V, lb)
            d, dlo, dhi = cluster_diff_ci(rows)
            ms = (av - 0.5) / (aw - 0.5) if aw >= 0.55 else float("nan")
            star = "$^{*}$" if (dlo > 0 or dhi < 0) else ""
            mss = f"{ms:.2f}" if not np.isnan(ms) else "n/a"
            rows_tex.append(
                f"{names[bv]} & {fam} & {aw:.3f} & {av:.3f} & "
                f"{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]{star} & {mss} \\\\")
        rows_tex.append("\\addlinespace")
    body = "\n".join(rows_tex)
    write_tex("tab1_master.tex", f"""\\begin{{table}}[t]
\\centering
\\caption{{\\textbf{{Workspace and verbal channels by provenance regime (7--8B, four model families).}} AUC of the workspace
readout ($W$) and the verbal importance report ($V$) at recovering oracle-labeled
load-bearing items, with the paired episode-cluster bootstrap difference (2{{,}}000 resamples; $^{{*}}$:
95\\% CI excludes 0) and machine metacognitive efficiency
$M_s{{=}}\\frac{{\\mathrm{{AUC}}(V)-0.5}}{{\\mathrm{{AUC}}(W)-0.5}}$ (reported only when $W \\ge 0.55$; the denominator is unstable otherwise).
Verbal report wins on explicit content (at chance for both channels in Qwen3-8B), is anti-calibrated on the
construct-valid inferred regimes (Decoupled and Compositional, $M_s<0$ in five
of five reportable cells), and never catches up on these sparse-cue benchmarks (generator dependence: App.~\\ref{{app:replications}}).}}
\\label{{tab:master}}
\\small
\\begin{{tabular}}{{llcccc}}
\\toprule
Regime & Family & $W$ & $V$ & $W-V$ [95\\% CI] & $M_s$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab2_metacog():
    def get(path, key):
        rows = load(path)
        lb = [r["label"] == "load_bearing" for r in rows]
        return auc_ci([r[key] for r in rows], lb)
    rows_tex = []
    groups = [
        ("OLMo-2-1B",
         [("Decoupled", "results_v4f_olmo1b-rlvr.json", "results_v4f_olmo1b-metacog.json"),
          ("Compositional", "results_v3f_olmo1b-rlvr.json", "results_v3f_olmo1b-metacog.json")]),
        ("Qwen2.5-0.5B",
         [("Decoupled", "results_v4f_0.5B-Instruct.json", "results_v4f_qwen05b-metacog.json"),
          ("Compositional", "results_v3f_0.5B-Instruct.json", "results_v3f_qwen05b-metacog.json")]),
    ]
    for gi, (model, benches) in enumerate(groups):
        for bv, base_f, ft_f in benches:
            vb = get(base_f, "V"); va = get(ft_f, "V")
            wb = get(base_f, "W_rr"); wa = get(ft_f, "W_rr")
            rows_tex.append(f"{model} & {bv} & {vb[0]:.3f} [{vb[1]:.3f}, {vb[2]:.3f}] & "
                            f"\\textbf{{{va[0]:.3f}}} [{va[1]:.3f}, {va[2]:.3f}] & "
                            f"{wb[0]:.3f} & {wa[0]:.3f} \\\\")
        if gi == 0:
            rows_tex.append("\\addlinespace")
    body = "\n".join(rows_tex)
    write_tex("tab2_metacog.tex", f"""\\begin{{table}}[t]
\\centering
\\caption{{\\textbf{{Metacognitive alignment}}, two model families: ${{\\sim}}$500
fine-tuning steps on yes/no labels derived from each model's own workspace
ranking, using disjoint benchmarks (Evoked, Explicit, Evoked-G2). The verbal
channel is repaired on fully held-out benchmarks (+0.28--0.39 AUC) while the
workspace channel is untouched, full-context QA is statistically unchanged,
and general capability (MMLU, GSM8K, ARC) is unaffected (Table~\\ref{{tab:geneval}})
(App.~\\ref{{app:metacog}}). For the Qwen-0.5B Compositional row the repaired
reporter converges to its teacher's (chance) level; where generation-time
computation adds signal (OLMo Compositional) the student can exceed the
static teacher (\\S\\ref{{sec:metacog}}).}}
\\label{{tab:metacog}}
\\small
\\begin{{tabular}}{{llcccc}}
\\toprule
 & & \\multicolumn{{2}}{{c}}{{Verbal report $V$ (AUC)}} & \\multicolumn{{2}}{{c}}{{Workspace $W$}} \\\\
\\cmidrule(lr){{3-4}} \\cmidrule(lr){{5-6}}
Model & Held-out benchmark & before & after & before & after \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab3_causal():
    runs = [("Qwen2.5-7B", "0.5", "causal_ind_7b_s0.5.json"), ("Qwen2.5-7B", "1.0", "causal_ind_7b_s1.0.json"),
            ("Qwen2.5-3B", "0.5", "causal_ind_3b_s0.5.json"), ("Qwen2.5-3B", "1.0", "causal_ind_3b_s1.0.json"),
            ("Qwen2.5-0.5B", "1.0", "causal_ind_05b_s1.0.json"),
            ("Qwen3-8B", "0.5", "causal_ind_qwen3-8b_s0.5.json"), ("Qwen3-8B", "1.0", "causal_ind_qwen3-8b_s1.0.json")]
    rows_tex = []
    for size, scale, f in runs:
        d = load(f)
        m = d["mcnemar_real_vs_sham"]
        # exact two-sided Wilcoxon signed-rank on paired real-vs-sham delta log-odds
        from scipy.stats import wilcoxon
        deltas = ([(p_["patched_odds"] - p_["base_odds"]) - (p_["sham_odds"] - p_["base_odds"])
                   for p_ in d["pairs"]]
                  if d.get("pairs") and "patched_odds" in d["pairs"][0] else None)
        if deltas:
            wp = wilcoxon(deltas, alternative="two-sided", method="exact").pvalue
        else:
            wp = d["wilcoxon_dlogodds_p"]
        rows_tex.append(f"{size} & {scale} & {d['flip_rate']:.2f} & {d['sham_flip_rate']:.2f} & "
                        f"$+{m['real_only']}/-{m['sham_only']}$ & {m['p']:.4f} & "
                        f"{wp:.1e} \\\\")
    body = "\n".join(rows_tex)
    write_tex("tab3_causal.tex", f"""\\begin{{table}}[t]
\\centering
\\caption{{\\textbf{{Causal effect of workspace interventions on memory-based answers.}} Steering the residual stream
along the stored bridge's input-side direction (real) vs.\\ an unrelated,
norm-matched direction (sham) while the model answers from memory; Qwen2.5
and Qwen3 checkpoints; 20 disjoint episode pairs; a flip is a sign change of the
stored-vs-target answer log-odds. Exact McNemar on paired real-vs-sham flips; exact two-sided
Wilcoxon signed-rank on paired $\\Delta$log-odds. Specificity (real high, sham
low) holds at gentle steering scales and degrades as strength grows.}}
\\label{{tab:causal}}
\\small
\\begin{{tabular}}{{llccccc}}
\\toprule
Model & Scale & Real flip & Sham flip & McNemar & $p$ & Wilcoxon $p$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab_q3scale():
    """Qwen3 within-family scale series: W-V dissociation by regime x size.
    Values are AUC(W_rr)-AUC(V) with 95% episode-cluster bootstrap CIs
    (cluster_diff_ci over results_{v}_qwen3-{size}.json); starred = CI excludes 0.
    Hardcoded from the measured result files to avoid the multi-cell bootstrap on
    every regen (reproduce with cluster_diff_ci)."""
    # (delta, lo, hi) per (regime, size)
    D = {
        ("Evoked", "0.6B"): (0.105, 0.025, 0.186), ("Evoked", "1.7B"): (0.076, -0.022, 0.175),
        ("Evoked", "4B"): (0.206, 0.126, 0.296), ("Evoked", "8B"): (0.237, 0.157, 0.314),
        ("Decoupled", "0.6B"): (0.035, -0.055, 0.129), ("Decoupled", "1.7B"): (0.235, 0.130, 0.339),
        ("Decoupled", "4B"): (0.113, 0.031, 0.199), ("Decoupled", "8B"): (0.317, 0.238, 0.402),
        ("Compositional", "0.6B"): (-0.003, -0.106, 0.096), ("Compositional", "1.7B"): (0.131, 0.023, 0.234),
        ("Compositional", "4B"): (0.195, 0.099, 0.296), ("Compositional", "8B"): (0.220, 0.146, 0.306),
        ("Explicit (control)", "0.6B"): (-0.021, -0.100, 0.058), ("Explicit (control)", "1.7B"): (-0.137, -0.218, -0.059),
        ("Explicit (control)", "4B"): (0.051, -0.033, 0.136), ("Explicit (control)", "8B"): (0.005, -0.070, 0.077),
    }
    sizes = ["0.6B", "1.7B", "4B", "8B"]
    def cell(reg, sz):
        d, lo, hi = D[(reg, sz)]
        star = "$^{*}$" if (lo > 0 or hi < 0) else ""
        return f"{d:+.3f}{star}"
    rows_tex = []
    for reg in ["Evoked", "Decoupled", "Compositional"]:
        rows_tex.append(f"{reg} & " + " & ".join(cell(reg, s) for s in sizes) + " \\\\")
    rows_tex.append("\\addlinespace")
    rows_tex.append("Explicit (control) & " + " & ".join(cell("Explicit (control)", s) for s in sizes) + " \\\\")
    body = "\n".join(rows_tex)
    write_tex("tab_q3scale.tex", f"""\\begin{{table}}[t]
\\centering
\\caption{{\\textbf{{The dissociation scales with model capability (Qwen3 family).}}
Workspace$-$verbal gap $\\Delta{{=}}\\mathrm{{AUC}}(W){{-}}\\mathrm{{AUC}}(V)$ by
provenance regime and model size (episode-cluster bootstrap; $^{{*}}$: 95\\% CI
excludes zero; per-cell $W$, $V$ in App.~\\ref{{app:master}}). On
silently inferred content the gap is significantly positive at both capable
scales (4B and 8B, every inferred regime) and peaks at $+0.317$ (Decoupled,
8B), on par with the strongest dissociation in any family
(Table~\\ref{{tab:master}}); it emerges by 1.7B. On explicit content the gap is null or favors the
verbal channel (significantly so at 1.7B): the anti-calibration is specific to
inferred content and strengthens as the model gets larger---bigger models
carry more in the workspace yet report it no better.}}
\\label{{tab:q3scale}}
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
Regime & Qwen3-0.6B & Qwen3-1.7B & Qwen3-4B & Qwen3-8B \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab4_family():
    cells = {
        ("Spotlight $W$ (static)", "v2f"): 0.633, ("Spotlight $W$ (static)", "v4"): 0.641,
        ("Spotlight $W$ (static)", "v3"): 0.514,
        ("Broadcast breadth (static)", "v2f"): 0.637, ("Broadcast breadth (static)", "v4"): 0.557,
        ("Broadcast breadth (static)", "v3"): 0.486,
        ("Rehearsal $W_{rep}$ (dynamic)", "v2f"): 0.684, ("Rehearsal $W_{rep}$ (dynamic)", "v4"): 0.578,
        ("Rehearsal $W_{rep}$ (dynamic)", "v3"): 0.553,
        ("Leverage $W_{J}$ (dynamic)", "v2f"): 0.506, ("Leverage $W_{J}$ (dynamic)", "v4"): 0.550,
        ("Leverage $W_{J}$ (dynamic)", "v3"): 0.639,
    }
    members = ["Spotlight $W$ (static)", "Broadcast breadth (static)",
               "Rehearsal $W_{rep}$ (dynamic)", "Leverage $W_{J}$ (dynamic)"]
    best = {bv: max(members, key=lambda m: cells[(m, bv)]) for bv in ["v2f", "v4", "v3"]}
    rows_tex = []
    for m in members:
        vals = []
        for bv in ["v2f", "v4", "v3"]:
            v = f"{cells[(m, bv)]:.3f}"
            vals.append(f"\\textbf{{{v}}}" if best[bv] == m else v)
        rows_tex.append(f"{m} & " + " & ".join(vals) + " \\\\")
    body = "\n".join(rows_tex)
    write_tex("tab4_family.tex", f"""\\begin{{table}}[t]
\\centering
\\caption{{\\textbf{{Availability readouts by provenance regime}} (Qwen-7B AUC; full grids
incl.\\ CIs in App.~\\ref{{app:family}}). Decode-based members win on evoked content;
the utility-gradient member is the only family member whose paired gain over
the static readout is significant on Compositional
($W_J{{-}}W_{{rr}}$: $+0.125$ [$+0.010$, $+0.244$]; head-to-head vs.\ the
conditional-likelihood baseline it does not separate, App.~\\ref{{app:vrobust}});
replay's Compositional ceiling is budget-independent. Bold: column best.}}
\\label{{tab:family}}
\\small
\\begin{{tabular}}{{lccc}}
\\toprule
Member & Evoked & Decoupled & Compositional \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


# ---------------- appendix tables ----------------

def appA1_full_grid():
    models = [
        ("Qwen2.5-0.5B", "results_{v}_0.5B-base.json"), ("Qwen2.5-0.5B-I", "results_{v}_0.5B-Instruct.json"),
        ("Qwen2.5-1.5B", "results_{v}_1.5B-base.json"), ("Qwen2.5-1.5B-I", "results_{v}_1.5B-Instruct.json"),
        ("Qwen2.5-3B", "results_{v}_3B-base.json"), ("Qwen2.5-3B-I", "results_{v}_3B-Instruct.json"),
        ("Qwen2.5-7B", "results_{v}_7B-base.json"), ("Qwen2.5-7B-I", "results_{v}_7B-Instruct.json"),
        ("Qwen3-0.6B", "results_{v}_qwen3-0.6B.json"), ("Qwen3-1.7B", "results_{v}_qwen3-1.7B.json"),
        ("Qwen3-4B", "results_{v}_qwen3-4B.json"), ("Qwen3-8B", "results_{v}_qwen3-8B.json"),
        ("GPT-2", "results_{v}_gpt2-base.json"),
        ("OLMo-2-1B (RLVR)", "results_{v}_olmo1b-rlvr.json"),
        ("OLMo-2-7B", "results_{v}_olmo7b-base.json"), ("OLMo-2-7B-I", "results_{v}_olmo7b-Instruct.json"),
        ("Mistral-7B-I", "results_{v}_mistral7b.json"),
    ]
    batts = [("v1f", "Explicit"), ("v2f", "Evoked"), ("v4f", "Decoupled"), ("v3f", "Compositional")]
    lines = []
    for mn, tmpl in models:
        cells = [mn]
        for bv, _ in batts:
            path = tmpl.format(v=bv)
            full = os.path.join(HERE, path)
            if not os.path.exists(full):
                cells.append("---"); continue
            rows = load(path)
            lb = [r["label"] == "load_bearing" for r in rows]
            aw, _, _ = auc_ci([r["W_rr"] for r in rows], lb)
            s = f"{aw:.3f}"
            prefer_raw = "-base.json" in tmpl
            if prefer_raw and all(r.get("V_raw") is not None for r in rows):
                av, _, _ = auc_ci([r["V_raw"] for r in rows], lb)
                s += f" / ({av:.3f})"
            elif all("V" in r for r in rows):
                av, _, _ = auc_ci([r["V"] for r in rows], lb)
                s += f" / {av:.3f}"
            elif all(r.get("V_raw") is not None for r in rows):
                av, _, _ = auc_ci([r["V_raw"] for r in rows], lb)
                s += f" / ({av:.3f})"
            cells.append(s)
        lines.append(" & ".join(cells) + " \\\\")
    body = "\n".join(lines)
    write_tex("appA1_full_grid.tex", f"""\\begin{{table}}[h]
\\centering
\\caption{{Full $W$ / $V$ AUC grid, all measured checkpoints $\\times$ final
batteries. Parenthesized $V$ values are the template-free probe $V_{{raw}}$ (base
models). ``---'': not measured (see App.~\\ref{{app:repro}} for the run manifest).}}
\\label{{tab:appA1}}
\\scriptsize
\\begin{{tabular}}{{lcccc}}
\\toprule
Checkpoint & Explicit & Evoked & Decoupled & Compositional \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def app_downstream_tables():
    """A2: compact wide format (one row per battery x size; policies as columns,
    k=1/2/3 folded into each cell). A4: McNemar in wide format via longtable."""
    settings = [
        ("Evoked (24t)", "downstream_v2f_{s}-Instruct.json", ["0.5B", "1.5B", "3B", "7B"], "24"),
        ("Decoupled (24t)", "downstream_v4f_{s}-Instruct.json", ["0.5B", "1.5B", "3B", "7B"], "24"),
        ("Decoupled (64t)", "downstream_v4x_{s}-Instruct.json", ["3B", "7B"], "64"),
        ("Decoupled-L (64t)", "downstream_v4xl_{s}-Instruct.json", ["0.5B", "1.5B", "3B", "7B"], "64"),
    ]
    pol_order = ["workspace", "verbal", "embedding", "recency", "random", "oracle"]
    lines = []
    mc = {}
    for tagname, tmpl, sizes, ntok in settings:
        for sz in sizes:
            path = os.path.join(HERE, tmpl.format(s=sz))
            if not os.path.exists(path):
                continue
            d = json.load(open(path))
            cells = [tagname, sz]
            for pol in pol_order:
                vals = [d["per_condition"].get(f"{pol}@{k}") for k in (1, 2, 3)]
                if all(v is None for v in vals):
                    cells.append("---")
                else:
                    cells.append("/".join(f"{v:.2f}"[1:] if v < 1 else "1.0"
                                          for v in vals))
            cells.append(f"{d['refs']['no_memory']:.2f}"[1:])
            cells.append(f"{d['refs']['full_context']:.2f}"[1:])
            lines.append(" & ".join(cells) + " \\\\")
            for key, m in sorted(d.get("mcnemar", {}).items()):
                if "workspace_vs" not in key or "no_memory" in key:
                    continue
                rival = key.replace("workspace_vs_", "").split("@")[0]
                k = key.split("@")[1]
                wonly = m.get("workspace_only"); ronly = m.get(f"{rival}_only")
                pstr = f"{m['p']:.3f}" if m["p"] >= 1e-3 else "$<\\!10^{-3}$"
                mc.setdefault((tagname, sz, rival), {})[k] = f"$+{wonly}/-{ronly}$ ({pstr})"
        lines.append("\\addlinespace")
    write_tex("appA2_downstream.tex", """\\begin{table}[h]
\\centering
\\caption{Budget-limited recall QA, all runs. Each cell shows accuracy at
$k{=}1/2/3$ (leading zeros omitted); generation budget
as marked per row; random = 3-seed mean.}
\\label{tab:appA2}
\\scriptsize
\\setlength{\\tabcolsep}{3.5pt}
\\begin{tabular}{llcccccccc}
\\toprule
Benchmark & Size & workspace & verbal & embedding & recency & random & oracle & no-mem & full-ctx \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")
    mlines = []
    prev = None
    for (tagname, sz, rival), kd in mc.items():
        b = tagname if (tagname, sz) != prev else ""
        s_ = sz if (tagname, sz) != prev else ""
        prev = (tagname, sz)
        mlines.append(f"{b} & {s_} & {rival} & " +
                      " & ".join(kd.get(str(k), "---") for k in (1, 2, 3)) + " \\\\")
    write_tex("appA4_mcnemar.tex", """{\\scriptsize
\\setlength{\\tabcolsep}{4pt}
\\begin{longtable}{lllccc}
\\caption{Exact McNemar tests, workspace vs.\\ each rival policy on identical
episodes; cells show discordant counts $+a/-b$ (episodes only workspace / only
rival correct) with exact $p$ in parentheses.}
\\label{tab:appA4} \\\\
\\toprule
Benchmark & Size & Rival & $k{=}1$ & $k{=}2$ & $k{=}3$ \\\\
\\midrule
\\endfirsthead
\\toprule
Benchmark & Size & Rival & $k{=}1$ & $k{=}2$ & $k{=}3$ \\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
""" + "\n".join(mlines) + """
\\end{longtable}}
""")


def app_olmo_stages():
    lines = []
    for bv, nm in [("v2f", "Evoked"), ("v3f", "Compositional")]:
        for stage in ["base", "sft", "dpo", "rlvr"]:
            path = f"results_{bv}_olmo1b-{stage}.json"
            rows = load(path)
            lb = [r["label"] == "load_bearing" for r in rows]
            aw, lo, hi = auc_ci([r["W_rr"] for r in rows], lb)
            cells = [nm if stage == "base" else "", stage.upper(), f"{aw:.3f} [{lo:.3f}, {hi:.3f}]"]
            for key in ["V", "V_raw"]:
                if all(key in r for r in rows):
                    av, _, _ = auc_ci([r[key] for r in rows], lb)
                    cells.append(f"{av:.3f}")
                else:
                    cells.append("---")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    write_tex("appA5_olmo.tex", """\\begin{table}[h]
\\centering
\\caption{OLMo-2-0425-1B post-training stages. The workspace channel never improves
(base is highest on Evoked); the verbal channel is miscalibrated already in the raw-probed
base model and becomes more confidently wrong through the chat template.}
\\label{tab:appA5}
\\small
\\begin{tabular}{llccc}
\\toprule
Benchmark & Stage & $W$ [95\\% CI] & $V$ (chat) & $V_{raw}$ \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")


def app_futurelens():
    lines = []
    for bv, base_tmpl in [("Compositional", "results_v3f_{s}-Instruct.json"), ("Evoked", "results_v2f_{s}-Instruct.json")]:
        for s in SIZES:
            fl_p = f"results_fl_{ {'Compositional':'v3','Evoked':'v2f'}[bv] }_Qwen2.5-{s}-Instruct.json"
            if not os.path.exists(os.path.join(HERE, fl_p)):
                continue
            fl = load(fl_p)
            base = {(r["episode"], r["concept"]): r["W_rr"] for r in load(base_tmpl.format(s=s))}
            lb = [r["label"] == "load_bearing" for r in fl]
            F = [r["W_fl"] for r in fl]
            W = [base[(r["episode"], r["concept"])] for r in fl]
            af, flo, fhi = auc_ci(F, lb)
            aw, _, _ = auc_ci(W, lb)
            d, dlo, dhi, p = auc_diff_ci(F, W, lb)
            star = "$^{*}$" if (dlo > 0 or dhi < 0) else ""
            lines.append(f"{bv} & {s} & {af:.3f} [{flo:.3f}, {fhi:.3f}] & {aw:.3f} & "
                         f"{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]{star} \\\\")
    write_tex("appA6_futurelens.tex", """\\begin{table}[h]
\\centering
\\caption{Trained tuned-future-lens negative control: initialized at the model's own
unembedding, trained on disjoint Explicit-benchmark contexts to place mass on a 12-token future
window. It never rescues Compositional-benchmark bridges and is significantly \\emph{worse}
than the logit lens on evoked bridges at 7B; the static-state boundary is not a
readout artifact.}
\\label{tab:appA6}
\\small
\\begin{tabular}{llccc}
\\toprule
Benchmark & Size & FutureLens AUC & Logit-lens $W$ & FL$-W$ [95\\% CI] \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")


def app_family_grids():
    lines = []
    dispb = {"v2f": "Evoked", "v4": "Decoupled", "v3": "Compositional"}
    for bv in ["v2f", "v4", "v3"]:
        for m in ["0.5B-Instruct", "1.5B-Instruct", "3B-Instruct", "7B-Instruct", "olmo7b-Instruct"]:
            rep_p = f"results_rep_{bv}_{m}.json"
            pul_p = f"results_pul_{bv}_{m}.json"
            if not os.path.exists(os.path.join(HERE, rep_p)):
                continue
            rep = load(rep_p); pul = load(pul_p)
            lb = [r["label"] == "load_bearing" for r in rep]
            lbp = [r["label"] == "load_bearing" for r in pul]
            ae, _, _ = auc_ci([r["W_rep_emit"] for r in rep], lb)
            ad, _, _ = auc_ci([r["W_rep_dec"] for r in rep], lb)
            ap_, plo, phi = auc_ci([r["W_pul"] for r in pul], lbp)
            lines.append(f"{dispb[bv]} & {m.replace('-Instruct','')} & {ae:.3f} & {ad:.3f} & "
                         f"{ap_:.3f} [{plo:.3f}, {phi:.3f}] \\\\")
        lines.append("\\addlinespace")
    ig_lines = []
    for bv in ["v2f", "v4", "v3"]:
        for m in ["3B-Instruct", "7B-Instruct"]:
            p = f"results_ig_{bv}_{m}.json"
            if not os.path.exists(os.path.join(HERE, p)):
                continue
            rows = load(p)
            lb = [r["label"] == "load_bearing" for r in rows]
            cells = [dispb.get(bv, bv), m.replace("-Instruct", "")]
            for k in ["W_ig", "breadth", "persistence", "sharpness"]:
                a, _, _ = auc_ci([r[k] for r in rows], lb)
                cells.append(f"{a:.3f}")
            ig_lines.append(" & ".join(cells) + " \\\\")
    write_tex("appA7_family.tex", """\\begin{table}[h]
\\centering
\\caption{Availability-family grids, dynamic members: Rehearsal (emission /
decoded reactivation, $K{=}8$ rollouts $\\times$ 40 tokens) and Leverage
(utility-gradient) across models and batteries.}
\\label{tab:appA7}
\\small
\\begin{tabular}{llccc}
\\toprule
Benchmark & Model & $W_{rep}^{emit}$ & $W_{rep}^{dec}$ & $W_J$ [95\\% CI] \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")
    write_tex("appA7b_ignition.tex", """\\begin{table}[h]
\\centering
\\caption{Availability-family grids, static Broadcast (ignition) member and its
components over the (layer $\\times$ position) decodability grid.}
\\label{tab:appA7b}
\\small
\\begin{tabular}{llcccc}
\\toprule
Benchmark & Model & $W_{ig}$ & breadth & persistence & sharpness \\\\
\\midrule
""" + "\n".join(ig_lines) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")


def app_multimodal():
    lines = []
    for model, stem in [("Qwen2-VL-2B", ""), ("LLaVA-1.5-7B", "_llava")]:
        for name, f in [("Round 1 (GPT-written text)", "results_vlm"),
                        ("Round 2 (neutral templates)", "results_vlm2"),
                        ("Round 3 (within-class cities)", "results_vlm3")]:
            for cond, suff in [("with image", f"{stem}.json"),
                               ("no image", f"{stem}_noimg.json")]:
                # round-1 files predate the numbering scheme: results_vlm{,_llava}{,_noimg}
                path = f"{f}{suff}" if f != "results_vlm" or stem else f"results_vlm{suff}"
                rows = load(path)
                lb = [r["label"] == "load_bearing" for r in rows]
                aw, wl, wh = auc_ci([r["W_rr"] for r in rows], lb)
                av, vl, vh = auc_ci([r["V"] for r in rows], lb)
                first = f"{model}, {name}" if cond == "with image" else ""
                lines.append(f"{first} & {cond} & "
                             f"{aw:.3f} [{wl:.3f}, {wh:.3f}] & {av:.3f} [{vl:.3f}, {vh:.3f}] \\\\")
            lines.append("\\addlinespace")
    write_tex("appA9_multimodal.tex", """\\begin{table}[h]
\\centering
\\caption{Multimodal boundary, two vision-language families (46 landmark
episodes). Ask-time verbal probing reads the pictured identity nearly
perfectly in both models. The word-level workspace trace of the image is
family-dependent: absent in Qwen2-VL-2B (within-class contrast $\\approx$
no-image ablation) but present in LLaVA-1.5-7B (paired image$-$ablation
$\\Delta$AUC $+0.261$ $[+0.162, +0.359]$ on neutral templates and $+0.131$
$[+0.049, +0.221]$ within-class).}
\\label{tab:appA9}
\\small
\\begin{tabular}{llcc}
\\toprule
Model, benchmark variant & Condition & $W$ [95\\% CI] & $V$ [95\\% CI] \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")


def app_replications():
    lines = []
    for name, f in [("Evoked-G2 (gpt-4o), Qwen-0.5B-I", "results_v2g2_0.5B-Instruct.json"),
                    ("Evoked-G2 (gpt-4o), Qwen-3B-I", "results_v2g2_3B-Instruct.json"),
                    ("Evoked-G2 (gpt-4o), Qwen-7B-I", "results_v2g2_7B-Instruct.json"),
                    ("Evoked-G2 (gpt-4o), OLMo-1B (RLVR)", "results_v2g2_olmo1b-rlvr.json"),
                    ("Decoupled, Mistral-7B-I", "results_v4f_mistral7b.json"),
                    ("Compositional, Mistral-7B-I", "results_v3f_mistral7b.json"),
                    ("Explicit, Mistral-7B-I", "results_v1f_mistral7b.json"),
                    ("Explicit, OLMo-2-7B-I", "results_v1f_olmo7b-instruct.json"),
                    ("Decoupled-G3 (gpt-5.6), Qwen-0.5B-I", "results_v4g56_0.5B-Instruct.json"),
                    ("Decoupled-G3 (gpt-5.6), Qwen-7B-I", "results_v4g56_7B-Instruct.json")]:
        if not os.path.exists(os.path.join(HERE, f)):
            continue
        rows = load(f)
        lb = [r["label"] == "load_bearing" for r in rows]
        W = [r["W_rr"] for r in rows]; V = [r["V"] for r in rows]
        aw, _, _ = auc_ci(W, lb); av, _, _ = auc_ci(V, lb)
        d, dlo, dhi, p = auc_diff_ci(W, V, lb)
        star = "$^{*}$" if (dlo > 0 or dhi < 0) else ""
        lines.append(f"{name} & {aw:.3f} & {av:.3f} & {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]{star} \\\\")
    write_tex("appA10_replications.tex", """\\begin{table}[h]
\\centering
\\caption{Generator- and family-independence replications. The $W{-}V$ dissociation
survives an independent benchmark generator (gpt-4o; note the absolute $W$ level
is generator-dependent: gpt-4o's bridges are weakly evoked) and a third model
family (Mistral-7B). The Explicit rows complete the regime map across
families: on stated content the sign flips and the verbal report beats the
workspace readout in every family tested. The Decoupled-G3 rows (a third
generator, gpt-5.6, whose episodes carry several convergent cues per bridge)
show the two channels' stability profiles: the workspace readout stays in a
narrow band across all three generators while the verbal report swings from
anti-calibrated to well-calibrated with generator and scale.}
\\label{tab:appA10}
\\small
\\begin{tabular}{lccc}
\\toprule
Setting & $W$ & $V$ & $W-V$ [95\\% CI] \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")


def app_vrobust_fusion():
    if os.path.exists(os.path.join(HERE, "results_vrobust_v4_7B.json")):
        rows = load("results_vrobust_v4_7B.json")
        base = {(r["episode"], r["concept"]): r["W_rr"]
                for r in load("results_v4f_7B-Instruct.json")}
        W = [base[(r["episode"], r["concept"])] for r in rows]
        lb = [r["label"] == "load_bearing" for r in rows]
        lines = []
        for k, nm in [("V_P1", "P1 (original)"), ("V_P2", "P2 (keep-notes)"),
                      ("V_P3", "P3 (useful-later)"), ("V_ens", "3-prompt ensemble")]:
            a, lo, hi = auc_ci([r[k] for r in rows], lb)
            d, dlo, dhi, p = auc_diff_ci(W, [r[k] for r in rows], lb)
            lines.append(f"{nm} & {a:.3f} [{lo:.3f}, {hi:.3f}] & "
                         f"$+{d:.3f}$ [$+{dlo:.3f}$, $+{dhi:.3f}$] \\\\")
        lines.append("\\addlinespace")
        for f, nm in [("results_vrating_v4_7B.json", "1--10 rating, Decoupled 7B"),
                      ("results_vrating_v4xl_7B.json", "1--10 rating, Decoupled-L 7B"),
                      ("results_vrating_v2_05B.json", "1--10 rating, Evoked 0.5B")]:
            if not os.path.exists(os.path.join(HERE, f)):
                continue
            rr = load(f)
            lbr = [r["label"] == "load_bearing" for r in rr]
            a, lo, hi = auc_ci([r["V_rating"] for r in rr], lbr)
            lines.append(f"{nm} & {a:.3f} [{lo:.3f}, {hi:.3f}] & --- \\\\")
        write_tex("appA11_vrobust.tex", """\\begin{table}[h]
\\centering
\\caption{Verbal-probe paraphrase robustness and the canonical 1--10 rating
baseline (Qwen; $W$ reference at 7B on the Decoupled benchmark: 0.641).
Rating-gated selection runs use their own generation budgets, so compare
within-run columns only.}
\\label{tab:appA11}
\\small
\\begin{tabular}{lcc}
\\toprule
Importance probe & $V$ AUC [95\\% CI] & paired $W-V$ [95\\% CI] \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")
    fus = []
    for name, f in [("Qwen-0.5B, Evoked (24 tok)", "downstream_fusion_v2f_05b.json"),
                    ("Qwen-7B, Decoupled (64 tok)", "downstream_fusion_v4x_7b.json")]:
        if not os.path.exists(os.path.join(HERE, f)):
            continue
        d = load(f)
        for pol in ["workspace", "verbal", "fusion"]:
            vals = [d["per_condition"].get(f"{pol}@{k}") for k in (2, 3)]
            if all(v is None for v in vals):
                continue
            fus.append(f"{name} & {pol} & " +
                       " & ".join(f"{v:.3f}" if v is not None else "---" for v in vals) + " \\\\")
    if fus:
        write_tex("appA12_fusion.tex", """\\begin{table}[h]
\\centering
\\caption{Surface-feature fusion baseline (rank-average of recency, frequency, and
embedding relevance, an admission-control proxy without internals;
generation budgets as marked per row, so compare within-run columns).}
\\label{tab:appA12}
\\small
\\begin{tabular}{llcc}
\\toprule
Setting & Policy & $k{=}2$ & $k{=}3$ \\\\
\\midrule
""" + "\n".join(fus) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")


if __name__ == "__main__":
    fig1_hero(); fig2_regime_map(); fig3_downstream()
    tab1_master(); tab2_metacog(); tab3_causal(); tab4_family(); tab_q3scale()
    appA1_full_grid(); app_downstream_tables(); app_olmo_stages()
    app_futurelens(); app_family_grids(); app_multimodal(); app_replications()
    app_vrobust_fusion()
    print("ALL ASSETS DONE")


def fig4_development_repair():
    """Main-text: (a) OLMo-2 stage trajectory; (b) metacognitive alignment dumbbell."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 2.7),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    stages = ["base", "sft", "dpo", "rlvr"]
    xlab = ["Base", "SFT", "DPO", "RLVR"]
    for bv, nm, ls in [("v2f", "Evoked", "-"), ("v3f", "Compositional", "--")]:
        ws, vs = [], []
        for st in stages:
            rows = load(f"results_{bv}_olmo1b-{st}.json")
            lb = [r["label"] == "load_bearing" for r in rows]
            ws.append(auc_ci([r["W_rr"] for r in rows], lb)[0])
            key = "V" if all("V" in r for r in rows) else "V_raw"
            vs.append(auc_ci([r[key] for r in rows], lb)[0])
        axA.plot(range(4), ws, marker="o", ms=5, color=C["W"], ls=ls,
                 lw=1.7, label=f"$W$, {nm}")
        axA.plot(range(4), vs, marker="s", ms=5, color=C["V"], ls=ls,
                 lw=1.7, label=f"$V$, {nm}")
    axA.axhline(0.5, color=CREF, lw=0.8)
    axA.set_xticks(range(4)); axA.set_xticklabels(xlab)
    axA.set_ylim(0.15, 0.75); axA.set_ylabel("AUC")
    axA.set_title("(a) OLMo-2-1B post-training stages", fontsize=9)
    axA.legend(fontsize=6.8, ncol=2, frameon=False, loc="lower left")
    # (b) dumbbell: metacog alignment before/after on held-out benchmarks
    items = [("Decoupled", "results_v4f_olmo1b-rlvr.json", "results_v4f_olmo1b-metacog.json"),
             ("Compositional", "results_v3f_olmo1b-rlvr.json", "results_v3f_olmo1b-metacog.json")]
    ypos = [1.0, 0.0]
    for y, (nm, bf, ff) in zip(ypos, items):
        for key, col, dy in [("V", C["V"], 0.16), ("W_rr", C["W"], -0.16)]:
            b = load(bf); a = load(ff)
            lb_b = [r["label"] == "load_bearing" for r in b]
            lb_a = [r["label"] == "load_bearing" for r in a]
            vb = auc_ci([r[key] for r in b], lb_b)[0]
            va = auc_ci([r[key] for r in a], lb_a)[0]
            axB.plot([vb, va], [y + dy, y + dy], color=col, lw=2.2, alpha=0.45,
                     zorder=1)
            axB.scatter([vb], [y + dy], color="white", edgecolor=col, s=42,
                        zorder=3, lw=1.6)
            axB.scatter([va], [y + dy], color=col, s=46, zorder=3)
            if key == "V":
                axB.annotate("", xy=(va - 0.012, y + dy), xytext=(vb + 0.012, y + dy),
                             arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4))
            axB.text(max(vb, va) + 0.025, y + dy, f"{vb:.2f}$\\rightarrow${va:.2f}",
                     va="center", fontsize=7, color=col)
    axB.axvline(0.5, color=CREF, lw=0.8)
    axB.set_yticks([1.0, 0.0])
    axB.set_yticklabels(["Decoupled\n(held-out)", "Compositional\n(held-out)"], fontsize=8)
    axB.set_xlim(0.15, 0.95); axB.set_ylim(-0.55, 1.55)
    axB.set_xlabel("AUC")
    axB.set_title("(b) Metacognitive alignment: ${\\sim}$500 steps", fontsize=9)
    from matplotlib.lines import Line2D
    axB.legend(handles=[
        Line2D([], [], color=C["V"], lw=2.2, label="Verbal $V$"),
        Line2D([], [], color=C["W"], lw=2.2, label="Workspace $W$"),
        Line2D([], [], marker="o", mfc="white", mec="#666", ls="", label="before"),
        Line2D([], [], marker="o", color="#666", ls="", label="after")],
        fontsize=6.8, ncol=2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_development_repair.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig4 done")


def figA_causal():
    runs = [("3B\nscale 0.5", "causal_ind_3b_s0.5.json"),
            ("3B\nscale 1.0", "causal_ind_3b_s1.0.json"),
            ("0.5B\nscale 1.0", "causal_ind_05b_s1.0.json")]
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    xs = np.arange(len(runs))
    real = [load(f)["flip_rate"] for _, f in runs]
    sham = [load(f)["sham_flip_rate"] for _, f in runs]
    ps = [load(f)["mcnemar_real_vs_sham"]["p"] for _, f in runs]
    ax.bar(xs - 0.18, real, 0.34, color=C["W"], label="Bridge direction (real)")
    ax.bar(xs + 0.18, sham, 0.34, color=C["G"], hatch="///",
           edgecolor="white", label="Unrelated direction (sham)")
    for x, r, s_, p in zip(xs, real, sham, ps):
        ax.text(x - 0.18, r + 0.02, f"{r:.2f}", ha="center", fontsize=7.5, color=C["W"])
        ax.text(x + 0.18, s_ + 0.02, f"{s_:.2f}", ha="center", fontsize=7.5, color=CREF)
        ax.text(x, max(r, s_) + 0.11, f"$p{{=}}{p:.4f}$", ha="center", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels([n for n, _ in runs], fontsize=8)
    ax.set_ylabel("Answer flip rate"); ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figA_causal.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("figA_causal done")


def figA_family_heatmap():
    members = ["Spotlight $W$\n(static)", "Broadcast breadth\n(static)",
               "Rehearsal $W_{rep}$\n(dynamic)", "Leverage $W_J$\n(dynamic)"]
    data = np.array([[0.633, 0.641, 0.514],
                     [0.637, 0.557, 0.486],
                     [0.684, 0.578, 0.553],
                     [0.506, 0.550, 0.639]])
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    im = ax.imshow(data, cmap=_muted_diverging(), vmin=0.30, vmax=0.70, aspect="auto")
    for (i, j), v in np.ndenumerate(data):
        best = data[:, j].max() == v
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=9.5, fontweight="bold" if best else "normal",
                color="white" if abs(v - 0.5) > 0.12 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Evoked", "Decoupled", "Compositional"], fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(members, fontsize=7.6)
    ax.set_title("Availability-family AUC (Qwen-7B); bold: column best", fontsize=8.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[0.35, 0.5, 0.65])
    cb.ax.tick_params(labelsize=7); cb.ax.axhline(0.5, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figA_family.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("figA_family done")


def figA_multimodal():
    rounds = [("Round 1\n(GPT text)", "results_vlm"), ("Round 2\n(neutral)", "results_vlm2"),
              ("Round 3\n(within-class)", "results_vlm3")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.5), sharey=True)
    for ax, key, title in [(axes[0], "W_rr", "(a) Workspace $W$"),
                           (axes[1], "V", "(b) Verbal $V$ (ask-time)")]:
        xs = np.arange(len(rounds))
        for dx, suff, col, lab in [(-0.17, ".json", C["W"], "with image"),
                                   (0.17, "_noimg.json", C["G"], "no image")]:
            vals, los, his = [], [], []
            for _, f in rounds:
                rows = load(f + suff)
                lb = [r["label"] == "load_bearing" for r in rows]
                a, lo, hi = auc_ci([r[key] for r in rows], lb)
                vals.append(a); los.append(a - lo); his.append(hi - a)
            ax.bar(xs + dx, vals, 0.32, color=col, label=lab,
                   yerr=[los, his], capsize=2, error_kw=dict(lw=0.8))
        ax.axhline(0.5, color=CREF, lw=0.8)
        ax.set_xticks(xs); ax.set_xticklabels([n for n, _ in rounds], fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("AUC")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figA_multimodal.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("figA_multimodal done")


def figA_vrobust():
    rows = load("results_vrobust_v4_7B.json")
    lb = [r["label"] == "load_bearing" for r in rows]
    names = ["P1 (original)", "P2 (keep-notes)", "P3 (useful-later)", "Ensemble",
             "1--10 rating", "1--10 rating (L)"]
    vals, los, his = [], [], []
    for k in ["V_P1", "V_P2", "V_P3", "V_ens"]:
        a, lo, hi = auc_ci([r[k] for r in rows], lb)
        vals.append(a); los.append(a - lo); his.append(hi - a)
    for f in ["results_vrating_v4_7B.json", "results_vrating_v4xl_7B.json"]:
        rr = load(f)
        lbr = [r["label"] == "load_bearing" for r in rr]
        a, lo, hi = auc_ci([r["V_rating"] for r in rr], lbr)
        vals.append(a); los.append(a - lo); his.append(hi - a)
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    ys = np.arange(len(names))[::-1]
    ax.barh(ys, vals, 0.6, xerr=[los, his], color=C["V"], capsize=2,
            error_kw=dict(lw=0.8))
    ax.axvline(0.5, color=CREF, lw=0.8)
    ax.axvline(0.641, color=C["W"], lw=1.4, ls="--")
    ax.text(0.641, len(names) - 0.15, " $W$ (7B)", color=C["W"], fontsize=7.5,
            ha="left", va="top")
    ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 0.85); ax.set_xlabel("AUC on Decoupled benchmark")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figA_vrobust.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("figA_vrobust done")


def fig5_causal():
    """Causal steering as log-odds distributions (box+strip), no line charts:
    (a) baseline->sham->real at 7B crossing the decision boundary; (b) effect
    size real vs sham across sizes."""
    import numpy as _np
    CREAL = C["W"]
    rng = _np.random.default_rng(0)

    def load(size, scale="1.0"):
        return json.load(open(os.path.join(RESULTS, f"causal_ind_{size}_s{scale}.json")))

    def strip_box(ax, x, vals, col, w=0.42):
        vals = _np.asarray(vals)
        ax.boxplot([vals], positions=[x], widths=w, patch_artist=True, showfliers=False,
                   zorder=2, medianprops=dict(color="black", lw=1.4),
                   boxprops=dict(facecolor=col, alpha=.32, edgecolor=col, lw=1.2),
                   whiskerprops=dict(color=col, lw=1.1), capprops=dict(color=col, lw=1.1))
        ax.scatter(rng.normal(x, 0.055, len(vals)), vals, s=17, color=col, alpha=.85,
                   edgecolors="white", linewidths=.4, zorder=3)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.7),
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    d = load("7b"); p = d["pairs"]
    base = _np.array([x["base_odds"] for x in p]); sham = _np.array([x["sham_odds"] for x in p])
    real = _np.array([x["patched_odds"] for x in p])
    lo, hi = -42, 26
    axA.axhspan(0, hi, color=CREAL, alpha=.05, zorder=0)
    axA.axhspan(lo, 0, color="#000000", alpha=.035, zorder=0)
    axA.text(2.44, hi - 2, "answer\nlikely", fontsize=8, color=CREAL, style="italic", va="top", ha="right")
    axA.text(2.44, lo + 2, "answer\nunlikely", fontsize=8, color="#888", style="italic", va="bottom", ha="right")
    for x, vals, col in [(0, base, CBASE), (1, sham, CSHAM), (2, real, CREAL)]:
        strip_box(axA, x, vals, col)
        axA.annotate(f"{(vals > 0).mean() * 100:.0f}% above 0", (x, vals.max() + 2.5),
                     ha="center", va="bottom", fontsize=8.5, color=col, fontweight="bold")
    axA.axhline(0, color="black", lw=1.2, ls="--", alpha=.8, zorder=4)
    axA.set_ylim(lo, hi); axA.set_xlim(-0.55, 2.5)
    axA.set_xticks([0, 1, 2]); axA.set_xticklabels(["baseline", "+ sham\nsteering", "+ real\nsteering"])
    axA.set_ylabel("log-odds of the memory answer")
    axA.set_title("(a) Real steering pushes the answer across the\ndecision boundary; sham mostly does not (7B)", fontsize=10)

    xt, xl, pos = [], [], 0
    for sk, sl in [("05b", "0.5B"), ("3b", "3B"), ("7b", "7B")]:
        d = load(sk); p = d["pairs"]
        dr = _np.array([x["patched_odds"] - x["base_odds"] for x in p])
        dsh = _np.array([x["sham_odds"] - x["base_odds"] for x in p])
        strip_box(axB, pos, dsh, CSHAM); strip_box(axB, pos + 0.9, dr, CREAL)
        xt.append(pos + 0.45); xl.append(sl); pos += 2.2
    axB.axhline(0, color="black", lw=.8, ls=":")
    axB.set_xticks(xt); axB.set_xticklabels(xl); axB.set_xlabel("model size")
    axB.set_ylabel("$\\Delta$ log-odds (steered $-$ baseline)")
    axB.set_title("(b) Real steering (blue) shifts every pair upward,\nmore than sham (gray), at all scales", fontsize=10)
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(facecolor=CREAL, alpha=.5, label="real steering"),
                        Patch(facecolor=CSHAM, alpha=.5, label="sham steering")],
               fontsize=8.5, loc="upper left")
    plt.tight_layout(pad=1.3)
    fig.savefig(os.path.join(FIG, "fig5_causal.pdf"), bbox_inches="tight")
    plt.close(fig); print("fig5_causal done")


def fig6_mechanism():
    """Layer x position reciprocal-rank heatmaps for 3 episodes whose bridge
    peaks at its semantic trigger token."""
    import numpy as _np
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("wsblue", ["#f7f9fc", "#C2D0E2", "#6E8FB2", "#3E577E"])
    xl = json.load(open(os.path.join(RESULTS, "layer_trace_cases.json")))
    vf = json.load(open(os.path.join(RESULTS, "layer_trace_v4f_all.json")))
    cases = [(xl, "8",   "``the largest planet'' $\\rightarrow$ Jupiter"),
             (vf, "41",  "``the sacred river'' $\\rightarrow$ India"),
             (xl, "107", "``a dark shaft'' $\\rightarrow$ gold")]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.6))
    for ax, (src, ep, tag) in zip(axes, cases):
        e = src[ep]; g = _np.array(e["grid"]); toks = [t.strip() for t in e["tokens"]]
        L, seq = g.shape
        im = ax.imshow(g, aspect="auto", cmap=cmap, vmin=0, vmax=max(0.02, _np.quantile(g, 0.999)),
                       interpolation="nearest", origin="lower")
        pi, pj = _np.unravel_index(g.argmax(), g.shape)
        ax.add_patch(plt.Rectangle((pj - 0.5, pi - 0.5), 1, 1, fill=False, ec=CACCENT, lw=1.8, zorder=5))
        ax.set_title(f"bridge “{e['bridge']}”  ({tag})", fontsize=10.5, loc="left")
        ax.set_ylabel("layer (depth)")
        ax.set_yticks([0, L // 2, L - 1]); ax.set_yticklabels(["emb", str(L // 2), str(L - 1)])
        ax.set_xticks(range(seq))
        labs = ax.set_xticklabels(toks, rotation=90, fontsize=5.0)
        labs[pj].set_color(CACCENT); labs[pj].set_fontweight("bold"); labs[pj].set_fontsize(7.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.006); cb.ax.tick_params(labelsize=6)
        cb.set_label("recip. rank", fontsize=6.5)
    plt.tight_layout(pad=0.9)
    fig.savefig(os.path.join(FIG, "fig6_mechanism.pdf"), bbox_inches="tight")
    plt.close(fig); print("fig6_mechanism done")

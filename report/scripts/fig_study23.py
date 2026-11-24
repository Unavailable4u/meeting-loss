"""Figures of Study 2 (selective backward, prereg-v2) and Study 3 (prereg-v3). Run: python fig_study23.py"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from common import REPO, TARGETS, arr, curve, curve_steps, load_prereg2, load_prereg3, n_never, pct
from src import prereg3  # repository code: cached score, warm-up credit

style.setup()
OUT = pathlib.Path(__file__).resolve().parents[1] / "figures"
PROC = REPO / "results" / "processed"
G = 1e9
ALPHA3 = 0.05 / 3


def band(ax, x, Y, color, label, ls="-", lw=1.4):
    m = Y.mean(0)
    se = Y.std(0, ddof=1) / np.sqrt(Y.shape[0])
    ax.plot(x, m, color=color, label=label, ls=ls, lw=lw)
    ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.18, lw=0)


# ------------------------------------------------------------------ Study 2
def fig_s2(p2):
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axs[0]
    grid = np.linspace(0, 300e9, 400)
    for k, lab, ls in (("A1", "A1 (tuned baseline)", "-"), ("B48", "B48 (tuned batch 48)", "--"), ("S30", "S30 (selective, kept 30\\%)".replace("\\%", "%"), "-"),
                       ("R30", "R30 (random rows, 30%)", ":")):
        x, Y = curve(p2[k], grid=grid)
        band(ax, x / G, Y, style.ARM[k], lab, ls=ls)
    for t in TARGETS:
        ax.axhline(t, color="#999999", lw=0.7, ls=":")
    ax.set_ylim(0.94, 0.985); ax.set_xlim(0, 300)
    ax.set_xlabel("cumulative GFLOPs"); ax.set_ylabel("validation accuracy")
    ax.legend(loc="lower right")
    style.panel_label(ax, "(a)", -0.14)
    ax = axs[1]
    for j, (t, c) in enumerate(zip(TARGETS, (style.OI["blue"], style.OI["orange"]))):
        r = arr(p2["S30"], t) / arr(p2["A1"], t)
        ax.plot(np.random.default_rng(j).normal(j, 0.06, len(r)), r, "o", color=c, ms=4, alpha=0.8)
        ax.hlines(r.mean(), j - 0.25, j + 0.25, color="black", lw=1.5)
        ax.text(j + 0.28, r.mean(), f"{r.mean():.2f}", va="center", fontsize=8)
    ax.axhline(1, color="black", lw=0.7)
    ax.axhline(0.85, color="#009E73", lw=0.8, ls="--")
    ax.text(1.55, 0.865, "15% margin", color="#009E73", fontsize=7, ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["to 0.97", "to 0.975"]); ax.set_xlim(-0.5, 1.7)
    ax.set_ylim(0.4, 1.1)
    ax.set_ylabel("per-seed FLOPs ratio S30 / A1")
    style.panel_label(ax, "(b)", -0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_s2_curves.pdf")
    plt.close(fig)


def fig_s2_controls():
    sf = pd.read_csv(PROC / "pilot_selective_sf1.csv")
    bc = pd.read_csv(PROC / "batch_control_sb1.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ax = axs[0]
    fr = [0.1, 0.2, 0.3, 0.5]
    cost = {0.1: 0.500, 0.2: 0.557, 0.3: 0.613, 0.5: 0.722}  # exact per-step cost relative to A (E-018)
    s = sf[sf.target == 0.975].set_index("variant")
    steps = [s.loc[f"S{int(f*100)}", "steps_ratio"] for f in fr]
    prod = [s.loc[f"S{int(f*100)}", "ratio_of_means"] for f in fr]
    ax.plot(fr, [cost[f] for f in fr], "s-", color=style.OI["blue"], label="FLOPs per step")
    ax.plot(fr, steps, "o-", color=style.OI["vermillion"], label="steps to 0.975")
    ax.plot(fr, prod, "D-", color="black", label="FLOPs to 0.975")
    ax.axhline(1, color="#999999", lw=0.7)
    ax.set_xlabel("kept fraction"); ax.set_ylabel("ratio to arm A"); ax.set_ylim(0.4, 1.2)
    ax.legend(loc="lower right", fontsize=7.5)
    style.panel_label(ax, "(a)", -0.16)
    ax = axs[1]
    names = ["b32", "b48", "b64", "S30"]
    w = 0.36
    for j, (t, c) in enumerate(zip(TARGETS, (style.OI["blue"], style.OI["orange"]))):
        v = [bc[(bc.variant == n) & (bc.target == t)].ratio_of_means.iloc[0] for n in names]
        ax.bar(np.arange(4) + (j - 0.5) * w, v, w, color=c, label=f"to {t}")
    ax.axhline(1, color="black", lw=0.7)
    ax.set_xticks(range(4)); ax.set_xticklabels(["batch 32", "batch 48", "batch 64", "S30\n(batch 128)"])
    ax.set_ylabel("FLOPs ratio to A (batch 128)"); ax.set_ylim(0.5, 1.3)
    ax.legend(loc="upper left", fontsize=7.5)
    style.panel_label(ax, "(b)", -0.16)
    fig.tight_layout()
    fig.savefig(OUT / "fig_s2_controls.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Study 3
def fig_s3_curves(p3):
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))
    spec = [("A1", "A1 baseline", "-"), ("S30", "S30", "-"), ("S20w0", "S20w0", "-"), ("SBT", "Selective-Backprop (tuned)", "-"), ("R20w0", "random rows, 20%", ":")]
    grid = np.linspace(0, 260e9, 400)
    for k, lab, ls in spec:
        x, Y = curve(p3[k], grid=grid)
        band(axs[0], x / G, Y, style.ARM[k], lab, ls=ls)
        st, Y = curve_steps(p3[k])
        band(axs[1], st, Y, style.ARM[k], lab, ls=ls)
    for ax in axs:
        for t in TARGETS:
            ax.axhline(t, color="#999999", lw=0.7, ls=":")
        ax.set_ylim(0.93, 0.985)
    axs[0].set_xlim(0, 260); axs[1].set_xlim(0, 2200)
    axs[0].set_xlabel("cumulative GFLOPs (selection passes included)"); axs[1].set_xlabel("optimizer updates")
    axs[0].set_ylabel("validation accuracy")
    axs[0].legend(loc="lower right", fontsize=7)
    style.panel_label(axs[0], "(a)", -0.16); style.panel_label(axs[1], "(b)", -0.1)
    fig.tight_layout()
    fig.savefig(OUT / "fig_s3_curves.pdf")
    plt.close(fig)


def ci_mean(v, alpha=0.05, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    return np.quantile(v[idx].mean(1), [alpha / 2, 1 - alpha / 2])


def fig_s3_bars(p3):
    names = [("A1", "A1 baseline"), ("A2", "A2"), ("A5", "A5"), ("A6", "A6"), ("A7", "A7"), ("A8", "A8"), ("SBT", "Selective-Backprop (tuned)"),
             ("R20w0", "random rows, 20%, no warm-up"), ("S30", "S30"), ("S20", "S20"), ("S15", "S15"), ("S20w10", "S20w10"), ("S20w0", "S20w0")]
    fam = lambda k: style.OI["black"] if k.startswith("A") else (style.OI["vermillion"] if k == "SBT" else (style.OI["grey"] if k.startswith("R") else style.OI["blue"]))
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
    for ax, t in zip(axs, TARGETS):
        for i, (k, lab) in enumerate(names):
            v = arr(p3[k], t) / G
            lo, hi = ci_mean(v)
            ax.barh(i, v.mean(), color=fam(k), alpha=0.85, height=0.7)
            ax.errorbar(v.mean(), i, xerr=[[v.mean() - lo], [hi - v.mean()]], color="black", capsize=2, lw=0.9)
            ax.text(hi + 4, i, f"{v.mean():.0f}", va="center", fontsize=7)
        ax.set_xlabel(f"GFLOPs to {t}")
        ax.set_xlim(0, 330)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(names)))
    axs[0].set_yticklabels([n[1] for n in names])
    axs[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "fig_s3_bars.pdf")
    plt.close(fig)


def h_rows(p3):
    rows = []
    for name, ref, margin, key in (("H8", "A1", 15, "score"), ("H9", "S30", 10, "score"), ("H10", "SBT", 15, "score")):
        for t in TARGETS:
            r = arr(p3[ref], t)
            g = arr(p3["S20w0"], t)
            rows.append((name, ref, margin, t) + pct(r, g, alpha=ALPHA3))
    return rows


def fig_s3_forest(p3):
    rows = h_rows(p3)
    labs = {"H8": "H8: vs. tuned baseline A1", "H9": "H9: vs. S30", "H10": "H10: vs. tuned Selective-Backprop"}
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    ypos = {"H8": 0, "H9": 1, "H10": 2}
    for name, ref, margin, t, p, lo, hi in rows:
        y = ypos[name] + (-0.15 if t == 0.97 else 0.15)
        ax.errorbar(p, y, xerr=[[p - lo], [hi - p]], fmt="o" if t == 0.97 else "s", color=style.ARM["S20w0"], mfc=style.ARM["S20w0"] if t == 0.97 else "white",
                    capsize=2.5, lw=1.2, ms=5, label=f"target {t}" if name == "H8" else None)
        ax.plot([-margin, -margin], [ypos[name] - 0.38, ypos[name] + 0.38], color="black", lw=1.6)
        ax.text(hi + 1.2, y, f"{p:+.1f}", va="center", fontsize=7.5)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(range(3)); ax.set_yticklabels([labs[k] for k in ("H8", "H9", "H10")]); ax.invert_yaxis()
    ax.set_xlim(-70, 12)
    ax.set_xlabel("S20w0: paired change in FLOPs-to-target (%), 98.3% interval; tick = margin")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.72), fontsize=7.5)
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig_s3_forest.pdf")
    plt.close(fig)


def fig_s3_paired(p3):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    ref_spec = [("A1", style.ARM["A1"], "o"), ("S30", style.ARM["S30"], "s"), ("SBT", style.ARM["SBT"], "^")]
    for ax, t in zip(axs, TARGETS):
        g = arr(p3["S20w0"], t) / G
        for k, c, m in ref_spec:
            r = arr(p3[k], t) / G
            ax.plot(r, g, m, color=c, ms=4, alpha=0.85, label=f"vs. {k if k != 'SBT' else 'Selective-Backprop'}")
        lim = [40, 420]
        ax.plot(lim, lim, color="black", lw=0.8)
        ax.plot(lim, [0.85 * l for l in lim], color="#009E73", lw=0.7, ls="--")
        ax.set_xlim(lim); ax.set_ylim(40, 300)
        ax.set_xlabel(f"reference GFLOPs to {t}")
    axs[0].set_ylabel("S20w0 GFLOPs to target (same seed)")
    axs[0].text(300, 232, "$y = x$", fontsize=7.5, rotation=0)
    axs[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_s3_paired.pdf")
    plt.close(fig)


def fig_s3_waterfall(p3):
    t = 0.975
    m = {k: arr(p3[k], t).mean() / G for k in ("A1", "S30", "S20", "S20w0", "S20w10", "S15")}
    rows20 = p3["S20"][sorted(p3["S20"])[0]]["rows"]
    full = prereg3.full_step_flops(p3["A1"][sorted(p3["A1"])[0]]["rows"])
    sel20 = prereg3.selective_step_flops(rows20, 100)
    arith = 100 * (full - sel20) / G  # cost of 100 full steps over 100 selective steps (kept 0.2)
    steps = [("A1\nbaseline", m["A1"], None), ("S30", m["S30"], m["S30"] - m["A1"]), ("S20", m["S20"], m["S20"] - m["S30"]), ("S20w0", m["S20w0"], m["S20w0"] - m["S20"])]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    prev = None
    labels = ["A1\nbaseline", "S30\n(select, 0.3,\nwarm-up 100)", "S20\n(kept 0.2)", "S20w0\n(no warm-up)"]
    for i, (lab, val, d) in enumerate(steps):
        if d is None:
            ax.bar(i, val, color=style.OI["black"], width=0.6)
        else:
            ax.bar(i, -d, bottom=val, color=style.OI["green"], width=0.6)
            ax.plot([i - 1 + 0.3, i - 0.3], [prev, prev], color="#888888", lw=0.7)
        if i == 3:
            ax.bar(i, arith, bottom=val, width=0.6, color="none", edgecolor="black", hatch="////", lw=0.6)
            ax.text(i + 0.36, val + arith / 2, f"{arith:.1f} GFLOPs: saved\ncost of full steps", fontsize=6.5, va="center")
        ax.text(i, (val + (-d if d else 0)) + 4, f"{val:.0f}", ha="center", fontsize=8)
        prev = val
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("mean GFLOPs to 0.975"); ax.set_ylim(0, 250); ax.set_xlim(-0.5, 4.2)
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "fig_s3_waterfall.pdf")
    plt.close(fig)
    return arith


def fig_metrics(p3):
    t = 0.975
    a1_f, a1_s = arr(p3["A1"], t).mean(), arr(p3["A1"], t, key="steps").mean()
    rows = []
    for k in ("S20w0", "S30", "SBT"):
        rows.append((k, arr(p3[k], t, key="steps").mean() / a1_s, arr(p3[k], t).mean() / a1_f))
    # cached count of SBT
    cached = np.array([prereg3.score_run_cached(p3["SBT"][s]["rows"], t) for s in sorted(p3["SBT"])], float)
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    labs = ["S20w0", "S30", "Sel.-Backprop\n(uncached)", "Sel.-Backprop\n(cached)"]
    steps = [r[1] for r in rows] + [rows[2][1]]
    fl = [r[2] for r in rows] + [cached.mean() / a1_f]
    x = np.arange(4); w = 0.36
    ax.bar(x - w / 2, steps, w, color=style.OI["vermillion"], label="optimizer updates to 0.975")
    ax.bar(x + w / 2, fl, w, color=style.OI["blue"], label="FLOPs to 0.975")
    for xi, a, b in zip(x, steps, fl):
        ax.text(xi - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7); ax.text(xi + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7)
    ax.axhline(1, color="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7.5); ax.set_ylim(0, 1.35); ax.set_ylabel("ratio to A1")
    ax.legend(loc="upper left", fontsize=7)
    style.panel_label(ax, "(a)", -0.15)
    ax = axs[1]
    tm = pd.read_csv(PROC / "timing_sw1.csv")
    flops_ratio = 0.613
    vals = []
    for th in (1, 8):
        d = tm[tm.threads == th]
        a, s = d[d.variant == "A"].ms_per_step.median(), d[d.variant == "S30"].ms_per_step.median()
        vals.append(s / a)
    ax.bar([0, 1, 2], [flops_ratio, vals[0], vals[1]], color=[style.OI["blue"], style.OI["orange"], style.OI["orange"]], width=0.6)
    for i, v in enumerate([flops_ratio] + vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(1, color="black", lw=0.7)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["FLOPs\nper step", "time per step\n1 thread", "time per step\n8 threads"], fontsize=7.5)
    ax.set_ylim(0, 1.2); ax.set_ylabel("S30 relative to A")
    style.panel_label(ax, "(b)", -0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_metrics.pdf")
    plt.close(fig)
    return rows, cached.mean() / a1_f


if __name__ == "__main__":
    p2, p3 = load_prereg2(), load_prereg3()
    fig_s2(p2)
    fig_s2_controls()
    fig_s3_curves(p3)
    fig_s3_bars(p3)
    fig_s3_forest(p3)
    fig_s3_paired(p3)
    print("warm-up arithmetic GFLOPs (S20):", fig_s3_waterfall(p3))
    print(fig_metrics(p3))
    for r in h_rows(p3):
        print(r)

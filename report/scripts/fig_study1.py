"""Figures of Study 1 (meeting loss) and of the mechanism checks. Run: python fig_study1.py"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import style
from common import REPO, TARGETS, arr, curve, curve_steps, load_stage3, load_prereg2, load_prereg3, pct

style.setup()
OUT = pathlib.Path(__file__).resolve().parents[1] / "figures"
PROC = REPO / "results" / "processed"
G = 1e9


def band(ax, x, Y, color, label, ls="-", lw=1.4):
    m = Y.mean(0)
    se = Y.std(0, ddof=1) / np.sqrt(Y.shape[0])
    ax.plot(x, m, color=color, label=label, ls=ls, lw=lw)
    ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.18, lw=0)


# ---------------------------------------------------------------- learning curves, Study 1
def fig_s1_curves(s3):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    order = [("A", "A (tuned baseline)"), ("A2", "A2 (second tuning)"), ("B", "B (deep supervision)"), ("D", "D (meeting loss)")]
    for k, lab in order:
        ls = "--" if k == "A2" else "-"
        st, Y = curve_steps(s3[k])
        band(axs[0], st, Y, style.ARM[k], lab, ls=ls)
        x, Y = curve(s3[k], grid=np.linspace(0, 400e9, 400))
        band(axs[1], x / G, Y, style.ARM[k], None, ls=ls)
        style.mean_marker(axs[1], arr(s3[k], 0.975).mean() / G, style.ARM[k])
    for ax in axs:
        for t in TARGETS:
            ax.axhline(t, color="#999999", lw=0.7, ls=":")
        ax.set_ylim(0.94, 0.985)
    axs[0].set_xlim(0, 2200)
    axs[1].set_xlim(0, 360)
    axs[0].set_xlabel("training step")
    axs[1].set_xlabel("cumulative GFLOPs")
    axs[0].set_ylabel("validation accuracy")
    # target labels sit outside the data area, in the right margin of panel (b)
    for t in TARGETS:
        axs[1].text(1.01, t, f"{t:.3f}", transform=axs[1].get_yaxis_transform(), ha="left", va="center", fontsize=7, color="#666666", clip_on=False)
    h, l = axs[0].get_legend_handles_labels()
    h.append(style.mean_marker_handle()); l.append(h[-1].get_label())
    style.panel_label(axs[0], "(a)", -0.2)
    style.panel_label(axs[1], "(b)", -0.1)
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    fig.savefig(OUT / "fig_s1_curves.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- forest plot, Study 1
def fig_s1_forest(s3):
    rows = [("D", "D meeting loss ($H1$)", 20), ("B", "B deep supervision ($H6$)", 20), ("A2", "A2 second tuning of A", 20),
            ("V3", "V3 frozen $G$", 10), ("V2", "V2 augmented output", 10), ("V1", "V1 noisy inverse", 10), ("F", "F exact-gradient target", 10),
            ("E", "E lower-layer lr multiplier", 10), ("B-code", "B-code (DSN reference code)", 10), ("C", "C autoencoder", 10)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axvspan(-100, -15, color="#009E73", alpha=0.10, lw=0)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(-15, color="#009E73", lw=0.8, ls="--")
    for i, (k, lab, n) in enumerate(rows):
        for j, (t, mk, off) in enumerate(zip(TARGETS, ("o", "s"), (-0.14, 0.14))):
            seeds = list(range(n))
            ref = arr({s: s3["A"][s] for s in seeds}, t)
            grp = arr({s: s3[k][s] for s in seeds}, t)
            p, lo, hi = pct(ref, grp, alpha=0.025)
            col = style.ARM.get(k, "#555555")
            ax.errorbar(p, i + off, xerr=[[p - lo], [hi - p]], fmt=mk, color=col, ms=4.5, lw=1.1, capsize=2,
                        mfc=col if j == 0 else "white", label=f"target {t}" if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(-32, 180)
    ax.set_xlabel("paired change in FLOPs-to-target vs. tuned arm A (%), 97.5% bootstrap interval\nnegative = fewer FLOPs than A")
    # the success zone is labelled inside its own band (rotated), so that no text touches a marker or an interval
    ax.text(-23.5, (len(rows) - 1) / 2, "gain $\\geq$ 15%", color="#007A5A", fontsize=7.5, ha="center", va="center", rotation=90)
    ax.axhline(2.5, color="#BBBBBB", lw=0.6)
    gt = ax.get_yaxis_transform()
    ax.text(0.99, 1.5, "20 seeds", transform=gt, ha="right", va="center", fontsize=7.5, color="#666666")
    ax.text(0.99, 5.0, "10 seeds (descriptive)", transform=gt, ha="right", va="center", fontsize=7.5, color="#666666")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig_s1_forest.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- per-step overhead
def fig_overhead():
    d = pd.read_csv(PROC / "flops_overhead_v1.1.csv")
    names = {"base": "A baseline", "deepsup": "B deep supervision", "meet_frozen": "V3 frozen $G$", "meet": "D meeting loss",
             "meet+sigma_inv": "V1 noisy inverse", "meet_ao": "V2 augmented output", "ae": "C autoencoder", "meet_grad": "F exact-gradient target"}
    d = d[d.arm != "base"].copy()
    d["lab"] = d.arm.map(names)
    d = d.sort_values("overhead_vs_A_pct")
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    cols = [style.OI["vermillion"] if a.startswith("meet") else style.OI["sky"] for a in d.arm]
    ax.barh(d.lab, d.overhead_vs_A_pct, color=cols, height=0.62)
    for y, v in enumerate(d.overhead_vs_A_pct):
        be = 100 * (1 - 1 / (1 + v / 100))
        ax.text(v + 1.5, y, f"+{v:.1f}%  (break-even: {be:.0f}% fewer steps)", va="center", fontsize=7.5)
    ax.set_xlim(0, 215)
    ax.set_xlabel("extra FLOPs per training step relative to arm A (%)")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig_overhead.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- tuning noise
def sel_score(csv):
    return pd.read_csv(PROC / csv).sort_values("score_flops").iloc[0].score_flops / G


def fig_tuning_noise(s3, p2):
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]
    items = [("A", sel_score("sweep_t1_base_two.csv")), ("A2", sel_score("sweep_t2_base_two.csv")),
             ("B", sel_score("sweep_t1_deepsup_two.csv")), ("D", sel_score("sweep_t1_meet_two.csv"))]
    for i, (k, sel) in enumerate(items):
        fin = arr(s3[k], 0.975).mean() / G
        ax.plot([sel, fin], [i, i], color="#999999", lw=1)
        ax.plot(sel, i, "o", mfc="white", mec=style.ARM[k], ms=6, mew=1.4)
        ax.plot(fin, i, "o", color=style.ARM[k], ms=6)
        ax.text(max(sel, fin) + 6, i, f"{sel:.0f}$\\to${fin:.0f}", va="center", fontsize=7.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["A", "A2", "B", "D"])
    ax.invert_yaxis()
    ax.set_xlim(170, 340)
    ax.set_xlabel("GFLOPs to 0.975")
    ax.set_title("selection score (2 seeds, open)\nvs. final (20 seeds, filled)", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    style.panel_label(ax, "(a)", -0.22)

    ax = axs[1]
    sel = {"A1": sel_score("sweep_t1_base_two.csv"), "A2": sel_score("sweep_t2_base_two.csv"), "A5": sel_score("sweep_t5_base_two.csv"),
           "A6": sel_score("sweep_t6_base_two.csv"), "A7": sel_score("sweep_t7_base_two.csv"), "A8": sel_score("sweep_t8_base_two.csv")}
    rng = np.random.default_rng(0)
    for i, k in enumerate(sel):
        v = arr(p2[k], 0.975) / G
        m = v.mean()
        idx = rng.integers(0, len(v), (10000, len(v)))
        lo, hi = np.quantile(v[idx].mean(1), [0.025, 0.975])
        ax.plot(rng.normal(i, 0.07, len(v)), v, ".", color="#BBBBBB", ms=3, zorder=1)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=style.OI["blue"], capsize=3, ms=5, zorder=3)
        ax.plot(i, sel[k], "D", mfc="white", mec="black", ms=5.5, zorder=4)
    ax.set_xticks(range(6))
    ax.set_xticklabels(list(sel))
    ax.set_ylabel("GFLOPs to 0.975")
    ax.set_title("six tunings of the baseline, 20 fresh seeds", fontsize=8.5)
    ax.plot([], [], "D", mfc="white", mec="black", label="selection score")
    ax.plot([], [], "o", color=style.OI["blue"], label="mean, 95% CI")
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_, hi_ + 0.2 * (hi_ - lo_))      # headroom for the legend, so that it covers no point
    ax.legend(loc="upper center", ncol=2, fontsize=7.5)
    style.panel_label(ax, "(b)", -0.18)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tuning_noise.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- mechanism
def fig_dose():
    d1, d2, d3 = (pd.read_csv(PROC / f"dose_response_mr{i}.csv") for i in (1, 2, 3))
    relu = pd.concat([d2[d2.rho > 0], d1[d1.rho > 0]])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, tgt in zip(axs[:2], (0.97, 0.975)):
        for d, lab, col in ((relu, "relu", style.OI["vermillion"]), (d3[d3.rho > 0], "tanh", style.OI["blue"])):
            s = d[d.target == tgt].sort_values("rho")
            ax.errorbar(s.rho, s.steps_change_pct, yerr=[s.steps_change_pct - s.steps_lo, s.steps_hi - s.steps_change_pct],
                        fmt="o-", color=col, capsize=2, ms=4, label=lab, lw=1.1)
        ax.axhline(0, color="black", lw=0.7)
        ax.axhline(-10, color="#009E73", lw=0.8, ls="--", label="10% gain" if tgt == 0.97 else None)
        ax.set_xscale("log")
        ax.set_ylim(-30, 240)
        ax.set_xlabel(r"dose $\rho$")
        ax.set_title(f"steps to {tgt}", fontsize=8.5)
    axs[0].set_ylabel("change vs. $\\rho=0$ (%)")
    axs[0].legend(loc="upper left")
    ax = axs[2]
    for d, lab, col in ((pd.concat([d2, d1[d1.rho > 0]]).drop_duplicates(["rho", "target"]), "relu", style.OI["vermillion"]), (d3, "tanh", style.OI["blue"])):
        s = d[d.target == 0.975].sort_values("rho")
        ax.plot(np.where(s.rho == 0, 3e-4, s.rho), s.diag_h_var, "o-", color=col, ms=4, lw=1.1, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"dose $\rho$ ($\rho=0$ drawn at $3\!\times\!10^{-4}$)", fontsize=8)
    ax.set_title("scale of the middle state $h$", fontsize=8.5)
    ax.set_ylabel("variance of $h$")
    for ax, l in zip(axs, "(a)(b)(c)".replace(")(", ")|(").split("|")):
        style.panel_label(ax, l, -0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dose.pdf")
    plt.close(fig)


def fig_concentration():
    e = pd.read_csv(PROC / "gradient_concentration_gc1.csv")
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    cmap = {0.1: style.OI["orange"], 0.3: style.OI["blue"], 0.5: style.OI["green"]}
    for f, c in cmap.items():
        for mode, ls in (("top", "-"), ("random", "--")):
            s = e[(e["mode"] == mode) & (e.fraction == f)].groupby("step").norm_share.mean()
            ax.plot(s.index, s.values, ls=ls, color=c if mode == "top" else "#888888", marker="o" if mode == "top" else None, ms=3,
                    label=f"top {int(f*100)}%" if mode == "top" else None)
    ax.plot([], [], "--", color="#888888", label="random (same size)")
    ax.set_xscale("log")
    ax.set_xlabel("training step (arm A, fixed batches)")
    ax.set_ylabel("share of per-example gradient norm")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7)   # below the axes: no line crosses it
    fig.savefig(OUT / "fig_concentration.pdf")
    plt.close(fig)


def fig_geometry():
    d = pd.read_csv(PROC / "mechanism_a_ma1.csv")
    m = d[d.arm == "meet"]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax = axs[0]
    for q, lab, c in (("cos_task_meet", "cosine$(g_{task}, g_{meet})$", style.OI["vermillion"]), ("par_share", "parallel share of $\\|g_{meet}\\|^2$", style.OI["blue"])):
        s = m[m.quantity == q].groupby("step").value
        mu, lo, hi = s.mean(), s.min(), s.max()
        ax.plot(mu.index, mu.values, "o-", color=c, ms=3.5, label=lab)
        ax.fill_between(mu.index, lo.values, hi.values, color=c, alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("training step"); ax.set_title("lower-layer parameter space", fontsize=8.5)
    ax = axs[1]
    for q, lab, c in (("cos_meet_bp_w", "meeting vs. backprop", style.OI["vermillion"]), ("cos_meet_pinv_w", "meeting vs. Gauss--Newton", style.OI["green"]),
                      ("cos_bp_pinv_w", "backprop vs. Gauss--Newton", style.OI["black"])):
        s = m[m.quantity == q].groupby("step").value
        mu, lo, hi = s.mean(), s.min(), s.max()
        ax.plot(mu.index, mu.values, "o-", color=c, ms=3.5, label=lab)
        ax.fill_between(mu.index, lo.values, hi.values, color=c, alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("training step"); ax.set_title("middle-state space ($h$), per example", fontsize=8.5)
    style.panel_label(axs[0], "(a)", -0.15); style.panel_label(axs[1], "(b)", -0.15)
    fig.tight_layout()
    for ax in axs:                                 # legends go below the panels, created after tight_layout so that they do not shrink the axes
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=7)
    fig.savefig(OUT / "fig_geometry.pdf")
    plt.close(fig)


if __name__ == "__main__":
    s3, p2 = load_stage3(), load_prereg2()
    fig_s1_curves(s3)
    fig_s1_forest(s3)
    fig_overhead()
    fig_tuning_noise(s3, p2)
    fig_dose()
    fig_concentration()
    fig_geometry()
    print("done")

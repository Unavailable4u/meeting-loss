"""Write the LaTeX table bodies used in the thesis (tables/*.tex), computed from the raw logs. Run: python make_tables.py"""
import json
import pathlib

import numpy as np
import yaml

from common import REPO, TARGETS, arr, load_prereg2, load_prereg3, load_stage3, n_never, pct

OUT = pathlib.Path(__file__).resolve().parents[1] / "tables"
G = 1e9


def sg(v, d=1):
    return f"${v:+.{d}f}$"


def ci_mean(v, alpha=0.05, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    return np.quantile(v[idx].mean(1), [alpha / 2, 1 - alpha / 2])


import re


def write(name, text):
    text = re.sub(r"\$([+-])\$(\d+\.\d)", r"$\1\2$", text)
    text = text.rstrip()
    if text.endswith("\\\\"):
        text = text[:-2].rstrip()  # no trailing row end: a row end at EOF of an \\input breaks the tabular; the .tex adds it
    (OUT / name).write_text(text + "\n")


def table_s1(s3):
    names = {"A": "A (tuned baseline)", "D": "D (meeting loss)", "B": "B (deep supervision)", "A2": "A2 (second tuning of A)"}
    for t, tag in zip(TARGETS, ("97", "975")):
        lines = []
        ref = arr(s3["A"], t)
        for k in ("A", "D", "B", "A2"):
            v = arr(s3[k], t) / G
            lo, hi = ci_mean(v * G, alpha=0.025)
            st = np.nanmean(arr(s3[k], t, key="steps"))
            if k == "A":
                ch = "--"
                verdict = "reference"
            else:
                p, plo, phi = pct(ref, arr(s3[k], t), alpha=0.025)
                ch = f"{sg(p)} [{plo:+.1f}, {phi:+.1f}]".replace("[+", "[$+$").replace("[-", "[$-$").replace(", +", ", $+$").replace(", -", ", $-$")
                verdict = {"D": "refuted", "B": "refuted"}.get(k, "descriptive")
            lines.append(f"{names[k]} & {v.mean():.1f} & [{lo/G:.1f}, {hi/G:.1f}] & {st:.0f} & {n_never(s3[k], t)} & {ch} & {verdict} \\\\")
        write(f"tab_s1_main_{tag}.tex", "\n".join(lines) + "\n")


def table_s1_desc(s3):
    names = {"V3": "V3 frozen random $G$", "V2": "V2 augmented output", "V1": "V1 noisy inverse", "F": "F exact-gradient target (control)",
             "E": "E lower-layer lr multiplier (control)", "B-code": "B-code (DSN reference-code variant)", "C": "C autoencoder auxiliary"}
    lines = []
    for k, lab in names.items():
        cells = []
        for t in TARGETS:
            seeds = list(range(10))
            ref = arr({s: s3["A"][s] for s in seeds}, t)
            grp = arr({s: s3[k][s] for s in seeds}, t)
            p, lo, hi = pct(ref, grp, alpha=0.025)
            cells.append(f"{sg(p)} [{lo:+.1f}, {hi:+.1f}]".replace("[+", "[$+$").replace("[-", "[$-$").replace(", +", ", $+$").replace(", -", ", $-$"))
        lines.append(f"{lab} & {cells[0]} & {cells[1]} \\\\")
    write("tab_s1_desc.tex", "\n".join(lines) + "\n")


def mean_steps(g, t):
    return np.nanmean(arr(g, t, key="steps"))


def table_s2(p2):
    names = [("A1", "A1 (tuned baseline)"), ("A2", "A2"), ("A5", "A5"), ("A6", "A6"), ("A7", "A7"), ("A8", "A8"), ("B48", "B48 (tuned batch 48)"),
             ("S30", "S30 (selective, kept 0.3)"), ("S20", "S20 (kept 0.2, descriptive)"), ("S30T", "S30T (S tuned once, descriptive)"), ("R30", "R30 (random 30\\%, control)")]
    lines = []
    for k, lab in names:
        cells = []
        for t in TARGETS:
            v = arr(p2[k], t) / G
            if k == "A1":
                ch = "--"
            else:
                p, lo, hi = pct(arr(p2["A1"], t), arr(p2[k], t), alpha=0.025)
                ch = f"{sg(p)} [{lo:+.1f}, {hi:+.1f}]".replace("[+", "[$+$").replace("[-", "[$-$").replace(", +", ", $+$").replace(", -", ", $-$")
            cells.append(f"{v.mean():.1f} & {mean_steps(p2[k], t):.0f} & {n_never(p2[k], t)} & {ch}")
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        if k in ("A8", "B48", "S30"):
            lines.append("\\midrule" if k != "S30" else "\\midrule")
    write("tab_s2_main.tex", "\n".join(lines) + "\n")


def table_s3(p3):
    names = [("A1", "A1 (tuned baseline)"), ("A2", "A2"), ("A5", "A5"), ("A6", "A6"), ("A7", "A7"), ("A8", "A8"),
             ("SBT", "SBT (Selective-Backprop, tuned)"), ("S30", "S30"), ("S20", "S20"), ("S15", "S15"), ("S20w10", "S20w10"),
             ("S20w0", "S20w0 (primary arm)"), ("R20w0", "R20w0 (random 20\\%, no warm-up)")]
    lines = []
    for k, lab in names:
        cells = []
        for t in TARGETS:
            v = arr(p3[k], t) / G
            if k == "A1":
                ch = "--"
            else:
                p, lo, hi = pct(arr(p3["A1"], t), arr(p3[k], t), alpha=0.05 / 3)
                ch = f"{sg(p)} [{lo:+.1f}, {hi:+.1f}]".replace("[+", "[$+$").replace("[-", "[$-$").replace(", +", ", $+$").replace(", -", ", $-$")
            cells.append(f"{v.mean():.1f} & {mean_steps(p3[k], t):.0f} & {n_never(p3[k], t)} & {ch}")
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        if k in ("A8", "SBT", "S20w0"):
            lines.append("\\midrule")
    write("tab_s3_main.tex", "\n".join(lines) + "\n")


def table_hyp(p3, p2, s3):
    """Summary of every pre-registered hypothesis with its outcome (numbers recomputed)."""
    L = []

    def row(h, comp, margin, tgt_cells, verdict):
        L.append(f"{h} & {comp} & {margin} & {tgt_cells[0]} & {tgt_cells[1]} & {verdict} \\\\")

    def cell(p, lo, hi):
        return f"{sg(p)} [{lo:+.1f}, {hi:+.1f}]".replace("[+", "[$+$").replace("[-", "[$-$").replace(", +", ", $+$").replace(", -", ", $-$")

    for h, k, ver in (("H1", "D", "refuted"), ("H6", "B", "refuted")):
        c = [cell(*pct(arr(s3["A"], t), arr(s3[k], t), alpha=0.025)) for t in TARGETS]
        row(h, f"{k} vs.\\ A", "15\\%", c, ver)
    L.append("\\midrule")
    c = [cell(*pct(arr(p2["A1"], t), arr(p2["S30"], t), alpha=0.025)) for t in TARGETS]
    row("H7a", "S30 vs.\\ A1", "15\\%", c, "supported")
    c = [cell(*pct(arr(p2["B48"], t), arr(p2["S30"], t), alpha=0.025)) for t in TARGETS]
    row("H7b", "S30 vs.\\ B48", "0\\%", c, "supported")
    L.append("\\midrule")
    for h, ref, m in (("H8", "A1", "15\\%"), ("H9", "S30", "10\\%"), ("H10", "SBT", "15\\%")):
        c = [cell(*pct(arr(p3[ref], t), arr(p3["S20w0"], t), alpha=0.05 / 3)) for t in TARGETS]
        row(h, f"S20w0 vs.\\ {ref}", m, c, "supported")
    write("tab_hyp.tex", "\n".join(L) + "\n")


def table_shift(p3, p2):
    L = []
    for k in ("A2", "A5", "A6", "A7", "A8"):
        cells = []
        for P in (p2, p3):
            for t in TARGETS:
                p, lo, hi = pct(arr(P["A1"], t), arr(P[k], t), alpha=0.025)
                cells.append(f"{sg(p)}")
        L.append(f"{k} & " + " & ".join(cells) + " \\\\")
    # noise floor: max/min of the means over the six tunings
    nf = []
    for P in (p2, p3):
        for t in TARGETS:
            m = [arr(P[k], t).mean() for k in ("A1", "A2", "A5", "A6", "A7", "A8")]
            nf.append(f"{max(m)/min(m):.3f}")
    L.append("\\midrule")
    L.append("max/min of the six means & " + " & ".join(nf) + " \\\\")
    write("tab_shift.tex", "\n".join(L) + "\n")


def table_tuned():
    base = REPO / "configs" / "tuned"
    spec = [("A1", "t1_base_two", "A"), ("A2", "t2_base_two", "A"), ("A5", "t5_base_two", "A"), ("A6", "t6_base_two", "A"), ("A7", "t7_base_two", "A"),
            ("A8", "t8_base_two", "A"), ("B", "t1_deepsup_two", "B"), ("D", "t1_meet_two", "D"), ("C", "t1_ae_two", "C"),
            ("B48", "t9_base_b48_two", "A"), ("SBT", "t11_sb_two", "SB")]
    L = []
    for name, f, arm in spec:
        c = yaml.safe_load(open(base / f"{f}.yaml"))
        extra = ""
        if arm == "D":
            extra = f"$\\lambda_{{meet}}$={c['lam_meet']:.4f}, $\\lambda_{{inv}}$={c['lam_inv']:.3f}, warm-up {c['warmup_steps']}"
        elif arm == "B":
            extra = f"$\\lambda_{{aux}}$={c['lam_aux']:.4f} (linear decay)"
        elif arm == "C":
            extra = f"$\\lambda_{{aux}}$={c['lam_aux']:.4f}"
        elif arm == "SB":
            extra = "$\\beta$=2, history 1024, one-epoch priming (fixed)"
        elif name == "B48":
            extra = "batch 48, 8000 steps"
        L.append(f"{name} & {f.replace('_', '\\_')} & {c['act']} & {c['lr']:.2e} & {c['wd']:.2e} & {c['lr_schedule']} & {extra} \\\\")
    write("tab_tuned.tex", "\n".join(L) + "\n")


def table_overhead():
    import pandas as pd
    d = pd.read_csv(REPO / "results" / "processed" / "flops_overhead_v1.1.csv")
    names = {"base": "A baseline", "deepsup": "B deep supervision", "ae": "C autoencoder auxiliary", "meet": "D meeting loss", "meet+sigma_inv": "V1 noisy inverse",
             "meet_ao": "V2 augmented output", "meet_frozen": "V3 frozen random $G$", "meet_grad": "F exact-gradient target"}
    L = []
    for _, r in d.iterrows():
        be = 100 * (1 - 1 / (1 + r.overhead_vs_A_pct / 100))
        L.append(f"{names[r.arm]} & {int(r.flops_per_step):,} & {r.overhead_vs_A_pct:+.2f} & {be:.1f} \\\\".replace(",", "\\,"))
    write("tab_overhead.tex", "\n".join(L) + "\n")


def table_selection():
    import pandas as pd
    spec = [("A (= A1)", "sweep_t1_base_two"), ("A2 (re-tuning)", "sweep_t2_base_two"), ("B", "sweep_t1_deepsup_two"), ("B-code", "sweep_t1_deepsup_code_two"),
            ("C", "sweep_t1_ae_two"), ("D", "sweep_t1_meet_two"), ("E", "sweep_t1_base_lowlr_two"), ("F", "sweep_t1_meet_grad_two"),
            ("V1", "sweep_t1_meet_noise_two"), ("V2", "sweep_t1_meet_ao_two"), ("V3", "sweep_t1_meet_frozen_two")]
    L = []
    for lab, f in spec:
        d = pd.read_csv(REPO / "results" / "processed" / f"{f}.csv").sort_values("score_flops")
        r = d.iloc[0]
        L.append(f"{lab} & {r.score_flops/G:.1f} & {d.score_flops.median()/G:.1f} & {r.late_val_acc:.4f} \\\\")
    write("tab_selection.tex", "\n".join(L) + "\n")


def table_vs_a2(s3):
    L = []
    for k, lab in (("D", "D (meeting loss)"), ("B", "B (deep supervision)")):
        cells = []
        for t in TARGETS:
            p, lo, hi = pct(arr(s3["A2"], t), arr(s3[k], t), alpha=0.025)
            cells.append(f"{sg(p)} $[{lo:+.1f}, {hi:+.1f}]$")
        L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("tab_s1_vs_a2.tex", "\n".join(L) + "\n")


def table_explore():
    import pandas as pd
    for name, f in (("tab_headroom.tex", "improve_headroom"), ("tab_look2.tex", "improve_look2")):
        d = pd.read_csv(REPO / "results" / "processed" / f"{f}.csv")
        d = d[d.metric == "flops"]
        groups = list(dict.fromkeys(d.group))
        L = []
        for g in groups:
            cells = []
            for ref in ("S30", "A1"):
                for t in TARGETS:
                    r = d[(d.group == g) & (d.reference == ref) & (d.target == t)]
                    if g == ref or r.empty:
                        cells.append("--")
                    else:
                        r = r.iloc[0]
                        cells.append(f"${r.paired_change_pct:+.1f}$ $[{r.lo:+.1f},{r.hi:+.1f}]$")
            gl = g.replace("_cached", " (cached)")
            L.append(f"{gl} & " + " & ".join(cells) + " \\\\")
        write(name, "\n".join(L) + "\n")


if __name__ == "__main__":
    s3, p2, p3 = load_stage3(), load_prereg2(), load_prereg3()
    table_s1(s3); table_s1_desc(s3); table_s2(p2); table_s3(p3); table_hyp(p3, p2, s3); table_shift(p3, p2); table_tuned(); table_overhead(); table_selection(); table_vs_a2(s3); table_explore()
    print("tables written:", sorted(p.name for p in OUT.glob("*.tex")))

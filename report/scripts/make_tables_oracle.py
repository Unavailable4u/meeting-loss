"""LaTeX table bodies for the root-cause checks B and D (chapter 6), computed from results/processed/oracle_*.csv, which
scripts/run_root_cause.py writes from the raw logs. Run: python make_tables_oracle.py   (from report/scripts)"""
import csv
import glob
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.crossing import DEFAULT_CROSSING, crossing_index  # noqa: E402
from src.sweep import paired_pct_change  # noqa: E402
OUT = pathlib.Path(__file__).resolve().parents[1] / "tables"
T = ("0.97", "0.975")

LABEL = {"F": "F$'$ exact gradient (rescaled)", "P": "P pseudo-inverse", "Q": "Q damped logit inverse", "Pr": "Pr P, radially bounded",
         "D": "D learned $G$ (tuned, descriptive)", "R1": "R1 learned $G$, cut at $h$", "R1w": "R1w as R1 after 300 backprop steps",
         "R2": "R2 oracle P, cut at $h$", "R4": "R4 frozen random $G$, cut at $h$",
         "A": "A baseline", "A2": "A2 second tuning", "Q": "Q damped logit inverse", "R1b": "R1b learned $G$, bounded (tuned)",
         "R4b": "R4b frozen random $G$, bounded", "R2q": "R2q oracle Q, unbounded", "R2qb": "R2qb oracle Q, bounded",
         "R2pb": "R2pb oracle P, bounded"}


def rows(name):
    return list(csv.DictReader(open(REPO / "results" / "processed" / name)))


def pct(r, t):
    v, lo, hi = float(r[f"change_pct_{t}"]), float(r[f"lo_{t}"]), float(r[f"hi_{t}"])
    return f"${v:+.1f}$ [${lo:+.1f}$, ${hi:+.1f}$]"


def label(group):
    key = group.split("-")[0]
    return LABEL.get(key, group)


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")
    print("wrote", OUT / name)


def table_b():
    p2, lines = rows("oracle_mb1_p2.csv"), []
    for setting in ("relu", "tanh"):
        lines.append(f"\\emph{{{setting}}} & & & \\\\")
        for r in p2:
            if r["setting"] != setting or r["group"].startswith("A") or not r.get("change_pct_0.97"):
                continue
            rho = f", $\\rho={float(r['rho']):g}$" if r["rho"] else ""
            lines.append(f"{label(r['group'])}{rho} & {pct(r, '0.97')} & {pct(r, '0.975')} & {int(r['n_reached_0.97'])}/{int(r['n_reached_0.975'])} \\\\")
    lines[-1] = lines[-1].rstrip(" \\")
    write("tab_oracle_b.tex", lines)


def table_d():
    d, lines = rows("oracle_md1_p1.csv"), []
    for r in d:
        if not r.get("change_pct_0.97"):
            continue
        sett = r["setting"] or "tanh"
        n = int(r["n"])
        lines.append(f"{label(r['group'])} ({sett}) & {int(r['n_reached_0.97'])}/{n} and {int(r['n_reached_0.975'])}/{n} & "
                     f"{pct(r, '0.97')} & {pct(r, '0.975')} & ${float(r['mean_final_acc']):.3f}$ \\\\")
    lines[-1] = lines[-1].rstrip(" \\")
    write("tab_oracle_d.tex", lines)


def table_b2():
    """mb2 (tanh): every group against its reference (matched for Q at lr x0.5)."""
    lines = []
    names = {"A-tanh-lr0p5": "A at lr $\\times0.5$ (against A)", "A2-tanh": "A2 second tuning (against A)", "F-tanh-rho1": "F$'$, $\\rho=1$ (against A)",
             "Q-tanh-rho0p3": "Q, $\\rho=0.3$ (against A)", "Q-tanh-rho0p3-lr0p5": "Q at lr $\\times0.5$ (against A at lr $\\times0.5$)"}
    by = {r["group"]: r for r in rows("oracle_mb2_p1.csv")}
    for g, lab in names.items():
        r = by[g]
        lines.append(f"{lab} & {pct(r, '0.97')} & {pct(r, '0.975')} \\\\")
    lines[-1] = lines[-1].rstrip(" \\")
    write("tab_oracle_b2.tex", lines)


def _flops(group, seeds):
    out = {t: [] for t in T}
    for sd in seeds:
        f = glob.glob(str(REPO / "results" / "raw" / "oracle" / "md2" / f"*orc-md2-{group}_*_s{sd}.jsonl"))
        r = [json.loads(l) for l in open(f[0])][1:]
        for t in T:
            i = crossing_index([x["val_acc"] for x in r], float(t), DEFAULT_CROSSING)
            out[t].append(r[-1]["flops"] if i is None else r[i]["flops"])
    return out


def table_d2():
    """md2 (relu, bounded): reached, steps and FLOPs to target against arm A, final accuracy. The oracle groups' FLOPs include the exact
    Jacobian computation and are not meaningful (dashes)."""
    d, lines = rows("oracle_md2_p1.csv"), []
    seeds = {r["group"]: list(range(930, 940)) for r in d}
    a = _flops("A-relu", seeds["A-relu"])
    for r in d:
        if not r.get("change_pct_0.97"):
            continue
        n = int(r["n"])
        if r["group"].startswith("R2"):
            fl = "-- & --"
        else:
            f = _flops(r["group"], seeds[r["group"]])
            c = {t: paired_pct_change(a[t], f[t], n_boot=10000) for t in T}
            fl = " & ".join(f"${c[t][0]:+.1f}$ [${c[t][1]:+.1f}$, ${c[t][2]:+.1f}$]" for t in T)
        lines.append(f"{label(r['group'])} & {int(r['n_reached_0.97'])}/{n} and {int(r['n_reached_0.975'])}/{n} & {pct(r, '0.97')} & "
                     f"{pct(r, '0.975')} & {fl} & ${float(r['mean_final_acc']):.3f}$ \\\\")
    lines[-1] = lines[-1].rstrip(" \\")
    write("tab_oracle_d2.tex", lines)


if __name__ == "__main__":
    table_b()
    table_d()
    table_b2()
    table_d2()

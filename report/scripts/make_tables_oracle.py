"""LaTeX table bodies for the root-cause checks B and D (chapter 6), computed from results/processed/oracle_*.csv, which
scripts/run_root_cause.py writes from the raw logs. Run: python make_tables_oracle.py   (from report/scripts)"""
import csv
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parents[1] / "tables"
T = ("0.97", "0.975")

LABEL = {"F": "F$'$ exact gradient (rescaled)", "P": "P pseudo-inverse", "Q": "Q damped logit inverse", "Pr": "Pr P, radially bounded",
         "D": "D learned $G$ (tuned, descriptive)", "R1": "R1 learned $G$, cut at $h$", "R1w": "R1w as R1 after 300 backprop steps",
         "R2": "R2 oracle P, cut at $h$", "R4": "R4 frozen random $G$, cut at $h$"}


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


if __name__ == "__main__":
    table_b()
    table_d()

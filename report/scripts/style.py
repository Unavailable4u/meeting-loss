"""Common matplotlib style: Okabe-Ito colour-blind-safe palette, LaTeX fonts, thin axes, vector PDF output."""
import matplotlib as mpl
import matplotlib.pyplot as plt

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", purple="#CC79A7", grey="#7F7F7F")

# one colour per arm, used consistently in every figure of the thesis
ARM = {
    "A": OI["black"], "A1": OI["black"], "A2": OI["grey"],
    "B": OI["sky"], "D": OI["vermillion"], "C": OI["purple"],
    "S30": OI["blue"], "S20": OI["sky"], "S20w0": OI["green"], "S15": OI["orange"], "S20w10": OI["purple"],
    "SBT": OI["vermillion"], "R20w0": OI["grey"], "R30": OI["grey"], "B48": OI["orange"],
}


def setup():
    mpl.rcParams.update({
        "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"], "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False,
    })


def panel_label(ax, s, dx=-0.12, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")

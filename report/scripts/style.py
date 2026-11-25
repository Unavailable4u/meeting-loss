"""Common matplotlib style: Okabe-Ito colour-blind-safe palette, LaTeX fonts, thin axes, vector PDF output."""
import matplotlib as mpl
import matplotlib.pyplot as plt

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", purple="#CC79A7", grey="#7F7F7F")

# one colour per arm, used consistently in every figure of the report
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


def mean_marker(ax, x, color, y=0.0, size=5.5):
    """Triangle on the lower axis edge: where the mean cost to the target lies (per-seed scoring rule)."""
    ax.plot(x, y, "^", color=color, ms=size, mec="white", mew=0.5, clip_on=False, zorder=6,
            transform=ax.get_xaxis_transform())


def mean_marker_handle(label="mean cost to 0.975 (per-seed rule)"):
    from matplotlib.lines import Line2D
    return Line2D([], [], marker="^", ls="", color="#555555", mec="white", mew=0.5, ms=5.5, label=label)


def legend_below(fig_or_ax, handles=None, labels=None, ncol=3, y=-0.02, **kw):
    """Legend that sits below the plot area, so that it can never cover data."""
    if hasattr(fig_or_ax, "legend") and not hasattr(fig_or_ax, "get_xaxis_transform"):
        return fig_or_ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, **kw)
    return fig_or_ax.legend(handles=handles, labels=labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, **kw)

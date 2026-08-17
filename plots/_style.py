"""Shared figure style + palette for the OccFPGA paper figures (cohesive, publication-grade)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- palette (muted, harmonious) ---
GRAY    = "#BFC6D1"   # FP32 / reference (cannot run on DPU -> kept muted/background)
AMBER   = "#E4A14B"   # INT8 algorithm (the achievable target)
TEAL    = "#1F9E89"   # on-board INT8 (this work -> the hero color)
TEAL_LT = "#8ED0C6"
CORAL   = "#E8684A"   # co-tenant load / GPU loaded
ORANGE  = "#F2A65A"
GOLD    = "#E9C46A"
CRIMSON = "#B5384A"   # workstation GPU
PLUM    = "#8E6FAE"
INK     = "#22303C"   # text
SUBTLE  = "#5F6B76"

def apply():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 220,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12.5, "axes.titleweight": "semibold", "axes.titlepad": 9,
        "axes.titlecolor": INK,
        "axes.labelsize": 10.5, "axes.labelcolor": INK, "axes.labelweight": "medium",
        "axes.edgecolor": "#AEB6BF", "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#EAEDF0", "grid.linewidth": 1.0,
        "xtick.color": SUBTLE, "ytick.color": SUBTLE,
        "xtick.labelsize": 10, "ytick.labelsize": 9.5,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 9, "legend.handlelength": 1.4,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "text.color": INK,
    })

def bar_labels(ax, bars, fmt="%.1f", dy=0.4, fs=8.5, color=INK, weight="medium"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + dy, fmt % h,
                ha="center", va="bottom", fontsize=fs, color=color, weight=weight)

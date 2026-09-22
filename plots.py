#!/usr/bin/env python3
"""Generate the report figures from saved run histories.

    python3 plots.py                                  # every figure, from runs/gpu_arch
    python3 plots.py --only depth width               # just the two required figures
    python3 plots.py --experiment selfdirected        # a different experiment

Each figure summarises the five replicate seeds of a condition as a mean curve
with a +/- 1 standard deviation band, never a single seed. Output goes to plots/
as both PNG (for viewing) and PDF (vector, for the report).
"""

import argparse
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import aggregate

# --------------------------------------------------------------------------- #
# Palette
#
# Depth and width are ORDERED quantities, so each is encoded as a single-hue
# ordinal ramp running light -> dark with increasing depth/width, rather than as
# unrelated categorical hues. Steps are taken from the blue sequential ramp and
# validated as ordinal ramps on the light chart surface: monotone lightness,
# adjacent lightness gaps >= 0.06, light end clearing 2:1 against the surface
# (#86b6ef at 2.06:1), single hue (4 degree spread).
# --------------------------------------------------------------------------- #

RAMP_3 = ["#86b6ef", "#2a78d6", "#0d366b"]
RAMP_4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

DEPTHS = [1, 2, 3]
WIDTHS = [32, 64, 128, 256]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
        "font.size": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.direction": "out", "ytick.direction": "out",
        "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "figure.dpi": 150, "savefig.bbox": "tight",
    })


# Per-epoch evaluation error is noisy at this batch size, and that noise is much
# larger than the spread between seeds. Each seed's curve is smoothed with a
# centred moving average BEFORE averaging across seeds, so the band shows
# seed-to-seed variability rather than epoch-to-epoch jitter. The unsmoothed mean
# is always drawn faintly behind it, so the smoothing never hides the real signal.
SMOOTH_WINDOW = 25


def smooth(curve, window=SMOOTH_WINDOW):
    """Centred moving average with a window that shrinks at the edges."""
    if window <= 1 or curve.size < 3:
        return curve
    cumulative = np.concatenate([[0.0], np.cumsum(curve, dtype=np.float64)])
    idx = np.arange(curve.size)
    lo = np.maximum(0, idx - window // 2)
    hi = np.minimum(curve.size, idx + window // 2 + 1)
    return (cumulative[hi] - cumulative[lo]) / (hi - lo)


def smoothed_band(cond, key="eval_all"):
    """Mean and +/- 1 SD across seeds, computed on per-seed smoothed curves."""
    stack = np.vstack([smooth(row) for row in cond[key]])
    std = stack.std(0, ddof=1) if stack.shape[0] > 1 else np.zeros(stack.shape[1])
    return stack.mean(0), std


def log_ticks(ax, values=None):
    """Plain decimal tick labels on a log axis, instead of 4 x 10^0 notation."""
    lo, hi = ax.get_ylim()          # the drawn range, not just the data range
    candidates = [1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100]
    ticks = [c for c in candidates if lo <= c <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{c:g}" for c in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def grid_on(ax, axis="both"):
    ax.grid(True, axis=axis, alpha=1.0, zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_conditions(runs_root, experiment):
    """Group runs by condition and stack their per-epoch histories over seeds."""
    rows = aggregate.collect(runs_root, experiment)
    if not rows:
        raise SystemExit(f"no runs found under {runs_root} for experiment {experiment!r}")

    conditions = {}
    for row in rows:
        if row["status"] != "completed":
            print(f"  skipping {row['run_name']}: status {row['status']}", file=sys.stderr)
            continue
        key = aggregate.condition_key(row)
        conditions.setdefault(key, {"rows": [], "curves": [], "train_curves": []})
        history = np.load(pathlib.Path(row["run_dir"]) / "history.npz")
        conditions[key]["rows"].append(row)
        conditions[key]["curves"].append(history["eval_rmse_liters"])
        conditions[key]["train_curves"].append(history["train_rmse_liters"])

    out = {}
    for key, value in conditions.items():
        cfg = dict(zip(aggregate.CONDITION_KEYS, key))
        # Truncate to the shortest history so a stopped run cannot skew the band.
        n = min(len(c) for c in value["curves"])
        evl = np.vstack([c[:n] for c in value["curves"]])
        trn = np.vstack([c[:n] for c in value["train_curves"]])
        best = np.array([float(r["best_eval_rmse_liters"]) for r in value["rows"]])
        out[key] = {
            "cfg": cfg,
            "depth": int(cfg["hidden_layers"]), "width": int(cfg["neurons"]),
            "n_seeds": len(value["rows"]),
            "epochs": np.arange(n),
            "eval_all": evl, "train_all": trn,
            "eval_mean": evl.mean(0), "eval_std": evl.std(0, ddof=1) if len(evl) > 1 else np.zeros(n),
            "train_mean": trn.mean(0),
            "best_mean": best.mean(), "best_std": best.std(ddof=1) if len(best) > 1 else 0.0,
            "best_epoch_mean": np.mean([float(r["best_epoch"]) for r in value["rows"]]),
            "best_epochs": np.array([float(r["best_epoch"]) for r in value["rows"]]),
            "time_mean": np.mean([float(r["train_time_s"]) for r in value["rows"]]),
            "time_std": (np.std([float(r["train_time_s"]) for r in value["rows"]], ddof=1)
                         if len(value["rows"]) > 1 else 0.0),
            "params": int(value["rows"][0]["n_parameters"]),
        }
    return out


def pick(conditions, depth=None, width=None):
    found = [c for c in conditions.values()
             if (depth is None or c["depth"] == depth) and (width is None or c["width"] == width)]
    return sorted(found, key=lambda c: (c["depth"], c["width"]))


def save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {out_dir / name}.png / .pdf")


def band_plot(ax, series, colors, labels, label_fmt="{}"):
    """Mean curve with a +/- 1 std band per series, plus direct end labels."""
    ends = []
    allvals = []
    for cond, color, label in zip(series, colors, labels):
        x = cond["epochs"]
        mean, std = smoothed_band(cond)
        ax.plot(x, cond["eval_mean"], color=color, alpha=0.25, linewidth=0.7, zorder=1)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.30, linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, label=label, zorder=3)
        ends.append((mean[-1], color, label))
        allvals += [mean.min(), cond["eval_mean"].max()]
    log_ticks(ax, allvals)

    # Direct labels at the right end, nudged apart if they would collide.
    ends.sort(key=lambda e: e[0])
    ymin, ymax = ax.get_ylim()
    span = np.log10(ymax) - np.log10(ymin) if ax.get_yscale() == "log" else ymax - ymin
    pos = lambda v: np.log10(v) if ax.get_yscale() == "log" else v
    # Computed directly rather than with a nudge loop: `(a + gap) - a < gap` can
    # be true in floating point when a >> gap, which makes such a loop spin forever.
    gap = 0.055 * span
    placed = []
    for value, color, label in ends:
        p = pos(value)
        if placed:
            p = max(p, placed[-1] + gap)
        placed.append(p)
        y = 10 ** p if ax.get_yscale() == "log" else p
        ax.annotate(label_fmt.format(label), xy=(series[0]["epochs"][-1], y),
                    xytext=(6, 0), textcoords="offset points", color=color,
                    fontsize=8.5, fontweight="bold", va="center", clip_on=False)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def fig_depth(conditions, out_dir, width=256):
    series = pick(conditions, width=width)
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    grid_on(ax)
    ax.set_yscale("log")
    band_plot(ax, series, RAMP_3, [f"{c['depth']} hidden layer" + ("s" if c["depth"] > 1 else "")
                                   for c in series])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Evaluation RMSE (liters)")
    ax.set_title(f"Deeper networks converge lower at every epoch\n"
                 f"{width} neurons per hidden layer; mean of 5 seeds, band = +/- 1 SD\n"
                 f"bold = {SMOOTH_WINDOW}-epoch moving average, faint = unsmoothed mean",
                 loc="left")
    ax.set_xlim(0, series[0]["epochs"][-1])
    ax.legend(loc="upper right", ncols=3)
    fig.subplots_adjust(right=0.84)
    save(fig, out_dir, "eval_rmse_vs_epoch_depth")


def fig_width(conditions, out_dir, depth=3):
    series = pick(conditions, depth=depth)
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    grid_on(ax)
    ax.set_yscale("log")
    band_plot(ax, series, RAMP_4, [f"{c['width']} neurons" for c in series])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Evaluation RMSE (liters)")
    ax.set_title(f"Wider hidden layers converge lower, with diminishing returns\n"
                 f"{depth} hidden layers; mean of 5 seeds, band = +/- 1 SD\n"
                 f"bold = {SMOOTH_WINDOW}-epoch moving average, faint = unsmoothed mean",
                 loc="left")
    ax.set_xlim(0, series[0]["epochs"][-1])
    ax.legend(loc="upper right", ncols=4)
    fig.subplots_adjust(right=0.84)
    save(fig, out_dir, "eval_rmse_vs_epoch_width")


def fig_grid(conditions, out_dir):
    """Small multiples: every condition, one panel per depth."""
    fig, axes = plt.subplots(1, len(DEPTHS), figsize=(11.5, 3.9), sharey=True)
    for ax, depth in zip(np.atleast_1d(axes), DEPTHS):
        series = pick(conditions, depth=depth)
        if not series:
            continue
        grid_on(ax)
        ax.set_yscale("log")
        for cond, color in zip(series, RAMP_4):
            mean, std = smoothed_band(cond)
            ax.plot(cond["epochs"], cond["eval_mean"], color=color, alpha=0.22,
                    linewidth=0.6, zorder=1)
            ax.fill_between(cond["epochs"], mean - std, mean + std, color=color,
                            alpha=0.30, linewidth=0, zorder=2)
            ax.plot(cond["epochs"], mean, color=color,
                    label=f"{cond['width']} neurons", zorder=3)
        log_ticks(ax)
        ax.set_title(f"{depth} hidden layer" + ("s" if depth > 1 else ""), loc="left")
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, series[0]["epochs"][-1])
    np.atleast_1d(axes)[0].set_ylabel("Evaluation RMSE (liters)")
    np.atleast_1d(axes)[-1].legend(loc="upper right")
    fig.suptitle("Every condition is still improving at the final epoch",
                 x=0.005, ha="left", fontsize=11, fontweight="bold", color=INK)
    save(fig, out_dir, "eval_rmse_grid")


def fig_params(conditions, out_dir):
    """Best evaluation RMSE against parameter count, to separate depth from capacity."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    grid_on(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for depth, color in zip(DEPTHS, RAMP_3):
        series = pick(conditions, depth=depth)
        if not series:
            continue
        x = [c["params"] for c in series]
        y = [c["best_mean"] for c in series]
        ax.errorbar(x, y, yerr=[c["best_std"] for c in series], color=color,
                    marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    capsize=3, elinewidth=1.2,
                    label=f"{depth} hidden layer" + ("s" if depth > 1 else ""), zorder=3)
        for cond in series:
            ax.annotate(f"{cond['width']}", xy=(cond["params"], cond["best_mean"]),
                        xytext=(0, -12), textcoords="offset points", ha="center",
                        fontsize=7.5, color=MUTED)
    log_ticks(ax)
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Best evaluation RMSE (liters)")
    ax.set_title("Depth buys more than width, but the gain saturates after two layers\n"
                 "point labels give neurons per hidden layer; bars = +/- 1 SD over 5 seeds",
                 loc="left")
    ax.legend(loc="upper right")
    save(fig, out_dir, "best_rmse_vs_parameters")


def fig_time(conditions, out_dir):
    """Training time by architecture."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    grid_on(ax, axis="y")
    n = len(DEPTHS)
    bar_w = 0.8 / n
    for i, (depth, color) in enumerate(zip(DEPTHS, RAMP_3)):
        series = {c["width"]: c for c in pick(conditions, depth=depth)}
        xs = np.arange(len(WIDTHS)) + (i - (n - 1) / 2) * bar_w
        ys = [series[w]["time_mean"] if w in series else 0 for w in WIDTHS]
        es = [series[w]["time_std"] if w in series else 0 for w in WIDTHS]
        ax.bar(xs, ys, bar_w * 0.9, yerr=es, color=color, capsize=2.5,
               error_kw={"elinewidth": 1.0, "ecolor": INK_2},
               edgecolor=SURFACE, linewidth=1.0,
               label=f"{depth} hidden layer" + ("s" if depth > 1 else ""), zorder=3)
    ax.set_xticks(np.arange(len(WIDTHS)), [str(w) for w in WIDTHS])
    ax.set_xlabel("Neurons per hidden layer")
    ax.set_ylabel("Training time (s, 1000 epochs)")
    ax.set_title("Training time tracks depth, and almost ignores width\n"
                 "264x more parameters costs only 1.7x more time: the cost here is\n"
                 "per-layer kernel launches, not arithmetic. Bars = +/- 1 SD over 5 seeds.",
                 loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncols=3, borderaxespad=0)
    save(fig, out_dir, "train_time_vs_architecture")


def fig_best_epoch(conditions, out_dir):
    """Diagnostic: where in training the best evaluation score occurred.

    A dot plot rather than bars. Every value sits within a few percent of the
    final epoch, so the informative range is narrow; bars encode magnitude from
    zero and must not be truncated, while dots read correctly on a zoomed axis.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    grid_on(ax, axis="y")

    total_epochs = max(len(c["epochs"]) for c in conditions.values())
    offsets = np.linspace(-0.22, 0.22, len(DEPTHS))
    for i, (depth, color) in enumerate(zip(DEPTHS, RAMP_3)):
        series = {c["width"]: c for c in pick(conditions, depth=depth)}
        for j, width in enumerate(WIDTHS):
            cond = series.get(width)
            if cond is None:
                continue
            x = j + offsets[i]
            # individual seeds, then the condition mean on top
            ax.plot(np.full(cond["best_epochs"].shape, x), cond["best_epochs"],
                    linestyle="none", marker="o", markersize=4, color=color,
                    alpha=0.45, markeredgewidth=0, zorder=3)
            ax.plot([x], [cond["best_epoch_mean"]], marker="o", markersize=9,
                    color=color, markeredgecolor=SURFACE, markeredgewidth=1.4,
                    linestyle="none", zorder=4,
                    label=(f"{depth} hidden layer" + ("s" if depth > 1 else "")) if j == 0 else None)

    ax.axhline(total_epochs - 1, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(f"final epoch ({total_epochs - 1})", xy=(len(WIDTHS) - 0.5, total_epochs - 1),
                xytext=(0, 4), textcoords="offset points", ha="right",
                fontsize=8, color=INK_2)

    lowest = min(c["best_epochs"].min() for c in conditions.values())
    ax.set_ylim(lowest - 15, total_epochs + 12)
    ax.set_xticks(np.arange(len(WIDTHS)), [str(w) for w in WIDTHS])
    ax.set_xlim(-0.5, len(WIDTHS) - 0.5)
    ax.set_xlabel("Neurons per hidden layer")
    ax.set_ylabel("Epoch of best evaluation RMSE")
    ax.set_title("Every run peaks in the last few percent of training\n"
                 "large dot = mean of 5 seeds, small dots = individual seeds; "
                 "note the zoomed vertical axis", loc="left")
    ax.legend(loc="lower left", ncols=3)
    save(fig, out_dir, "best_epoch_by_condition")


def fig_train_eval(conditions, out_dir):
    """Diagnostic: training versus evaluation error, to rule overfitting in or out."""
    picked = [c for c in [next((x for x in conditions.values()
                                if x["depth"] == d and x["width"] == w), None)
                          for d, w in ((1, 32), (2, 128), (3, 256))] if c]
    if not picked:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    grid_on(ax)
    ax.set_yscale("log")
    for cond, color in zip(picked, RAMP_3):
        ev, _ = smoothed_band(cond, "eval_all")
        tr, _ = smoothed_band(cond, "train_all")
        ax.plot(cond["epochs"], ev, color=color, zorder=3)
        ax.plot(cond["epochs"], tr, color=color, linewidth=1.4,
                linestyle=(0, (4, 2.5)), zorder=3)
    log_ticks(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (liters)")
    ax.set_xlim(0, picked[0]["epochs"][-1])
    ax.set_title("Evaluation error never turns up, but the gap grows with capacity\n"
                 f"solid = evaluation, dashed = training; mean of 5 seeds, "
                 f"{SMOOTH_WINDOW}-epoch moving average", loc="left")
    handles = [Line2D([], [], color=c, label=f"{p['depth']} x {p['width']}")
               for p, c in zip(picked, RAMP_3)]
    handles += [Line2D([], [], color=MUTED, label="evaluation"),
                Line2D([], [], color=MUTED, linestyle=(0, (4, 2.5)), linewidth=1.4, label="training")]
    ax.legend(handles=handles, loc="upper right", ncols=2)
    save(fig, out_dir, "train_vs_eval_rmse")


FIGURES = {
    "depth": fig_depth, "width": fig_width, "grid": fig_grid,
    "params": fig_params, "time": fig_time,
    "bestepoch": fig_best_epoch, "trainval": fig_train_eval,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("runs"))
    ap.add_argument("--experiment", default="gpu_arch")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("plots"))
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES),
                    help="generate only these figures (default: all)")
    ap.add_argument("--depth-fixed-width", type=int, default=256,
                    help="width held fixed in the depth figure (default: 256)")
    ap.add_argument("--width-fixed-depth", type=int, default=3,
                    help="depth held fixed in the width figure (default: 3)")
    args = ap.parse_args(argv)

    style()
    conditions = load_conditions(args.runs, args.experiment)
    print(f"{len(conditions)} conditions from {args.runs}/{args.experiment}")

    for name in (args.only or sorted(FIGURES)):
        if name == "depth":
            fig_depth(conditions, args.out, width=args.depth_fixed_width)
        elif name == "width":
            fig_width(conditions, args.out, depth=args.width_fixed_depth)
        else:
            FIGURES[name](conditions, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

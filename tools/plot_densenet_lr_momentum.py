from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = ROOT / "logs" / "ray_results" / "densenet-fashion-mnist"
OUTPUT_PATH = EXPERIMENT_ROOT / "lr_momentum_distribution.png"
TOP_RUNS_OUTPUT_PATH = EXPERIMENT_ROOT / "lr_momentum_distribution_top40.png"
PEAK_OUTPUT_PATH = EXPERIMENT_ROOT / "lr_momentum_distribution_peak_val_acc.png"
TOP_RUNS_PEAK_OUTPUT_PATH = EXPERIMENT_ROOT / "lr_momentum_distribution_top40_peak_val_acc.png"
TOP_RUNS_JSON_PATH = EXPERIMENT_ROOT / "analysis_25epochs_valacc_gt_0p925.json"
COLOR_RANGE = (0.92, 0.94)

TRIAL_PATTERN = re.compile(
    r"batch_size=(?P<batch_size>[^,]+),"
    r"cosine_annealing_t_max=(?P<tmax>[^,]+),"
    r"lr=(?P<lr>[^,]+),"
    r"momentum=(?P<momentum>[^_]+)_"
)


def load_val_acc_stats(progress_path: Path) -> tuple[float | None, float | None]:
    try:
        with progress_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None, None

    if not rows:
        return None, None

    vals = [float(row["val_acc"]) for row in rows if row.get("val_acc") not in (None, "")]
    if not vals:
        return None, None
    return vals[-1], max(vals)


def collect_trials() -> list[dict[str, float | str | None]]:
    trials: list[dict[str, float | str | None]] = []
    for trial_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith("trainable_"):
            continue

        match = TRIAL_PATTERN.search(trial_dir.name)
        if not match:
            continue

        progress_path = trial_dir / "progress.csv"
        final_val_acc, peak_val_acc = load_val_acc_stats(progress_path)
        trials.append(
            {
                "trial_dir": trial_dir.name,
                "lr": float(match.group("lr")),
                "momentum": float(match.group("momentum")),
                "final_val_acc": final_val_acc,
                "peak_val_acc": peak_val_acc,
            }
        )

    return trials


def collect_top_runs() -> list[dict[str, float | str | None]]:
    payload = json.loads(TOP_RUNS_JSON_PATH.read_text())
    trials: list[dict[str, float | str | None]] = []
    for run in payload["runs"]:
        match = TRIAL_PATTERN.search(str(run["trial_dir"]))
        if not match:
            continue
        trials.append(
            {
                "trial_dir": run["trial_dir"],
                "lr": float(match.group("lr")),
                "momentum": float(match.group("momentum")),
                "final_val_acc": float(run["final_val_acc"]),
                "peak_val_acc": max(float(metric["val_acc"]) for metric in run["metrics"]),
            }
        )
    return trials


def plot_trials(
    trials: list[dict[str, float | str | None]],
    output_path: Path,
    title: str,
    color_key: str,
    color_label: str,
    color_limits: tuple[float, float] | None = None,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> plt.Figure:
    lrs = [float(trial["lr"]) for trial in trials]
    momentums = [float(trial["momentum"]) for trial in trials]
    color_values = [trial[color_key] for trial in trials]

    missing_color = "#b0b0b0"
    cmap = plt.cm.viridis
    valid_vals = [value for value in color_values if value is not None]

    fig, ax = plt.subplots(figsize=(10, 7))
    if valid_vals:
        scatter = ax.scatter(
            [lr for lr, value in zip(lrs, color_values) if value is not None],
            [momentum for momentum, value in zip(momentums, color_values) if value is not None],
            c=[float(value) for value in color_values if value is not None],
            cmap=cmap,
            vmin=color_limits[0] if color_limits is not None else None,
            vmax=color_limits[1] if color_limits is not None else None,
            s=50,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.25,
        )
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label(color_label)

    if any(value is None for value in color_values):
        ax.scatter(
            [lr for lr, value in zip(lrs, color_values) if value is None],
            [momentum for momentum, value in zip(momentums, color_values) if value is None],
            color=missing_color,
            s=50,
            alpha=0.65,
            edgecolors="black",
            linewidths=0.25,
            label=f"Missing {color_label}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Momentum")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)

    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    if any(value is None for value in color_values):
        ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    return fig


def main(*, show: bool = False) -> None:
    trials = collect_trials()
    if not trials:
        raise SystemExit("No DenseNet trials with momentum were found.")

    x_limits = (min(float(trial["lr"]) for trial in trials), max(float(trial["lr"]) for trial in trials))
    y_limits = (min(float(trial["momentum"]) for trial in trials), max(float(trial["momentum"]) for trial in trials))
    figures = []
    figures.append(
        plot_trials(
            trials,
            OUTPUT_PATH,
            f"DenseNet Ray Tune: LR vs Momentum ({len(trials)} trials with momentum)",
            color_key="final_val_acc",
            color_label="Final val_acc",
            color_limits=COLOR_RANGE,
            x_limits=x_limits,
            y_limits=y_limits,
        )
    )
    figures.append(
        plot_trials(
            trials,
            PEAK_OUTPUT_PATH,
            f"DenseNet Ray Tune: LR vs Momentum ({len(trials)} trials, peak val_acc)",
            color_key="peak_val_acc",
            color_label="Peak val_acc",
            color_limits=COLOR_RANGE,
            x_limits=x_limits,
            y_limits=y_limits,
        )
    )

    top_runs = collect_top_runs()
    if not top_runs:
        raise SystemExit("No filtered top runs were found.")
    figures.append(
        plot_trials(
            top_runs,
            TOP_RUNS_OUTPUT_PATH,
            f"DenseNet Ray Tune: LR vs Momentum ({len(top_runs)} filtered top runs)",
            color_key="final_val_acc",
            color_label="Final val_acc",
            color_limits=COLOR_RANGE,
            x_limits=x_limits,
            y_limits=y_limits,
        )
    )
    figures.append(
        plot_trials(
            top_runs,
            TOP_RUNS_PEAK_OUTPUT_PATH,
            f"DenseNet Ray Tune: LR vs Momentum ({len(top_runs)} top runs, peak val_acc)",
            color_key="peak_val_acc",
            color_label="Peak val_acc",
            color_limits=COLOR_RANGE,
            x_limits=x_limits,
            y_limits=y_limits,
        )
    )

    print(f"Saved plot to {OUTPUT_PATH}")
    print(f"Saved plot to {PEAK_OUTPUT_PATH}")
    print(f"Saved plot to {TOP_RUNS_OUTPUT_PATH}")
    print(f"Saved plot to {TOP_RUNS_PEAK_OUTPUT_PATH}")
    print(f"Plotted {len(trials)} trials with lr and momentum")
    print(f"Plotted {len(top_runs)} filtered top runs with matching axes")
    print(f"Used shared accuracy color range {COLOR_RANGE}")

    if show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main(show=True)
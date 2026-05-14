from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import STAGE_COLUMNS, load_metrics, prepend_cpu_hd_encode_eta


FIGSIZE_WALL_TIME = (11.0, 6.0)
FIGSIZE_HEATMAP = (11.0, 4.8)
SYSTEM_SPECS = (
    "Server: Russell\n"
    "CPU: AMD Ryzen Threadripper 9985WX, 64C/128T, up to 5.4GHz\n"
    "Memory: 8x 96GB DDR5-6400 ECC (768GB)\n"
    "GPU: 4x NVIDIA RTX PRO 6000 Blackwell 96GB GDDR7 Max-Q, 300W TDP"
)
MULTI_GPU_NOTE = "Multi-GPU runs used 3 GPUs; 1 GPU was busy."
COLORS = {
    "multi": "#2f6f9f",
    "single": "#c65d3a",
    "grid": "#d7dde5",
    "text": "#20242a",
    "muted": "#68717d",
}
FAMILY_TITLES = {
    "multi": "Multi-GPU Optimization Progression",
    "single": "Single-GPU Optimization Progression",
}


def generate_plots(
    metrics_path: Path, out_dir: Path, output_format: str = "png"
) -> tuple[list[Path], list[dict[str, float | int | str]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated, wall_speedups, stage_speedups = load_metrics(metrics_path)
    gpu_hd_dir = out_dir / "gpu_hd_encode_baseline"
    cpu_hd_dir = out_dir / "cpu_hd_encode_baseline"
    gpu_hd_dir.mkdir(parents=True, exist_ok=True)
    cpu_hd_dir.mkdir(parents=True, exist_ok=True)
    cpu_hd_speedups = prepend_cpu_hd_encode_eta(wall_speedups)

    outputs = [
        plot_family_wall_time(wall_speedups, "multi", gpu_hd_dir, output_format),
        plot_family_wall_time(wall_speedups, "single", gpu_hd_dir, output_format),
        plot_family_stage_heatmap(stage_speedups, "multi", gpu_hd_dir, output_format),
        plot_family_stage_heatmap(stage_speedups, "single", gpu_hd_dir, output_format),
        plot_family_wall_time(cpu_hd_speedups, "multi", cpu_hd_dir, output_format),
        plot_family_wall_time(cpu_hd_speedups, "single", cpu_hd_dir, output_format),
    ]

    summary = build_summary("gpu_hd_encode_baseline", wall_speedups)
    summary.extend(build_summary("cpu_hd_encode_baseline", cpu_hd_speedups))
    return outputs, summary


def plot_family_wall_time(
    speedups: pd.DataFrame, family: str, out_dir: Path, output_format: str
) -> Path:
    data = speedups[speedups["family"] == family].sort_values("order")
    if data.empty:
        raise ValueError(f"No {family} runs found")

    x = np.arange(len(data))
    y = data["wall_s"].to_numpy()
    color = COLORS[family]

    fig, ax = plt.subplots(figsize=FIGSIZE_WALL_TIME)
    ax.plot(x, y, color=color, linewidth=2.2, marker="o", markersize=7)
    ax.vlines(x, ymin=0, ymax=y, color=color, alpha=0.25, linewidth=4)

    for i, row in enumerate(data.to_dict("records")):
        if int(row["run_count"]) > 1:
            low = float(row["wall_s_min"])
            high = float(row["wall_s_max"])
            ax.vlines(i, low, high, color=COLORS["text"], linewidth=1.4)
            ax.hlines([low, high], i - 0.08, i + 0.08, color=COLORS["text"], linewidth=1.4)

        ax.annotate(
            f"{format_seconds(float(row['wall_s']))}\n{row['wall_speedup']:.2f}x",
            (i, float(row["wall_s"])),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS["text"],
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )

    ax.set_title(FAMILY_TITLES[family], loc="left", fontsize=15, fontweight="bold")
    ax.set_ylabel("Wall-clock time (seconds)")
    ax.set_xticks(x, [wrap_label(label) for label in data["stage_label"]])
    ax.tick_params(axis="x", labelrotation=28, pad=8)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    ax.set_ylim(0, float(y.max()) * 1.2)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    despine(ax)
    add_system_specs_note(ax, family)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.3)

    return save_fig(fig, out_dir / f"{family}_gpu_speedup.{output_format}")


def plot_family_stage_heatmap(
    stage_speedups: pd.DataFrame, family: str, out_dir: Path, output_format: str
) -> Path:
    data = stage_speedups[stage_speedups["family"] == family].copy()
    if data.empty:
        raise ValueError(f"No {family} stage speedups found")

    pivot = data.pivot(index="metric_label", columns="stage_label", values="speedup")
    pivot = pivot.loc[list(STAGE_COLUMNS.keys())]
    stage_order = (
        data[["stage_label", "order"]]
        .drop_duplicates()
        .sort_values("order")["stage_label"]
        .tolist()
    )
    pivot = pivot[stage_order]

    values = pivot.to_numpy(dtype=float)
    log_values = np.log2(values)
    finite_abs = np.abs(log_values[np.isfinite(log_values)])
    limit = max(1.0, float(finite_abs.max())) if len(finite_abs) else 1.0

    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#e9edf2")
    image = ax.imshow(
        np.ma.masked_invalid(log_values),
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )

    ax.set_title(
        f"{FAMILY_TITLES[family]}: Stage Speedups",
        loc="left",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xticks(np.arange(len(pivot.columns)), [wrap_label(label) for label in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), list(pivot.index))
    ax.tick_params(axis="x", labelrotation=0)

    for row_index, metric_label in enumerate(pivot.index):
        for col_index, stage_label in enumerate(pivot.columns):
            value = float(pivot.loc[metric_label, stage_label])
            if not np.isfinite(value):
                label = "NA"
                text_color = COLORS["muted"]
            else:
                label = f"{value:.2f}x"
                text_color = "white" if abs(float(log_values[row_index, col_index])) > limit * 0.58 else COLORS["text"]
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    cbar = fig.colorbar(image, ax=ax, shrink=0.88)
    cbar.set_label("log2 speedup vs family baseline")
    fig.subplots_adjust(left=0.13, right=0.94, top=0.88, bottom=0.12)

    return save_fig(fig, out_dir / f"{family}_gpu_stage_speedups.{output_format}")


def build_summary(
    baseline: str, wall_speedups: pd.DataFrame
) -> list[dict[str, float | int | str]]:
    summaries: list[dict[str, float | int | str]] = []
    for row in wall_speedups.sort_values(["family", "order"]).to_dict("records"):
        summaries.append(
            {
                "baseline": baseline,
                "family": row["family"],
                "stage": row["stage_label"],
                "wall_s": float(row["wall_s"]),
                "wall_speedup": float(row["wall_speedup"]),
                "run_count": int(row["run_count"]),
            }
        )
    return summaries


def save_fig(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def format_seconds(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}s"
    if value >= 100:
        return f"{value:.0f}s"
    if value >= 10:
        return f"{value:.1f}s"
    return f"{value:.2f}s"


def wrap_label(value: str) -> str:
    return (
        value.replace(" + ", "\n+ ")
        .replace("copy removal", "copy\nremoval")
        .replace("Old hashset", "Old\nhashset")
    )


def add_system_specs_note(ax: plt.Axes, family: str) -> None:
    note = f"{SYSTEM_SPECS}"
    if family == "multi":
        note = f"{note}\n{MULTI_GPU_NOTE}"
    ax.text(
        0.985,
        0.985,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=COLORS["muted"],
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": COLORS["grid"],
            "linewidth": 0.6,
            "alpha": 0.9,
        },
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

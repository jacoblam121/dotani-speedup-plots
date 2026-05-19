from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


METRIC_RE = re.compile(r"^([A-Za-z0-9_]+_s)\s+([0-9]+(?:\.[0-9]+)?)$")

FAMILY_ORDER = {
    "multi": [
        "multi_hashset",
        "multi_sort",
        "multi_sort_scratchreuse",
        "multi_sort_scratchreuse_copy",
    ],
    "single": [
        "single_hashset",
        "single_sort",
        "single_sort_scratchreuse_copy",
    ],
}

STAGE_COLUMNS = {
    "Wall": "wall_s",
    "FASTA": "fasta_s",
    "Dedup/filter": "dedup_s",
    "HD encode": "hd_encode_s",
    "K-mer H2D": "cuda_h2d_s",
    "K-mer alloc": "cuda_alloc_s",
    "K-mer launch": "cuda_kmer_launch_s",
    "K-mer D2H": "cuda_d2h_s",
    "HD hash H2D": "cuda_hd_hash_h2d_s",
    "HD HV H2D": "cuda_hd_hv_h2d_s",
    "HD alloc": "cuda_hd_alloc_s",
    "HD kernel": "cuda_hd_kernel_s",
    "HD D2H": "cuda_hd_d2h_s",
}

CPU_HD_ENCODE_ETA_S = 110 * 60

SUMMARY_TSV_COLUMNS = {
    "wall_s": "sketch_wall_ns",
    "fasta_s": "fasta_ns",
    "dedup_s": "hash_and_dedup_ns",
    "hd_encode_s": "hd_encode_ns",
    "hv_norm_s": "hv_norm_ns",
    "compress_s": "hd_compress_ns",
    "worker_total_s": "total_worker_ns",
    "cuda_h2d_s": "cuda_h2d_ns",
    "cuda_alloc_s": "cuda_alloc_ns",
    "cuda_kmer_launch_s": "cuda_launch_ns",
    "cuda_d2h_s": "cuda_d2h_ns",
    "cuda_zero_filter_s": "cuda_zero_filter_ns",
    "cuda_filter_s": "cuda_filter_ns",
    "cuda_hd_hash_h2d_s": "cuda_hd_hash_h2d_ns",
    "cuda_hd_hv_h2d_s": "cuda_hd_hv_h2d_ns",
    "cuda_hd_alloc_s": "cuda_hd_alloc_ns",
    "cuda_hd_kernel_s": "cuda_hd_kernel_launch_ns",
    "cuda_hd_d2h_s": "cuda_hd_d2h_ns",
}


@dataclass(frozen=True)
class MetricRun:
    section: str
    run_index: int
    family: str
    stage_id: str
    stage_label: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class SummaryRunSpec:
    path: Path
    section: str
    family: str
    stage_id: str
    stage_label: str


def parse_metrics_markdown(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics markdown not found: {path}")

    runs: list[MetricRun] = []
    section = ""
    metrics: dict[str, float] = {}
    section_run_counts: dict[str, int] = {}

    def flush() -> None:
        nonlocal metrics
        if not section or not metrics:
            metrics = {}
            return
        if "wall_s" not in metrics:
            metrics = {}
            return

        run_index = section_run_counts.get(section, 0) + 1
        section_run_counts[section] = run_index
        family, stage_id, stage_label = classify_section(section)
        runs.append(
            MetricRun(
                section=section,
                run_index=run_index,
                family=family,
                stage_id=stage_id,
                stage_label=stage_label,
                metrics=dict(metrics),
            )
        )
        metrics = {}

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
            continue

        match = METRIC_RE.match(line)
        if not match:
            continue

        key, value = match.groups()
        if key == "wall_s" and metrics:
            flush()
        metrics[key] = float(value)

    flush()

    if not runs:
        raise ValueError(f"No metric runs found in {path}")

    records = []
    for run in runs:
        record = {
            "section": run.section,
            "run_index": run.run_index,
            "family": run.family,
            "stage_id": run.stage_id,
            "stage_label": run.stage_label,
        }
        record.update(run.metrics)
        records.append(record)

    df = pd.DataFrame(records)
    return df.sort_values(["family", "stage_id", "run_index"], kind="stable").reset_index(
        drop=True
    )


def parse_metrics_summary_tsv(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics summary TSV not found: {path}")

    df = pd.read_csv(path, sep="\t")
    total = df[df["file"] == "TOTAL"]
    if total.empty:
        raise ValueError(f"No TOTAL row found in {path}")

    row = total.iloc[0]
    metrics = {}
    for metric_name, column in SUMMARY_TSV_COLUMNS.items():
        if column not in row:
            raise ValueError(f"Missing column {column!r} in {path}")
        metrics[metric_name] = float(row[column]) / 1e9
    return metrics


def load_summary_tsv_runs(specs: list[SummaryRunSpec]) -> pd.DataFrame:
    records = []
    for run_index, spec in enumerate(specs, start=1):
        record = {
            "section": spec.section,
            "run_index": run_index,
            "family": spec.family,
            "stage_id": spec.stage_id,
            "stage_label": spec.stage_label,
        }
        record.update(parse_metrics_summary_tsv(spec.path))
        records.append(record)

    if not records:
        raise ValueError("No metrics summary TSV run specs provided")

    return pd.DataFrame(records)


def classify_section(section: str) -> tuple[str, str, str]:
    normalized = section.lower()
    is_multi = normalized.startswith("multi")
    is_single = normalized.startswith("single")

    if not is_multi and not is_single:
        raise ValueError(f"Cannot classify metrics section: {section!r}")

    family = "multi" if is_multi else "single"

    if "baseline old hashset" in normalized:
        return family, "single_hashset", "Old hashset"
    if "hashset" in normalized:
        return family, f"{family}_hashset", "Hashset"
    if "scratch reuse" in normalized and "copy removal" in normalized:
        return (
            family,
            f"{family}_sort_scratchreuse_copy",
            "sort_unstable + scratch reuse + copy removal",
        )
    if "scratch reuse" in normalized:
        return family, f"{family}_sort_scratchreuse", "sort_unstable + scratch reuse"
    if "sort unstable" in normalized:
        return family, f"{family}_sort", "sort_unstable"

    raise ValueError(f"Cannot map metrics section to an optimization stage: {section!r}")


def aggregate_family_runs(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        column
        for column in df.columns
        if column.endswith("_s") and pd.api.types.is_numeric_dtype(df[column])
    ]
    rows = []
    for family, stage_ids in FAMILY_ORDER.items():
        family_df = df[df["family"] == family]
        for order, stage_id in enumerate(stage_ids):
            stage_df = family_df[family_df["stage_id"] == stage_id]
            if stage_df.empty:
                continue

            row = {
                "family": family,
                "stage_id": stage_id,
                "stage_label": str(stage_df["stage_label"].iloc[0]),
                "order": order,
                "run_count": int(len(stage_df)),
                "source_sections": "; ".join(sorted(set(stage_df["section"]))),
            }
            for column in numeric_cols:
                values = stage_df[column].dropna()
                if values.empty:
                    row[column] = float("nan")
                    row[f"{column}_min"] = float("nan")
                    row[f"{column}_max"] = float("nan")
                else:
                    row[column] = float(values.median())
                    row[f"{column}_min"] = float(values.min())
                    row[f"{column}_max"] = float(values.max())
            rows.append(row)

    return pd.DataFrame(rows)


def compute_family_speedups(aggregated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in aggregated.groupby("family", sort=False):
        baseline_stage = FAMILY_ORDER[family][0]
        baseline_rows = group[group["stage_id"] == baseline_stage]
        if baseline_rows.empty:
            raise ValueError(f"Missing baseline stage for {family}: {baseline_stage}")
        baseline = baseline_rows.iloc[0]
        baseline_wall = float(baseline["wall_s"])

        for record in group.sort_values("order").to_dict("records"):
            wall_s = float(record["wall_s"])
            rows.append(
                {
                    "family": family,
                    "stage_id": record["stage_id"],
                    "stage_label": record["stage_label"],
                    "order": int(record["order"]),
                    "run_count": int(record["run_count"]),
                    "wall_s": wall_s,
                    "wall_s_min": float(record.get("wall_s_min", wall_s)),
                    "wall_s_max": float(record.get("wall_s_max", wall_s)),
                    "wall_speedup": baseline_wall / wall_s,
                    "baseline_label": str(baseline["stage_label"]),
                }
            )

    return pd.DataFrame(rows)


def prepend_cpu_hd_encode_eta(
    speedups: pd.DataFrame, eta_s: float = CPU_HD_ENCODE_ETA_S
) -> pd.DataFrame:
    rows = []
    for family, group in speedups.groupby("family", sort=False):
        rows.append(
            {
                "family": family,
                "stage_id": f"{family}_cpu_hd_encode_eta",
                "stage_label": "CPU HD encode ETA",
                "order": -1,
                "run_count": 0,
                "wall_s": float(eta_s),
                "wall_s_min": float(eta_s),
                "wall_s_max": float(eta_s),
                "wall_speedup": 1.0,
                "baseline_label": "CPU HD encode ETA",
            }
        )

        for record in group.sort_values("order").to_dict("records"):
            updated = dict(record)
            updated["order"] = int(updated["order"]) + 1
            updated["wall_speedup"] = float(eta_s) / float(updated["wall_s"])
            updated["baseline_label"] = "CPU HD encode ETA"
            rows.append(updated)

    return pd.DataFrame(rows)


def compute_stage_speedups(aggregated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in aggregated.groupby("family", sort=False):
        baseline_stage = FAMILY_ORDER[family][0]
        baseline_rows = group[group["stage_id"] == baseline_stage]
        if baseline_rows.empty:
            raise ValueError(f"Missing baseline stage for {family}: {baseline_stage}")
        baseline = baseline_rows.iloc[0]

        for record in group.sort_values("order").to_dict("records"):
            for stage_label, column in STAGE_COLUMNS.items():
                baseline_value = float(baseline[column])
                value = float(record[column])
                if baseline_value == 0 and value == 0:
                    speedup = 1.0
                elif baseline_value == 0 or value == 0:
                    speedup = float("nan")
                else:
                    speedup = baseline_value / value
                rows.append(
                    {
                        "family": family,
                        "stage_id": record["stage_id"],
                        "stage_label": record["stage_label"],
                        "order": int(record["order"]),
                        "metric_label": stage_label,
                        "metric": column,
                        "seconds": value,
                        "speedup": speedup,
                    }
                )

    return pd.DataFrame(rows)


def load_metrics(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = parse_metrics_markdown(path)
    aggregated = aggregate_family_runs(raw)
    speedups = compute_family_speedups(aggregated)
    stage_speedups = compute_stage_speedups(aggregated)
    return aggregated, speedups, stage_speedups

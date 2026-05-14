from __future__ import annotations

from pathlib import Path

import pytest

from dotani_speedup_plots.data import (
    aggregate_family_runs,
    compute_family_speedups,
    prepend_cpu_hd_encode_eta,
    parse_metrics_markdown,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT.parent / "dotani_outputs_server" / "5_14" / "metrics_5_14.md"


def test_parse_metrics_markdown_extracts_all_runs() -> None:
    df = parse_metrics_markdown(METRICS_PATH)

    assert len(df) == 8
    assert set(df["family"]) == {"multi", "single"}
    assert df["wall_s"].notna().all()
    assert df["hd_encode_s"].notna().all()


def test_duplicate_single_sort_runs_are_aggregated_by_median() -> None:
    raw = parse_metrics_markdown(METRICS_PATH)
    aggregated = aggregate_family_runs(raw)
    single_sort = aggregated[aggregated["stage_id"] == "single_sort"].iloc[0]

    assert single_sort["run_count"] == 2
    assert single_sort["wall_s_min"] == pytest.approx(1191.01)
    assert single_sort["wall_s_max"] == pytest.approx(1229.41)
    assert single_sort["wall_s"] == pytest.approx((1191.01 + 1229.41) / 2)


def test_family_order_and_known_speedups() -> None:
    raw = parse_metrics_markdown(METRICS_PATH)
    aggregated = aggregate_family_runs(raw)
    speedups = compute_family_speedups(aggregated)

    multi = speedups[speedups["family"] == "multi"].sort_values("order")
    single = speedups[speedups["family"] == "single"].sort_values("order")

    assert list(multi["stage_id"]) == [
        "multi_hashset",
        "multi_sort",
        "multi_sort_scratchreuse",
        "multi_sort_scratchreuse_copy",
    ]
    assert list(single["stage_id"]) == [
        "single_hashset",
        "single_sort",
        "single_sort_scratchreuse_copy",
    ]

    final_multi = multi[multi["stage_id"] == "multi_sort_scratchreuse_copy"].iloc[0]
    final_single = single[single["stage_id"] == "single_sort_scratchreuse_copy"].iloc[0]

    assert final_multi["wall_speedup"] == pytest.approx(749.298 / 347.352)
    assert final_single["wall_speedup"] == pytest.approx(1429.07 / 863.578)


def test_cpu_hd_encode_eta_baseline_is_prepended() -> None:
    raw = parse_metrics_markdown(METRICS_PATH)
    aggregated = aggregate_family_runs(raw)
    speedups = compute_family_speedups(aggregated)
    eta_speedups = prepend_cpu_hd_encode_eta(speedups)

    multi = eta_speedups[eta_speedups["family"] == "multi"].sort_values("order")
    single = eta_speedups[eta_speedups["family"] == "single"].sort_values("order")

    assert multi.iloc[0]["stage_label"] == "CPU HD encode ETA"
    assert single.iloc[0]["stage_label"] == "CPU HD encode ETA"
    assert multi.iloc[0]["wall_s"] == pytest.approx(6600.0)
    assert single.iloc[0]["wall_s"] == pytest.approx(6600.0)

    final_multi = multi[multi["stage_id"] == "multi_sort_scratchreuse_copy"].iloc[0]
    final_single = single[single["stage_id"] == "single_sort_scratchreuse_copy"].iloc[0]
    assert final_multi["wall_speedup"] == pytest.approx(6600.0 / 347.352)
    assert final_single["wall_speedup"] == pytest.approx(6600.0 / 863.578)

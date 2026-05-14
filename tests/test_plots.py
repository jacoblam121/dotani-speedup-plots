from __future__ import annotations

from pathlib import Path

from dotani_speedup_plots.plots import generate_plots


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT.parent / "dotani_outputs_server" / "5_14" / "metrics_5_14.md"


def test_generate_plots_writes_expected_files(tmp_path: Path) -> None:
    outputs, summary = generate_plots(METRICS_PATH, tmp_path)

    assert {path.relative_to(tmp_path).as_posix() for path in outputs} == {
        "gpu_hd_encode_baseline/multi_gpu_speedup.png",
        "gpu_hd_encode_baseline/single_gpu_speedup.png",
        "gpu_hd_encode_baseline/multi_gpu_stage_speedups.png",
        "gpu_hd_encode_baseline/single_gpu_stage_speedups.png",
        "cpu_hd_encode_baseline/multi_gpu_speedup.png",
        "cpu_hd_encode_baseline/single_gpu_speedup.png",
    }
    assert all(path.exists() for path in outputs)
    assert len(summary) == 16

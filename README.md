# dotANI Speedup Plots

Generate optimization-stage speedup plots from the 3x and 4x multi-GPU dotANI
metrics captures.

## Quickstart

```sh
python3 -m dotani_speedup_plots --out figures
```

The default inputs are:

```sh
../dotani_outputs_server/5_14/metrics_5_14.md
../dotani_outputs_server/5_18/
```

Use `--metrics` to point at another 3x markdown file with the same `##` section
and `*_s value` metric format. Use `--metrics-4x-dir` to point at another
directory containing the 4x summary TSV files.

Use `--legacy` to regenerate the original ungrouped 3x-only figure set.

## Generated Figures

The default output is split by GPU count and baseline.

`3x_gpu/` contains multi-GPU charts from
`../dotani_outputs_server/5_14/metrics_5_14.md`:

- `gpu_hd_encode_baseline/multi_gpu_speedup.png`: wall-clock time across the
  3x multi-GPU optimization stages, annotated with speedup versus 3x hashset.
- `gpu_hd_encode_baseline/multi_gpu_stage_speedups.png`: per-stage speedup
  heatmap for the 3x multi-GPU progression.
- `cpu_hd_encode_baseline/multi_gpu_speedup.png`: wall-clock time with a
  synthetic 110-minute CPU HD encode ETA baseline prepended.

`4x_gpu/` contains multi-GPU charts from the 2026-05-18 summary TSV files:

- `gpu_hd_encode_baseline/multi_gpu_speedup.png`: wall-clock time across the
  4x multi-GPU optimization stages, annotated with speedup versus 4x hashset.
- `gpu_hd_encode_baseline/multi_gpu_stage_speedups.png`: per-stage speedup
  heatmap for the 4x multi-GPU progression.
- `cpu_hd_encode_baseline/multi_gpu_speedup.png`: wall-clock time with the same
  synthetic 110-minute CPU HD encode ETA baseline prepended.

The original ungrouped output is still available with `--legacy`.

`gpu_hd_encode_baseline/` contains measured-stage plots from `metrics_5_14.md`:

- `multi_gpu_speedup.png`: wall-clock time across multi-GPU optimization
  stages, annotated with speedup versus `Multi (hashset)`.
- `single_gpu_speedup.png`: wall-clock time across single-GPU optimization
  stages, annotated with speedup versus `Single (baseline old hashset)`.
- `multi_gpu_stage_speedups.png`: per-stage speedup heatmap for the multi-GPU
  progression.
- `single_gpu_stage_speedups.png`: per-stage speedup heatmap for the single-GPU
  progression.

`cpu_hd_encode_baseline/` contains ETA baseline plots:

- `multi_gpu_speedup.png`: wall-clock time with a synthetic 110-minute CPU HD
  encode ETA baseline prepended to the multi-GPU progression.
- `single_gpu_speedup.png`: wall-clock time with the same 110-minute ETA baseline
  prepended to the single-GPU progression.

No stage heatmaps are generated for `cpu_hd_encode_baseline/` because the
110-minute baseline was an ETA, not a completed run with stage counters.

The duplicate single-GPU `sort_unstable` runs are summarized with the median
wall time in the main point and a min/max range marker on the speedup chart.

Stage timings are host-observed counters and are not additive wall-clock totals.

## System Specs

- Server: Russell.
- CPU: AMD Ryzen Threadripper 9985WX, 64 cores, 128 threads, up to 5.4GHz.
- Memory: 8x 96GB DDR5 6400 ECC, 768GB total.
- GPU: 4x NVIDIA RTX PRO 6000 Blackwell 96GB GDDR7 Max-Q Workstation Edition,
  300W TDP.

Multi-GPU runs used 3 GPUs; 1 GPU was busy.

The 2026-05-18 4x multi-GPU runs used all 4 GPUs.

## Tests

```sh
python3 -m pytest
```

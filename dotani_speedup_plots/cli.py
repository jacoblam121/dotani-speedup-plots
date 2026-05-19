from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dotANI optimization-stage speedup plots."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("../dotani_outputs_server/5_14/metrics_5_14.md"),
        help="Markdown file containing 3x dotANI stage timing captures.",
    )
    parser.add_argument(
        "--metrics-4x-dir",
        type=Path,
        default=Path("../dotani_outputs_server/5_18"),
        help="Directory containing 4x dotANI metrics summary TSV files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures"),
        help="Directory for generated plot images.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Output figure format.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Generate the original ungrouped 3x figure set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str((args.out / ".matplotlib").resolve()))

    from .plots import generate_gpu_count_plots, generate_plots

    if args.legacy:
        outputs, speedups = generate_plots(args.metrics, args.out, args.format)
    else:
        outputs, speedups = generate_gpu_count_plots(
            out_dir=args.out,
            output_format=args.format,
            metrics_3x_path=args.metrics,
            metrics_4x_dir=args.metrics_4x_dir,
        )

    for row in speedups:
        print(
            f"{row['baseline']} / {row['family']} / {row['stage']}: "
            f"wall={row['wall_s']:.3f}s "
            f"speedup={row['wall_speedup']:.2f}x "
            f"runs={row['run_count']}"
        )

    print("Wrote:")
    for output in outputs:
        print(f"  {output}")
    return 0

"""Generate comparison plots for pipeline-parallel vs data-parallel benchmarks.

Reads CSV results from benchmark_results/ and produces PNG charts.

Usage:
    python scripts/plot_benchmark_comparison.py \
        --dummy benchmark_results/comparison_20260215_131740.csv \
        --svd benchmark_results/comparison_20260215_132754.csv \
        --output-dir benchmark_results/figures
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _extract(rows: list[dict], mode: str) -> dict:
    """Extract GPU counts, throughput, avg_sample_time for a given mode."""
    filtered = [r for r in rows if r["mode"] == mode]
    gpus = [int(r["gpu_count"]) for r in filtered]
    throughput = [float(r["throughput_sps"]) for r in filtered]
    avg_time = [float(r["avg_sample_s"]) for r in filtered]
    first_time = [float(r["first_sample_s"]) for r in filtered]
    return {
        "gpus": gpus,
        "throughput": throughput,
        "avg_time": avg_time,
        "first_time": first_time,
    }


def plot_throughput(
    dummy_rows: list[dict] | None,
    svd_rows: list[dict] | None,
    output_dir: str,
) -> None:
    """Bar chart: throughput comparison for each GPU count."""
    fig, axes = plt.subplots(1, 2 if (dummy_rows and svd_rows) else 1, figsize=(14, 6))
    if not isinstance(axes, list):
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]

    datasets = []
    if dummy_rows:
        datasets.append(("DummyUNet", dummy_rows))
    if svd_rows:
        datasets.append(("SVD UNet", svd_rows))

    for ax, (title, rows) in zip(axes, datasets):
        pp = _extract(rows, "pipeline_parallel")
        dp = _extract(rows, "data_parallel")

        x = range(len(pp["gpus"]))
        width = 0.35

        bars_pp = ax.bar(
            [i - width / 2 for i in x],
            pp["throughput"],
            width,
            label="Pipeline Parallel",
            color="#4C72B0",
            edgecolor="black",
            linewidth=0.5,
        )
        bars_dp = ax.bar(
            [i + width / 2 for i in x],
            dp["throughput"],
            width,
            label="Data Parallel",
            color="#DD8452",
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_xlabel("GPU Count", fontsize=12)
        ax.set_ylabel("Throughput (samples/s)", fontsize=12)
        ax.set_title(f"Throughput: Pipeline vs Data Parallel ({title})", fontsize=13)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(g) for g in pp["gpus"]])
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars_pp:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.1f}" if h >= 1 else f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for bar in bars_dp:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.1f}" if h >= 1 else f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    path = os.path.join(output_dir, "throughput_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_latency(
    dummy_rows: list[dict] | None,
    svd_rows: list[dict] | None,
    output_dir: str,
) -> None:
    """Line chart: per-sample latency comparison."""
    fig, axes = plt.subplots(1, 2 if (dummy_rows and svd_rows) else 1, figsize=(14, 6))
    if not isinstance(axes, list):
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]

    datasets = []
    if dummy_rows:
        datasets.append(("DummyUNet", dummy_rows))
    if svd_rows:
        datasets.append(("SVD UNet", svd_rows))

    for ax, (title, rows) in zip(axes, datasets):
        pp = _extract(rows, "pipeline_parallel")
        dp = _extract(rows, "data_parallel")

        ax.plot(
            pp["gpus"],
            pp["avg_time"],
            "o-",
            label="Pipeline Parallel",
            color="#4C72B0",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            dp["gpus"],
            dp["avg_time"],
            "s-",
            label="Data Parallel",
            color="#DD8452",
            linewidth=2,
            markersize=8,
        )

        ax.set_xlabel("GPU Count", fontsize=12)
        ax.set_ylabel("Avg Sample Time (s)", fontsize=12)
        ax.set_title(f"Per-Sample Latency ({title})", fontsize=13)
        ax.set_xticks(pp["gpus"])
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        # Annotate values
        for g, t in zip(pp["gpus"], pp["avg_time"]):
            ax.annotate(
                f"{t:.4f}s" if t < 1 else f"{t:.2f}s",
                (g, t),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )
        for g, t in zip(dp["gpus"], dp["avg_time"]):
            ax.annotate(
                f"{t:.4f}s" if t < 1 else f"{t:.2f}s",
                (g, t),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=9,
            )

    plt.tight_layout()
    path = os.path.join(output_dir, "latency_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_scaling_efficiency(
    dummy_rows: list[dict] | None,
    svd_rows: list[dict] | None,
    output_dir: str,
) -> None:
    """Line chart: scaling efficiency (speedup / GPU count)."""
    fig, axes = plt.subplots(1, 2 if (dummy_rows and svd_rows) else 1, figsize=(14, 6))
    if not isinstance(axes, list):
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]

    datasets = []
    if dummy_rows:
        datasets.append(("DummyUNet", dummy_rows))
    if svd_rows:
        datasets.append(("SVD UNet", svd_rows))

    for ax, (title, rows) in zip(axes, datasets):
        pp = _extract(rows, "pipeline_parallel")
        dp = _extract(rows, "data_parallel")

        pp_base = pp["throughput"][0] if pp["throughput"] else 1.0
        dp_base = dp["throughput"][0] if dp["throughput"] else 1.0

        pp_speedup = [t / pp_base for t in pp["throughput"]]
        dp_speedup = [t / dp_base for t in dp["throughput"]]

        ax.plot(
            pp["gpus"],
            pp_speedup,
            "o-",
            label="Pipeline Parallel",
            color="#4C72B0",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            dp["gpus"],
            dp_speedup,
            "s-",
            label="Data Parallel",
            color="#DD8452",
            linewidth=2,
            markersize=8,
        )
        # Ideal scaling line
        ax.plot(
            pp["gpus"],
            pp["gpus"],
            "--",
            label="Ideal (linear)",
            color="gray",
            linewidth=1.5,
            alpha=0.7,
        )

        ax.set_xlabel("GPU Count", fontsize=12)
        ax.set_ylabel("Speedup (x)", fontsize=12)
        ax.set_title(f"Scaling Efficiency ({title})", fontsize=13)
        ax.set_xticks(pp["gpus"])
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        for g, s in zip(pp["gpus"], pp_speedup):
            ax.annotate(
                f"{s:.2f}x",
                (g, s),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )
        for g, s in zip(dp["gpus"], dp_speedup):
            ax.annotate(
                f"{s:.2f}x",
                (g, s),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=9,
            )

    plt.tight_layout()
    path = os.path.join(output_dir, "scaling_efficiency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark comparison charts")
    parser.add_argument("--dummy", type=str, help="Path to DummyUNet comparison CSV")
    parser.add_argument("--svd", type=str, help="Path to SVD comparison CSV")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results/figures",
    )
    args = parser.parse_args()

    if not args.dummy and not args.svd:
        parser.error("At least one of --dummy or --svd must be specified")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dummy_rows = _load_csv(args.dummy) if args.dummy else None
    svd_rows = _load_csv(args.svd) if args.svd else None

    plot_throughput(dummy_rows, svd_rows, args.output_dir)
    plot_latency(dummy_rows, svd_rows, args.output_dir)
    plot_scaling_efficiency(dummy_rows, svd_rows, args.output_dir)
    print(f"\nAll charts saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot grouped bar chart of Average MAD with 95% CI for 8exp (3 methods).

- Reads the wide combined CSV produced by `combine_method_summaries.py`.
- Filters to a target sample size (default: 100).
- Plots MAD means as bars and 95% CI half-widths as error bars.

Output:
  /home/NCMCounterfactuals_test/out/combined/8exp/fig_mad_ci_n<NSAMPLES>.png

Example:
  python src/process_result/plot_8exp_bar_mad_ci.py --n-samples 100
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_CSV = "/home/NCMCounterfactuals_test/out/combined/6large_cpt_onehot/combined_6large_cpt_onehot_wide.csv"
DEFAULT_OUTDIR = "/home/NCMCounterfactuals_test/out/combined/6large_cpt_onehot"


def _method_columns(method: str) -> Dict[str, str]:
    # Backward-compatible default (MAD / avg_error)
    return _method_columns_for_metric(method, metric="mad")


def _method_columns_for_metric(method: str, metric: str) -> Dict[str, str]:
    metric = (metric or "").strip().lower()
    if metric in {"mad", "avg_error", "avg"}:
        prefix = "avg_error"
    elif metric in {"train_time", "elapsed_time", "time"}:
        prefix = "elapsed_time"
    else:
        raise ValueError(f"Unknown metric: {metric}. Expected 'mad' or 'train_time'.")

    return {
        "mean": f"{prefix}_mean__{method}",
        "ci": f"{prefix}_ci_width__{method}",
    }


def _plot_meta(metric: str, n_samples: int) -> Dict[str, str]:
    metric = (metric or "").strip().lower()
    if metric in {"mad", "avg_error", "avg"}:
        return {
            "ylabel": "Average MAD",
            "default_title": f"Average MAD with 95% CI (Sample Size = {n_samples})",
            "outfile": f"fig_mad_ci_n{n_samples}.png",
        }
    if metric in {"train_time", "elapsed_time", "time"}:
        return {
            "ylabel": "Average Train Time (s)",
            "default_title": f"Average Train Time with 95% CI (Sample Size = {n_samples})",
            "outfile": f"fig_train_time_ci_n{n_samples}.png",
        }
    raise ValueError(f"Unknown metric: {metric}. Expected 'mad' or 'train_time'.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV, help="wide combined csv")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="output directory")
    ap.add_argument("--n-samples", type=int, default=100, help="sample size to plot")
    ap.add_argument(
        "--metric",
        default="mad",
        choices=["mad", "train_time"],
        help="what to plot: 'mad' (avg_error_*) or 'train_time' (elapsed_time_*)",
    )
    ap.add_argument("--title", default=None, help="plot title")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    meta = _plot_meta(args.metric, args.n_samples)

    df = pd.read_csv(args.csv)
    df["n_samples"] = pd.to_numeric(df["n_samples"], errors="coerce")

    df = df[df["n_samples"] == args.n_samples].copy()
    if df.empty:
        raise SystemExit(f"No rows for n_samples={args.n_samples} in {args.csv}")

    # Ensure exp1..exp8 order when present
    def _exp_order(g: str) -> int:
        if isinstance(g, str) and g.startswith("exp"):
            try:
                return int(g.replace("exp", ""))
            except Exception:
                return 999
        return 999

    df = df.sort_values(by="graph", key=lambda s: s.map(_exp_order)).reset_index(drop=True)

    graphs: List[str] = df["graph"].astype(str).tolist()

    methods = ["EM4CI", "GAN", "MLE"]
    colors = {
        "EM4CI": "#f39c12",  # orange
        "GAN": "#ff7f50",    # coral
        "MLE": "#4c72b0",    # blue
    }

    # Legend display names (keep internal method keys for column lookup)
    display_names = {
        "EM4CI": "EM4CI",
        "GAN": "GAN-NCM",
        "MLE": "MLE-NCM",
    }

    x = np.arange(len(graphs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, m in enumerate(methods):
        cols = _method_columns_for_metric(m, metric=args.metric)
        y = pd.to_numeric(df.get(cols["mean"]), errors="coerce").to_numpy(dtype=float)
        ci_series = df.get(cols["ci"])
        if ci_series is None:
            yerr = None
        else:
            yerr = pd.to_numeric(ci_series, errors="coerce").to_numpy(dtype=float)

        offset = (i - (len(methods) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            y,
            width,
            label=display_names.get(m, m),
            color=colors.get(m, None),
            edgecolor="none",
            yerr=yerr,
            capsize=6,
            error_kw={
                "elinewidth": 1.5,
                "ecolor": "black",
            },
        )

        # value labels on top of bars
        for b, val in zip(bars, y):
            if np.isfinite(val):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                    rotation=0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(graphs)
    ax.set_ylabel(meta["ylabel"])

    title = args.title or meta["default_title"]
    ax.set_title(title)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(loc="upper right")

    ax.set_ylim(bottom=0)
    fig.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, meta["outfile"])
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

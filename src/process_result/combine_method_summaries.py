#!/usr/bin/env python3
"""Combine per-method summary CSVs (GAN/EM4CI/MLE) into one comparison table.

Currently targets the 8exp setting.

Inputs (default locations):
- GAN (est):   /home/NCMCounterfactuals_test/out/<gan_subdir>/summary_<gan_subdir>.csv
- MLE:         /home/NCMCounterfactuals_test/out/<mle_subdir>/summary_<mle_subdir>.csv
- EM4CI:       /home/NCMCounterfactuals_test/out/em4ci/8exp/summary_em4ci_8exp.csv

Output:
- /home/NCMCounterfactuals_test/out/combined/8exp/combined_8exp.csv

The output is in *long* format, which is easiest for plotting and LaTeX later:
  graph,n_samples,method,avg_error_mean,avg_error_ci_width,elapsed_time_mean,elapsed_time_ci_width,n_trials

Example:
  python combine_method_summaries.py \
    --gan-subdir est_8exp_cpt1 \
    --mle-subdir mle_8exp_cpt1 \
    --out-subdir 8exp

Notes:
- Script is robust to missing rows for some methods (outer join).
- Rows are normalized so all methods share the same column names.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd

OUT_ROOT = "/home/NCMCounterfactuals_test/out"


def _read_summary(path: str, method: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing summary CSV for {method}: {path}")

    df = pd.read_csv(path)

    # normalize required columns
    needed = [
        "graph",
        "n_samples",
        "avg_error_mean",
        "avg_error_ci_width",
        "elapsed_time_mean",
        "elapsed_time_ci_width",
    ]

    # some older summaries might include an "index" column
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"{method} summary missing column '{col}': {path}")

    if "n_trials" not in df.columns:
        df["n_trials"] = float("nan")

    out = df[["graph", "n_samples", "avg_error_mean", "avg_error_ci_width", "elapsed_time_mean", "elapsed_time_ci_width", "n_trials"]].copy()
    out["method"] = method

    # ensure types
    out["n_samples"] = pd.to_numeric(out["n_samples"], errors="coerce")
    out["avg_error_mean"] = pd.to_numeric(out["avg_error_mean"], errors="coerce")
    out["avg_error_ci_width"] = pd.to_numeric(out["avg_error_ci_width"], errors="coerce")
    out["elapsed_time_mean"] = pd.to_numeric(out["elapsed_time_mean"], errors="coerce")
    out["elapsed_time_ci_width"] = pd.to_numeric(out["elapsed_time_ci_width"], errors="coerce")
    out["n_trials"] = pd.to_numeric(out["n_trials"], errors="coerce")

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan-subdir", default="est_8exp_cpt_onehot1", help="subdir under out/ for GAN(est) summary")
    ap.add_argument("--mle-subdir", default="mle_8exp_cpt_onehot1", help="subdir under out/ for MLE summary")
    ap.add_argument(
        "--em4ci-summary",
        default=os.path.join(OUT_ROOT, "em4ci", "6large", "summary_em4ci_6large.csv"),
        help="path to EM4CI 6large summary csv",
    )
    ap.add_argument("--out-subdir", default="8exp_onehot", help="output folder under out/combined/")
    ap.add_argument(
        "--wide",
        action="store_true",
        default=True,
        help="also write a wide-format CSV (one row per graph+n_samples, columns per method)",
    )

    args = ap.parse_args()

    gan_path = os.path.join(OUT_ROOT, args.gan_subdir, f"summary_{args.gan_subdir}.csv")
    mle_path = os.path.join(OUT_ROOT, args.mle_subdir, f"summary_{args.mle_subdir}.csv")
    em4ci_path = args.em4ci_summary

    gan = _read_summary(gan_path, method="GAN")
    mle = _read_summary(mle_path, method="MLE")
    em4ci = _read_summary(em4ci_path, method="EM4CI")

    combined = pd.concat([gan, em4ci, mle], ignore_index=True)
    combined = combined.sort_values(["graph", "n_samples", "method"]).reset_index(drop=True)

    out_dir = os.path.join(OUT_ROOT, "combined", args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    out_long = os.path.join(out_dir, f"combined_{args.out_subdir}.csv")
    combined.to_csv(out_long, index=False)
    print(f"Wrote: {out_long}")

    if args.wide:
        # MultiIndex columns: metric x method
        wide = combined.pivot_table(
            index=["graph", "n_samples"],
            columns="method",
            values=["avg_error_mean", "avg_error_ci_width", "elapsed_time_mean", "elapsed_time_ci_width", "n_trials"],
            aggfunc="first",
        )
        # flatten columns
        wide.columns = [f"{metric}__{method}" for (metric, method) in wide.columns]
        wide = wide.reset_index().sort_values(["graph", "n_samples"])

        out_wide = os.path.join(out_dir, f"combined_{args.out_subdir}_wide.csv")
        wide.to_csv(out_wide, index=False)
        print(f"Wrote: {out_wide}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Map EM4CI LaTeX table numbers to the same CSV summary format used by MLE.

This script is intentionally simple: it parses the embedded LaTeX table (you can
paste/update it), extracts per-(graph, n_samples) values, and writes a
`summary_em4ci_8exp.csv` into:

  /home/NCMCounterfactuals_test/out/em4ci/8exp/

Output columns match the existing `summary_mle_*.csv` format:

index,graph,n_samples,avg_error_mean,avg_error_ci_width,n_trials,elapsed_time_mean,elapsed_time_ci_width

Notes:
- The LaTeX table values appear to be formatted (rounded). We preserve them as
  floats.
- If MAD is missing ("\textemdash"), we write NaN for avg_error_mean/ci.
- n_trials is fixed at 10 to match the caption.
"""

from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

OUT_DIR = "/home/NCMCounterfactuals_test/out/em4ci/8exp"
OUT_CSV = os.path.join(OUT_DIR, "summary_em4ci_8exp.csv")
N_TRIALS = 10


LATEX_TABLE = r"""
exp1 & 100  & 0.094 & 0.100 & 0.231 & 0.020 \\
exp2 & 100  & 0.241 & 0.119 & 0.703 & 0.048 \\
exp3 & 100  & 0.052 & 0.005 & 0.408 & 0.062 \\
exp4 & 100  & 0.106 & 0.029 & 0.746 & 0.071 \\
exp5 & 100  & 0.468 & 0.012 & 0.875 & 0.152 \\
exp6 & 100  & 0.280 & 0.003 & 0.244 & 0.072 \\
exp7 & 100  & 0.214 & 0.045 & 0.422 & 0.028 \\
exp8 & 100  & 0.126 & 0.069 & 0.463 & 0.082 \\
exp1 & 1000 & 0.093 & 0.086 & 1.456 & 0.140 \\
exp2 & 1000 & 0.127 & 0.102 & 3.469 & 0.334 \\
exp3 & 1000 & 0.005 & 0.002 & 1.735 & 0.145 \\
exp4 & 1000 & 0.088 & 0.038 & 5.312 & 0.304 \\
exp5 & 1000 & 0.417 & 0.030 & 6.272 & 0.895 \\
exp6 & 1000 & \textemdash & \textemdash & 0.298 & 0.021 \\
exp7 & 1000 & 0.045 & 0.007 & 3.386 & 0.309 \\
exp8 & 1000 & 0.153 & 0.061 & 2.363 & 0.219 \\
""".strip()


@dataclass
class Row:
    graph: str
    n_samples: int
    mad: Optional[float]
    mad_ci: Optional[float]
    t: float
    t_ci: float


def _parse_float_or_none(token: str) -> Optional[float]:
    token = token.strip()
    if token in {"\\textemdash", "—", "-", ""}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_table(tex: str) -> List[Row]:
    rows: List[Row] = []

    # Remove LaTeX commands we don't need, keep \textemdash as sentinel.
    lines = [ln.strip() for ln in tex.splitlines() if ln.strip()]
    for ln in lines:
        # Strip trailing line terminator
        ln = ln.rstrip("\\")
        parts = [p.strip() for p in ln.split("&")]
        if len(parts) < 6:
            continue
        graph = parts[0]
        n_samples = int(parts[1])
        mad = _parse_float_or_none(parts[2])
        mad_ci = _parse_float_or_none(parts[3])
        t = float(parts[4])
        t_ci = float(parts[5])
        rows.append(Row(graph=graph, n_samples=n_samples, mad=mad, mad_ci=mad_ci, t=t, t_ci=t_ci))

    if not rows:
        raise ValueError("Failed to parse any rows from LATEX_TABLE. Please check formatting.")
    return rows


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = parse_table(LATEX_TABLE)

    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                "index": i,
                "graph": r.graph,
                "n_samples": r.n_samples,
                "avg_error_mean": float("nan") if r.mad is None else float(r.mad),
                "avg_error_ci_width": float("nan") if r.mad_ci is None else float(r.mad_ci),
                "n_trials": float(N_TRIALS),
                "elapsed_time_mean": float(r.t),
                "elapsed_time_ci_width": float(r.t_ci),
            }
        )

    df = pd.DataFrame(out)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

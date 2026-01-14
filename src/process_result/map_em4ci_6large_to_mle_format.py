#!/usr/bin/env python3
"""Map EM4CI LaTeX table numbers (6 large graphs) to the MLE summary CSV format.

Writes:
  /home/NCMCounterfactuals_test/out/em4ci/6large/summary_em4ci_6large.csv

Output columns match `summary_mle_*.csv`:
index,graph,n_samples,avg_error_mean,avg_error_ci_width,n_trials,elapsed_time_mean,elapsed_time_ci_width

Notes:
- Graph names are kept exactly as shown in the table.
- n_trials is unknown from the provided caption; we keep it as NaN by default.
  If you want a fixed value, set N_TRIALS to an int.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

OUT_DIR = "/home/NCMCounterfactuals_test/out/em4ci/6large"
OUT_CSV = os.path.join(OUT_DIR, "summary_em4ci_6large.csv")

# The caption doesn't specify trials for 6large. Keep NaN unless you set it.
N_TRIALS: Optional[int] = None

LATEX_TABLE = r"""
6\_cone\_cloud\_TD4\_10  & 1000  & 0.0154 & 0.0006 & 9.6141   & 1.5464 \\
15\_cone\_cloud\_TD4\_10 & 1000  & 0.0085 & 0.0012 & 21.6600  & 2.8595 \\
17\_diamond\_TD4\_10     & 1000  & 0.0419 & 0.0032 & 31.9534  & 2.7555 \\
49\_chain\_TD4\_10       & 1000  & 0.0015 & 0.0007 & 102.5710 & 11.8168 \\
65\_diamond\_TD4\_10     & 1000  & 0.0088 & 0.0020 & 238.1271 & 41.9003 \\
99\_chain\_TD4\_10       & 1000  & 0.0052 & 0.0017 & 318.7040 & 43.1746 \\
6\_cone\_cloud\_TD4\_10  & 10000 & 0.0129 & 0.0008 & 63.3512  & 8.6073 \\
15\_cone\_cloud\_TD4\_10 & 10000 & 0.0080 & 0.0017 & 195.4051 & 24.3136 \\
17\_diamond\_TD4\_10     & 10000 & 0.0413 & 0.0028 & 245.5714 & 23.7657 \\
49\_chain\_TD4\_10       & 10000 & 0.0044 & 0.0035 & 838.2275 & 32.4931 \\
65\_diamond\_TD4\_10     & 10000 & 0.0039 & 0.0004 & 2110.3909& 155.9789 \\
99\_chain\_TD4\_10       & 10000 & 0.0034 & 0.0011 & 2595.7048& 69.8427 \\
""".strip()


@dataclass
class Row:
    graph: str
    n_samples: int
    mad: float
    mad_ci: float
    t: float
    t_ci: float


def parse_table(tex: str) -> List[Row]:
    rows: List[Row] = []
    lines = [ln.strip() for ln in tex.splitlines() if ln.strip()]
    for ln in lines:
        ln = ln.rstrip("\\")
        parts = [p.strip() for p in ln.split("&")]
        if len(parts) < 6:
            continue

        # Unescape latex underscores for readability in CSV.
        graph = parts[0].replace("\\_", "_")

        n_samples = int(parts[1])
        mad = float(parts[2])
        mad_ci = float(parts[3])

        # Time columns sometimes have no space before '&', e.g. 2110.3909&
        t = float(parts[4].replace(" ", ""))
        t_ci = float(parts[5].replace(" ", ""))

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
                "avg_error_mean": float(r.mad),
                "avg_error_ci_width": float(r.mad_ci),
                "n_trials": float("nan") if N_TRIALS is None else float(N_TRIALS),
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

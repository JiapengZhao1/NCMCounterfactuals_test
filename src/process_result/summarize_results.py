import os
import sys
import pandas as pd

ROOT_DIR = "/home/NCMCounterfactuals_test/out/"


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python summarize_8exp_results.py <subdir> [dim]")
        print("Example: python summarize_8exp_results.py est_8exp_cpt1 1")
        return 1

    subdir = sys.argv[1]
    dim = int(sys.argv[2]) if len(sys.argv) >= 3 else None

    base_dir = os.path.join(ROOT_DIR, subdir)
    input_csv = os.path.join(base_dir, f"collected_results_{subdir}.csv")
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Missing collected results CSV: {input_csv}")

    df = pd.read_csv(input_csv)

    # Optional: restrict dim
    if dim is not None and "dim" in df.columns:
        df = df[df["dim"] == dim]

    # Include 10000 samples if present in the collected data
    wanted_samples = [100, 1000]
    if "n_samples" in df.columns and (df["n_samples"] == 10000).any():
        wanted_samples.append(10000)
    df = df[df["n_samples"].isin(wanted_samples)]

    required = {"graph", "n_samples", "trial_index", "avg_error", "elapsed_time_2"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"CSV missing required columns: {sorted(missing)}")

    # Drop incomplete rows
    df = df.dropna(subset=["graph", "n_samples", "trial_index", "avg_error", "elapsed_time_2"])

    # Focus on exp1..exp8 if present
    wanted_graphs = {f"exp{i}" for i in range(1, 9)}
    if df["graph"].isin(wanted_graphs).any():
        df = df[df["graph"].isin(wanted_graphs)]

    # Only keep trials 0..9 (10 trials) if column is present
    df = df[df["trial_index"].between(0, 9)]

    # 1) Per-exp, per-n_samples mean over trials
    # 计算均值与 95% CI 半宽度（t 分布，适用于 trials=10 这种小样本）
    def _mean_ci_width(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").dropna()
        n = int(s.shape[0])
        if n == 0:
            return pd.Series({"mean": float("nan"), "ci_width": float("nan"), "n": 0})
        mean = float(s.mean())
        if n == 1:
            return pd.Series({"mean": mean, "ci_width": 0.0, "n": 1})
        std = float(s.std(ddof=1))
        se = std / (n ** 0.5)
        # t_{0.975, n-1}
        try:
            from scipy.stats import t

            tcrit = float(t.ppf(0.975, df=n - 1))
        except Exception:
            # fallback: normal approx
            tcrit = 1.96
        half = abs(tcrit * se)
        return pd.Series({"mean": mean, "ci_width": half, "n": n})

    grouped = df.groupby(["graph", "n_samples"], as_index=False)

    avg_ci = grouped["avg_error"].apply(_mean_ci_width).reset_index()
    avg_ci = avg_ci.rename(
        columns={
            "mean": "avg_error_mean",
            "ci_width": "avg_error_ci_width",
            "n": "n_trials",
        }
    )

    time_ci = grouped["elapsed_time_2"].apply(_mean_ci_width).reset_index()
    time_ci = time_ci.rename(
        columns={
            "mean": "elapsed_time_mean",
            "ci_width": "elapsed_time_ci_width",
            "n": "n_trials_time",
        }
    )

    per_exp = avg_ci.merge(
        time_ci[["graph", "n_samples", "elapsed_time_mean", "elapsed_time_ci_width"]],
        on=["graph", "n_samples"],
        how="left",
    )
    per_exp = per_exp.sort_values(["graph", "n_samples"])

    # Write to same directory
    out_csv_exp = os.path.join(base_dir, f"summary_{subdir}.csv")
    per_exp.to_csv(out_csv_exp, index=False)

    print("Per-exp summary (mean over trials):")
    print(per_exp.to_string(index=False))
    print(f"\nWrote: {out_csv_exp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

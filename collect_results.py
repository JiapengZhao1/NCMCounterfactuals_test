import os
import sys
import json
import pandas as pd

# 固定根目录
root_dir = "/home/NCMCounterfactuals_test/out/"

# 获取子目录作为命令行参数
if len(sys.argv) < 2:
    print("Usage: python collect_results.py <subdir>")
    print("Example: python collect_results.py est_8exp_1")
    sys.exit(1)

sub_dir = sys.argv[1]
base_dir = os.path.join(root_dir, sub_dir)

# Initialize a list to store results
results_data = []

# Walk through the directory structure
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "results.json":
            results_file = os.path.join(root, file)
            # Extract graph name, n_samples, and trial_index from the directory structure
            path_parts = root.split(os.sep)
            exp_info = path_parts[-1]  # e.g. "gen=CTM-graph=exp1-n_samples=100-dim=1-trial_index=0"
            info_dict = dict(item.split("=", 1) for item in exp_info.split("-") if "=" in item)
            graph_name = info_dict.get("graph", "")
            n_samples = info_dict.get("n_samples", "")
            trial_index = info_dict.get("trial_index", "")

            # Load the results.json file
            with open(results_file, "r") as f:
                results = json.load(f)

            if isinstance(results, dict):
                results["graph"] = graph_name
                results["n_samples"] = int(n_samples) if n_samples else None
                results["trial_index"] = int(trial_index) if trial_index else None
                results_data.append(results)

# Convert the results data to a DataFrame
results_df = pd.DataFrame(results_data)

# 排序：先按 graph，再按 n_samples，再按 trial_index
results_df = results_df.sort_values(by=["graph", "n_samples", "trial_index"])

# Reorder columns for better readability
columns_order = [
    "graph", "n_samples", "trial_index",
    "avg_error", "elapsed_time"
]
results_df = results_df[[col for col in columns_order if col in results_df.columns]]

print(results_df)

# Save the results to a CSV file, using the exp(sub_dir) name
output_csv = os.path.join(base_dir, f"collected_results_{sub_dir}.csv")
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
results_df.to_csv(output_csv, index=False)
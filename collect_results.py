import os
import json
import pandas as pd

# Base directory
base_dir = "/NCMCounterfactuals_test/out/est_8exp_1"

# Initialize a list to store results
results_data = []

# Walk through the directory structure
for root, dirs, files in os.walk(base_dir):
    if "results.json" in files:
        # Extract graph name and n_samples from the directory structure
        path_parts = root.split("/")
        graph_info = path_parts[-1]  # Example: "gen=CTM-graph=exp1-n_samples=100-dim=2-trial_index=0"
        graph_name = graph_info.split("-")[1].split("=")[1]  # Extract "exp1"
        n_samples = graph_info.split("-")[2].split("=")[1]  # Extract "100"

        # Load the results.json file
        results_file = os.path.join(root, "results.json")
        with open(results_file, "r") as f:
            results = json.load(f)

        # Add graph name and n_samples to the results
        results["graph"] = graph_name
        results["n_samples"] = int(n_samples)

        # Append to the results data
        results_data.append(results)

# Convert the results data to a DataFrame
results_df = pd.DataFrame(results_data)

# Reorder columns for better readability
columns_order = ["graph", "n_samples", "total_true_KL", "total_dat_KL", "true_KL_P(V)", "dat_KL_P(V)", 
                 "true_ATE", "ncm_ATE", "err_ncm_ATE", "avg_error", "elapsed_time"]
results_df = results_df[[col for col in columns_order if col in results_df.columns]]

# Save the results to a CSV file
output_csv = "/NCMCounterfactuals_test/out/est_8exp_1/collected_results.csv"
results_df.to_csv(output_csv, index=False)

# Print the DataFrame
print(results_df)
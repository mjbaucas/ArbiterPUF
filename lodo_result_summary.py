import pandas as pd
import numpy as np
import sys

# =========================
# Load CSV
# =========================
input_file = sys.argv[1]
df = pd.read_csv(input_file)

# =========================
# Required columns check
# =========================
required_cols = [
    "k_train_datasets",
    "precision",
    "recall",
    "TP",
    "FP",
    "FN",
    "old_ber",
    "new_ber"
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# =========================
# Ensure TN exists
# =========================
if "TN" not in df.columns:
    # Try to infer TN if total_samples exists
    if "total_samples" in df.columns:
        df["TN"] = df["total_samples"] - (df["TP"] + df["FP"] + df["FN"])
    else:
        raise ValueError("CSV must contain either 'TN' or 'total_samples'")

# =========================
# BER percent reduction
# =========================
df["ber_percent_reduction"] = (
    (df["old_ber"] - df["new_ber"]) / (df["old_ber"] + 1e-12)
) * 100

df["ber_percent_reduction"] = df["ber_percent_reduction"].replace(
    [np.inf, -np.inf], np.nan
)

# =========================
# Aggregate
# =========================
summary = df.groupby("k_train_datasets").agg({
    "precision": "mean",
    "recall": "mean",
    "ber_percent_reduction": "mean",
    "TP": "sum",
    "FP": "sum",
    "TN": "sum",
    "FN": "sum"
}).reset_index()

# =========================
# Print results
# =========================
print("\n===== FINAL AGGREGATED RESULTS =====")

for _, row in summary.iterrows():
    k = int(row["k_train_datasets"])

    TP = int(row["TP"])
    FP = int(row["FP"])
    TN = int(row["TN"])
    FN = int(row["FN"])

    print(f"\n====================================")
    print(f"Training Datasets Used: {k}")
    print(f"====================================")

    print("\n--- AVERAGED METRICS ---")
    print(f"Precision: {row['precision']:.4f}")
    print(f"Recall:    {row['recall']:.4f}")
    print(f"BER Reduction: {row['ber_percent_reduction']:.2f}%")

    print("\n--- CONFUSION MATRIX ---")
    print(f"TP: {TP}, FP: {FP}")
    print(f"FN: {FN}, TN: {TN}")
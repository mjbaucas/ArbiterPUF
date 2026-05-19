import numpy as np
import pandas as pd
import sys
import itertools
import random

from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix


# -------------------------
# PARSER
# -------------------------
def parse_deltas(row):
    return np.array([float(x) for x in str(row["deltas"]).split(",") if x != ""])


# -------------------------
# RESPONSE MODEL (PUF abstraction)
# -------------------------
def get_response(deltas):
    return (deltas > 0).astype(int)


# -------------------------
# STABILITY LABEL (MC ground truth)
# -------------------------
def is_stable(deltas, threshold=1.0):
    return not np.any(np.abs(deltas) < threshold)


# -------------------------
# FEATURE ENGINEERING (HYBRID)
# -------------------------
def make_features(df):

    X = []
    y = []

    for _, row in df.iterrows():

        deltas = parse_deltas(row)
        if len(deltas) == 0:
            continue

        abs_d = np.abs(deltas)

        # LIGHTWEIGHT DELAY SUMMARY
        mean_abs = np.mean(abs_d)
        std_abs = np.std(abs_d)
        min_abs = np.min(abs_d)
        max_abs = np.max(abs_d)

        near_boundary = np.mean(abs_d < 1.0)

        # margin to decision boundary (proxy stability signal)
        margin_proxy = np.mean(abs_d - 1.0)

        # RESPONSE FEATURES
        response = get_response(deltas)

        bit_mean = np.mean(response)
        bit_entropy = -(bit_mean*np.log2(bit_mean + 1e-9) +
                        (1-bit_mean)*np.log2(1-bit_mean + 1e-9))

        transitions = np.sum(np.abs(np.diff(response)))

        # FINAL FEATURE VECTOR
        X.append([
            mean_abs,
            std_abs,
            min_abs,
            max_abs,
            near_boundary,
            margin_proxy,
            bit_mean,
            bit_entropy,
            transitions
        ])

        y.append(int(is_stable(deltas)))

    return np.array(X), np.array(y)


# MAIN
files = sys.argv[1:]

print("\n===== HYBRID PUF STABILITY MODEL =====")

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    eval_metric="logloss",
    random_state=42
)

for k in range(1, len(files)):

    print(f"\n--- Training with {k} dataset(s) ---")

    all_tp, all_tn, all_fp, all_fn = [], [], [], []

    all_subsets = list(itertools.combinations(files, k))
    sampled_subsets = random.sample(all_subsets, min(10, len(all_subsets)))

    for train_files in sampled_subsets:

        train_files = list(train_files)
        test_candidates = [f for f in files if f not in train_files]

        for test_file in test_candidates:

            df_train = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
            df_test = pd.read_csv(test_file)

            X_train, y_train = make_features(df_train)
            X_test, y_test = make_features(df_test)

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            if len(np.unique(y_train)) < 2:
                continue

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

            all_tp.append(tp)
            all_tn.append(tn)
            all_fp.append(fp)
            all_fn.append(fn)

    # -------------------------
    # METRICS
    # -------------------------
    if all_tp:

        TP = sum(all_tp)
        TN = sum(all_tn)
        FP = sum(all_fp)
        FN = sum(all_fn)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        ber = (FP + FN) / (TP + TN + FP + FN)

        print(f"k = {k}")
        print(f"Accuracy  = {accuracy:.4f}")
        print(f"Precision = {precision:.4f}")
        print(f"Recall    = {recall:.4f}")
        print(f"BER       = {ber:.4f}")
        print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")

    else:
        print(f"k = {k}, no valid runs")
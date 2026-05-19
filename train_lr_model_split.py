import numpy as np
import pandas as pd
import sys
import random
import itertools

from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler


# CLI ARGUMENTS
if len(sys.argv) < 4:
    print("Usage: python script.py <train_pct> <test_pct> <dataset1> <dataset2> ...")
    sys.exit(1)

train_pct = float(sys.argv[1])
test_pct  = float(sys.argv[2])
train_files = sys.argv[3:]

if not (0 < train_pct <= 1 and 0 < test_pct <= 1):
    print("train_pct and test_pct must be in (0, 1]")
    sys.exit(1)

print(f"\nUsing train_pct={train_pct}, test_pct={test_pct}")

# PARSING
def parse_deltas(s):
    if pd.isna(s):
        return np.array([0.0])

    vals = []
    for c in str(s).split("|"):
        for v in c.split(";"):
            try:
                vals.append(float(v.strip()))
            except:
                pass

    return np.array(vals if vals else [0.0])

def parse_responses(s):
    reps = str(s).split("|")
    parsed = []

    for r in reps:
        r = r.strip()
        if r:
            parsed.append(np.array([int(bit) for bit in r]))

    return np.array(parsed)

# BER
def compute_ber(responses):
    if len(responses) == 0:
        return 1.0

    majority = np.round(np.mean(responses, axis=0))
    flips = np.mean(np.abs(responses - majority))
    return flips

# FEATURES
def extract_features(df):
    X, y = [], []

    for _, row in df.iterrows():
        deltas = parse_deltas(row["deltas"])
        responses = parse_responses(row["responses"])
        ber = compute_ber(responses)

        feats = [
            np.mean(deltas),
            np.std(deltas),
            np.max(deltas) - np.min(deltas),
            np.mean(np.abs(deltas)),
            skew(deltas) if len(deltas) > 2 else 0,
            kurtosis(deltas) if len(deltas) > 3 else 0,
        ]

        if len(responses) > 0:
            bit_means = np.mean(responses, axis=0)

            response_feats = [
                len(responses),
                responses.shape[1],
                np.mean(bit_means),
                np.std(bit_means),
                np.max(bit_means) - np.min(bit_means),
            ]
        else:
            response_feats = [0, 0, 0, 0, 0]

        feats.extend(response_feats)

        X.append(feats)
        y.append(ber)

    return np.array(X), np.array(y)

# MODEL
def train_model(X_train, y_train, threshold=0.01):
    y_binary = (y_train < threshold).astype(int)

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1
    )

    model.fit(X_train, y_binary)
    return model

# THRESHOLD
def find_best_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    best_t = thresholds[0]
    best_score = -1

    for p, r, t in zip(precision, recall, thresholds):
        if 0.6 <= t <= 0.9:
            score = p * r
            if score > best_score:
                best_score = score
                best_t = t

    return best_t

# MAIN LODO LOOP
random.shuffle(train_files)
N = len(train_files)

threshold = 0.01
results = []

print("\n===== LOGISTIC REGRESSION (WITH DATA SCALING) =====")

for k in range(1, N):

    print(f"\n==============================")
    print(f"Training with {k} dataset(s)")
    print(f"==============================")

    for train_subset in itertools.combinations(train_files, k):
        train_subset = list(train_subset)

        test_candidates = [f for f in train_files if f not in train_subset]

        # LOAD + SAMPLE TRAIN DATA
        dfs = []
        for f in train_subset:
            df = pd.read_csv(f)

            if len(df) > 1:
                df = df.sample(frac=train_pct, random_state=42)

            dfs.append(df)

        df_train = pd.concat(dfs, ignore_index=True)

        X_train, y_train = extract_features(df_train)

        # shuffle
        perm = np.random.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        # split
        split = int(0.8 * len(X_train))
        X_tr, X_val = X_train[:split], X_train[split:]
        y_tr, y_val = y_train[:split], y_train[split:]

        # SCALE
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        # train
        model = train_model(X_tr, y_tr, threshold)

        # validation
        val_scores = model.predict_proba(X_val)[:, 1]
        y_val_bin = (y_val < threshold)

        best_threshold = find_best_threshold(y_val_bin, val_scores)

        # TEST LOOP
        for test_file in test_candidates:

            df_test = pd.read_csv(test_file)

            if len(df_test) > 1:
                df_test = df_test.sample(frac=test_pct, random_state=42)

            X_test, y_test = extract_features(df_test)

            X_test = scaler.transform(X_test)

            test_scores = model.predict_proba(X_test)[:, 1]
            y_test_bin = (y_test < threshold)
            pred = test_scores >= best_threshold

            TP = np.sum(pred & y_test_bin)
            FP = np.sum(pred & ~y_test_bin)
            FN = np.sum(~pred & y_test_bin)
            TN = len(y_test) - (TP + FP + FN)

            precision = TP / (TP + FP + 1e-9)
            recall = TP / (TP + FN + 1e-9)

            old_ber = np.mean(y_test)
            new_ber = np.mean(y_test[pred]) if np.sum(pred) > 0 else float("nan")

            print(f"\n[LogReg | Train k={k} | Test: {test_file}]")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, BER Reduction: {old_ber - new_ber:.6f}")

            results.append({
                "model": "LogReg",
                "k_train_datasets": k,
                "train_pct": train_pct,
                "test_pct": test_pct,
                "train_set": "|".join(train_subset),
                "test_dataset": test_file,
                "best_threshold": best_threshold,
                "precision": precision,
                "recall": recall,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,
                "old_ber": old_ber,
                "new_ber": new_ber,
                "ber_reduction": old_ber - new_ber
            })

# SAVE RESULTS
df_results = pd.DataFrame(results)
df_results.to_csv("logreg_lodo_results_with_pct.csv", index=False)

print("\nSaved: logreg_lodo_results_with_pct.csv")
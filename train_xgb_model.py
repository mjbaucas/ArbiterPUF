import numpy as np
import pandas as pd
import sys

from scipy.stats import skew, kurtosis
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve

# Dataset parsing
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

# BER computation
def compute_ber(responses):
    if len(responses) == 0:
        return 1.0

    majority = np.round(np.mean(responses, axis=0))
    flips = np.mean(np.abs(responses - majority))
    return flips

# Feature extraction
def extract_features(df):
    X, y = [], []

    for _, row in df.iterrows():
        deltas = parse_deltas(row["deltas"])
        responses = parse_responses(row["responses"])
        ber = compute_ber(responses)

        # delta features 
        feats = [
            np.mean(deltas),
            np.std(deltas),
            np.max(deltas) - np.min(deltas),
            np.mean(np.abs(deltas)),
            skew(deltas) if len(deltas) > 2 else 0,
            kurtosis(deltas) if len(deltas) > 3 else 0,
        ]

        # response features 
        if len(responses) > 0:
            bit_means = np.mean(responses, axis=0)

            response_feats = [
                len(responses),                         # number of samples
                responses.shape[1],                     # bit-length
                np.mean(bit_means),                     # overall bias 
                np.std(bit_means),                      # variability across bits 
                np.max(bit_means) - np.min(bit_means),  # spread 
            ]
        else:
            response_feats = [0, 0, 0, 0, 0]

        feats.extend(response_feats)

        X.append(feats)
        y.append(ber)

    return np.array(X), np.array(y)

# Training model
def train_model(X_train, y_train, threshold=0.01):
    y_binary = (y_train < threshold).astype(int)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,          
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=8,   
        gamma=2.0,           
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="aucpr"
    )

    model.fit(X_train, y_binary)
    return model

# Threshold optimization
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

# Main loop
train_files = sys.argv[1:-1]
test_file = sys.argv[-1]

df_train = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
df_test = pd.read_csv(test_file)

X_train, y_train = extract_features(df_train)
X_test, y_test = extract_features(df_test)

# Validation split
split = int(0.8 * len(X_train))

X_tr, X_val = X_train[:split], X_train[split:]
y_tr, y_val = y_train[:split], y_train[split:]

# Train
threshold = 0.01
model = train_model(X_tr, y_tr, threshold)

# Validation threshold tuning
val_scores = model.predict_proba(X_val)[:, 1]
y_val_bin = (y_val < threshold)

best_threshold = find_best_threshold(y_val_bin, val_scores)

# Test
test_scores = model.predict_proba(X_test)[:, 1]
y_test_bin = (y_test < threshold)

pred = test_scores >= best_threshold

# Evaluation
TP = np.sum(pred & y_test_bin)
FP = np.sum(pred & ~y_test_bin)
FN = np.sum(~pred & y_test_bin)

precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)

old_ber = np.mean(y_test)
new_ber = np.mean(y_test[pred]) if np.sum(pred) > 0 else float("nan")


print("\n===== CRP CLASSIFIER (FIXED) =====")

print("\n--- MODEL ---")
print(f"Best threshold: {best_threshold:.6f}")

print("\n--- CLASSIFICATION ---")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

print("\n--- CONFUSION MATRIX ---")
print(f"TP: {TP}, FP: {FP}, FN: {FN}")

print("\n--- BER IMPACT ---")
print(f"Old BER:  {old_ber:.6f}")
print(f"New BER:  {new_ber:.6f}")
print(f"Reduction:{old_ber - new_ber:.6f}")
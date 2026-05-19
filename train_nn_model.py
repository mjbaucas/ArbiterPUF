import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import sys
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor

def parse_deltas(s):
    if pd.isna(s):
        return np.array([0.0])

    chains = str(s).split("|")
    vals = []

    for c in chains:
        if c.strip() == "":
            continue

        parts = c.split(";")
        for v in parts:
            v = v.strip()
            if v == "":
                continue
            try:
                vals.append(float(v))
            except:
                pass

    return np.array(vals if len(vals) > 0 else [0.0])

def parse_responses(s):
    reps = str(s).split("|")
    parsed = []

    for r in reps:
        if r.strip() == "":
            continue
        parsed.append([int(bit) for bit in r.strip()])

    return np.array(parsed)  

def compute_ber(responses):
    majority = np.round(np.mean(responses, axis=0))
    flips = np.sum(responses != majority, axis=0)
    ber_per_bit = flips / responses.shape[0]
    return np.mean(ber_per_bit)

class CRPDataset(Dataset):
    def __init__(self, df, mean=None, std=None):
        self.X = []
        self.y = []

        for _, row in df.iterrows():
            deltas = parse_deltas(row["deltas"])
            responses = parse_responses(row["responses"])

            ber = compute_ber(responses)

            feats = [
                np.mean(deltas),
                np.std(deltas),
                np.min(deltas),
                np.max(deltas),
                np.max(deltas) - np.min(deltas),
                skew(deltas) if len(deltas) > 2 else 0,
                kurtosis(deltas) if len(deltas) > 3 else 0,
                abs(np.sum(deltas)),
            ]

            repeat_std = np.mean(np.std(responses, axis=0))
            feats.append(repeat_std)

            bit_variance = np.mean(np.var(responses, axis=0))
            feats.append(bit_variance)

            self.X.append(feats)
            self.y.append(ber)

        self.X = np.array(self.X)

        # Use provided normalization or compute from training
        if mean is None:
            self.mean = self.X.mean(axis=0)
            self.std = self.X.std(axis=0) + 1e-9
        else:
            self.mean = mean
            self.std = std

        self.X = (self.X - self.mean) / self.std

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_model(model, loader, optimizer, epochs=50):
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0

        for X, y in loader:
            optimizer.zero_grad()

            preds = model(X)
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

def evaluate(model, loader):
    model.eval()

    preds_all = []
    labels_all = []

    with torch.no_grad():
        for X, y in loader:
            preds = model(X)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y.cpu().numpy())

    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)

    # Regression metrics
    mse = np.mean((preds_all - labels_all) ** 2)
    mae = np.mean(np.abs(preds_all - labels_all))

    # Define how many CRPs to keep
    k = int(0.30 * len(preds_all)) 

    # Threshold
    threshold = np.percentile(labels_all, 30)

    # Select lowest BER predictions (best CRPs)
    topk_idx = np.argsort(preds_all)[:k]


    selected = preds_all < threshold
    selected[topk_idx] = 1
    
    true_stable = (labels_all < threshold).astype(int)

    # Metrics
    tp = np.sum((selected == 1) & (true_stable == 1))
    fp = np.sum((selected == 1) & (true_stable == 0))
    tn = np.sum((selected == 0) & (true_stable == 0))
    fn = np.sum((selected == 0) & (true_stable == 1))

    precision_at_k = tp / (tp + fp + 1e-9)
    recall_at_k = tp / (tp + fn + 1e-9)

    kept_ber = np.mean(labels_all[topk_idx])
    overall_ber = np.mean(labels_all)

    print("\n===== RESULTS =====")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    print("\n===== FILTERING RESULTS =====")
    print(f"Top-K kept: {k} / {len(preds_all)}")
    print(f"Precision at K: {precision_at_k:.4f}")
    print(f"Recall at K: {recall_at_k:.4f}")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    print("\n===== BER STATS =====")
    print("Min:", np.min(labels_all))
    print("Max:", np.max(labels_all))
    print("Mean:", np.mean(labels_all))
    print("Original BER:", overall_ber)
    print("Filtered BER:", kept_ber)

    print("\n===== PREDICTION STATS =====")
    print("Min:", np.min(preds_all))
    print("Max:", np.max(preds_all))
    print("Mean:", np.mean(preds_all))

    
# Main loop
train_datasets = sys.argv[1:-1]  # all except last = training
test_dataset = sys.argv[-1]      # last = test

# Load and concatenate training datasets
df_train_list = [pd.read_csv(f) for f in train_datasets]
df_train = pd.concat(df_train_list, ignore_index=True)

# Load test dataset
df_test = pd.read_csv(test_dataset)

# Build training dataset first
train_data = CRPDataset(df_train)

# Reuse its normalization for test
test_data = CRPDataset(df_test, mean=train_data.mean, std=train_data.std)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

model = MLP(input_dim=train_data.X.shape[1])

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, optimizer, epochs=50)

rf = RandomForestRegressor(n_estimators=200, max_depth=10)
rf.fit(train_data.X.numpy(), train_data.y.numpy())

rf_preds = rf.predict(test_data.X.numpy())

print("\nRF MSE:", np.mean((rf_preds - test_data.y.numpy())**2))

evaluate(model, test_loader)
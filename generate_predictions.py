import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

# ===============================
# Load datasets
# ===============================

train_df = pd.read_csv("TRAIN.csv")
test_df = pd.read_csv("TEST.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# ===============================
# Feature list
# ===============================

FC = [f"F{str(i).zfill(2)}" for i in range(1,48)]
FC = [c for c in FC if c in train_df.columns]

# ===============================
# Feature Engineering
# ===============================

def engineer(df, fc):

    X = df[fc].copy()
    vals = df[fc]

    # statistical features
    X["stat_mean"] = vals.mean(axis=1)
    X["stat_std"] = vals.std(axis=1)
    X["stat_max"] = vals.max(axis=1)
    X["stat_min"] = vals.min(axis=1)
    X["stat_range"] = X["stat_max"] - X["stat_min"]
    X["stat_skew"] = vals.skew(axis=1)
    X["stat_kurt"] = vals.kurt(axis=1)

    # norms
    X["l1_norm"] = vals.abs().sum(axis=1)
    X["l2_norm"] = np.sqrt((vals**2).sum(axis=1))

    # grouped sensor behaviour
    for name,grp in [
        ("g1",fc[0:10]),
        ("g2",fc[10:20]),
        ("g3",fc[20:30]),
        ("g4",fc[30:40]),
        ("g5",fc[40:47])
    ]:
        X[f"{name}_mean"] = df[grp].mean(axis=1)
        X[f"{name}_std"] = df[grp].std(axis=1)
        X[f"{name}_range"] = df[grp].max(axis=1) - df[grp].min(axis=1)

    # spike detection
    spike = [c for c in ["F31","F32","F33"] if c in df.columns]

    X["spike_sum"] = df[spike].sum(axis=1)
    X["spike_max"] = df[spike].max(axis=1)

    X["spike_gt10"] = (df[spike] > 10).any(axis=1).astype(int)
    X["spike_gt50"] = (df[spike] > 50).any(axis=1).astype(int)

    # anomaly counters
    X["count_large"] = (vals.abs() > 10).sum(axis=1)

    return X

# ===============================
# Apply feature engineering
# ===============================

X_train_raw = engineer(train_df, FC)
y = train_df["Class"]

X_test_raw = engineer(test_df, FC)

test_ids = test_df["ID"]

# ===============================
# Preprocessing
# ===============================

imputer = SimpleImputer(strategy="median")
scaler = RobustScaler()

X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw))
X_test = scaler.transform(imputer.transform(X_test_raw))

print("Total features:", X_train.shape[1])

# ===============================
# Validation split
# ===============================

Xtr, Xval, ytr, yval = train_test_split(
    X_train,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# Model (LightGBM)
# ===============================

model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("Training model...")

model.fit(Xtr, ytr)

# ===============================
# Optimize threshold
# ===============================

val_prob = model.predict_proba(Xval)[:,1]

best_f1 = 0
best_thr = 0.5

for thr in np.arange(0.2,0.8,0.01):

    pred = (val_prob >= thr).astype(int)

    score = f1_score(yval,pred)

    if score > best_f1:

        best_f1 = score
        best_thr = thr

print("Best threshold:", best_thr)
print("Validation F1:", best_f1)

# ===============================
# Train on full dataset
# ===============================

print("Training on full dataset...")

model.fit(X_train,y)

# ===============================
# Test predictions
# ===============================

test_prob = model.predict_proba(X_test)[:,1]

final_pred = (test_prob >= best_thr).astype(int)

# ===============================
# Create submission file
# ===============================

submission = pd.DataFrame({

    "ID": test_ids,
    "CLASS": final_pred

})

submission.to_csv("FINAL.csv",index=False)

print("FINAL.csv generated successfully!")
print("Normal:", (final_pred==0).sum())
print("Faulty:", (final_pred==1).sum())
"""
Tempo Classification - Improved Model

Uses lightweight temporal features:
- mean & std (baseline-style)
- delta mean & delta std (change over time)
- lag-1 autocorrelation (captures repetitive structure)

Gives a clear improvement over the baseline model.
"""

import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

N_CHANNELS = 64
N_FRAMES = 101


def compute_autocorr(x):
    """Lag-1 autocorrelation along time axis."""
    x = x - np.mean(x)
    if np.std(x) == 0:
        return 0.0
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def extract_features(X):
    n = X.shape[0]
    fb = X.reshape(n, N_CHANNELS, N_FRAMES)

    # baseline-like
    mean_ = fb.mean(axis=2)
    std_ = fb.std(axis=2)

    # delta features
    delta = np.diff(fb, axis=2)
    delta_mean = np.mean(np.abs(delta), axis=2)
    delta_std = delta.std(axis=2)

    # autocorrelation
    autocorr = np.zeros((n, N_CHANNELS))
    for i in range(n):
        for ch in range(N_CHANNELS):
            autocorr[i, ch] = compute_autocorr(fb[i, ch])

    # concatenate
    features = np.hstack([mean_, std_, delta_mean, delta_std, autocorr])
    return features


def main():
    if len(sys.argv) != 3:
        print("Usage: python train_tempo.py <TRAIN_DATA> <MODEL_FILE>")
        sys.exit(1)

    train_file = sys.argv[1]
    model_file = sys.argv[2]

    print("Loading training data...")
    data = joblib.load(train_file)

    X_train = data["features"]
    y_train = data["target"]

    print("Training RandomForest...")
    model = Pipeline([
        ("features", FunctionTransformer(extract_features, validate=False)),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ))
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, model_file)
    print("✓ Tempo model training complete.")

if __name__ == "__main__":
    main()

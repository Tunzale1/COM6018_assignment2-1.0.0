"""
Tempo Classification Model - Random Forest with Temporal Features
Tempo detection is harder than speed because tempo only changes duration,
not pitch. I need to extract temporal features that capture these changes.

key features:
- mean and std (baseline)
- delta features (rate of change)
- autocorrelation (captures repetitive patterns)
"""

import sys
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

# constants for filterbank feature dimensions
N_FREQ_BINS = 64
N_TIME_FRAMES = 101
N_FEATURES_TOTAL = N_FREQ_BINS * N_TIME_FRAMES  # 6464


def compute_autocorr(signal: np.ndarray) -> float:
    """
    calculate lag-1 autocorrelation for a time series.
    This measures how similar consecutive time frames are,which helps detect tempo changes.
    """
    # center signal
    signal_centered = signal - np.mean(signal)
    
    # handle constant signals
    if np.std(signal) == 0:
        return 0.0
    
    # compute correlation between x[:-1] and x[1:]
    autocorr_coef = np.corrcoef(signal_centered[:-1], signal_centered[1:])[0, 1]
    
    if np.isnan(autocorr_coef):
        return 0.0
    
    return autocorr_coef


def extract_feats(data: np.ndarray) -> np.ndarray:
    """
    extract temporal features from filterbank data.
    I extract 5 types of features (each giving 64 values):
    1. mean over time
    2. std over time
    3. delta mean (how fast energy changes)
    4. delta std (variability in changes)
    5. autocorrelation (repetitive patterns)
    total: 320 features
    """

    n_samples = data.shape[0]
    spectrograms = data.reshape(n_samples, N_FREQ_BINS, N_TIME_FRAMES)
    
    # === 1. baseline statistics ===
    feat_mean = np.mean(spectrograms, axis=2)
    feat_std = np.std(spectrograms, axis=2)
    
    # === 2. first-order temporal derivatives (velocity) ===
    # captures rate of change over time
    deltas = np.diff(spectrograms, axis=2)
    feat_delta_mean = np.mean(np.abs(deltas), axis=2)
    feat_delta_std = np.std(deltas, axis=2)
    
    # === 3. lag-1 autocorrelation (periodicity) ===
    # captures repetitive temporal patterns sensitive to tempo
    feat_autocorr = np.zeros((n_samples, N_FREQ_BINS))
    
    for sample_idx in range(n_samples):
        for freq_idx in range(N_FREQ_BINS):
            time_series = spectrograms[sample_idx, freq_idx, :]
            feat_autocorr[sample_idx, freq_idx] = compute_autocorr(time_series)
    
    # === 4. concatenate all features ===
    features = np.hstack([
        feat_mean,
        feat_std,
        feat_delta_mean,
        feat_delta_std,
        feat_autocorr
    ])
    
    return features


def build_model() -> Pipeline:
    """
    construct RF classification pipeline for tempo detection.
    """
    model = Pipeline([
        ('feature_extraction', FunctionTransformer(
            extract_feats,
            validate=False
        )),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,              # Use all CPU cores
            random_state=42,
            verbose=0
        ))
    ])
    
    return model


def train_and_save_model(train_file: str, model_file: str) -> None:
    """
    complete training workflow for tempo classification model.
    """
    # === 1. load training data ===
    print(f"Loading training data from {train_file}...")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found: {train_file}")
    
    data = joblib.load(train_file)
    X_train = data['features']
    y_train = data['target']
    
    print(f"  Loaded {X_train.shape[0]} samples")
    print(f"  Classes: {np.unique(y_train)} (5-class problem)")
    
    # === 2. build model ===
    print("\nBuilding Random Forest pipeline...")
    print("  Features: Temporal statistics + autocorrelation (320 features)")
    print("  Classifier: Random Forest (400 trees)")
    
    model = build_model()
    
    # === 3. train model ===
    print("\nTraining model (this may take 60-90 seconds)...")
    model.fit(X_train, y_train)
    
    # === 4. save model ===
    print(f"\nSaving model to {model_file}...")
    os.makedirs(os.path.dirname(model_file) or '.', exist_ok=True)
    joblib.dump(model, model_file)
    
    # === 5. validate model size ===
    size_mb = os.path.getsize(model_file) / (1024 * 1024)
    print(f"✓ Training complete!")
    print(f"  Model size: {size_mb:.2f} MB")
    
    if size_mb > 80:
        raise ValueError(f"Model size ({size_mb:.1f} MB) exceeds 80 MB limit!!!")
    else:
        print(f"  ✓ Model size is within 80 MB limit")


def main() -> None:
    """
    command-line interface for training tempo classification model.
    """
    if len(sys.argv) != 3:
        print("ERROR: Invalid number of arguments")
        print("\nUsage: python train_tempo.py <TRAIN_DATA> <MODEL_FILE>")
        print("\nExample:")
        print("  python train_tempo.py data/fbank_tempo.train.joblib models/model.tempo.joblib")
        sys.exit(1)
    
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    
    try:
        train_and_save_model(train_file, model_file)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
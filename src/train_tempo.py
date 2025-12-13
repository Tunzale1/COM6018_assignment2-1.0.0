"""
Tempo Classification Model - Random Forest with Advanced Features
This script trains a Random Forest classifier to detect tempo modifications in speech.
Tempo changes rhythm/timing without affecting pitch, requiring sophisticated features.

key approach:
- extract temporal dynamics features (mean, std, delta, autocorrelation)
- compute frame similarity features to capture rhythm patterns
- train Random Forest with optimized hyperparameters
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
    compute lag-1 autocorrelation for a 1D time series signal.
    measures how much a signal correlates with itself shifted by 1 frame.
    useful for detecting periodic patterns in tempo.
    """
    # center the signal
    signal_centered = signal - np.mean(signal)
    
    # handle zero variance case
    if np.std(signal_centered) == 0:
        return 0.0
    
    # compute correlation between signal[:-1] and signal[1:]
    correlation = np.corrcoef(signal_centered[:-1], signal_centered[1:])[0, 1]
    
    # handle NaN values
    return 0.0 if np.isnan(correlation) else correlation


def extract_feat(data: np.ndarray) -> np.ndarray:
    """
    extract features from filterbank spectrograms.
    reduces dimensionality from 6464 to 326 by computing:
        - mean over time (64)
        - std deviation over time (64)
        - delta mean (64) - captures dynamics
        - delta std (64) - captures variation in dynamics
        - autocorrelation lag-1 (64) - captures periodicity
        - frame similarity statistics (6) - strongest tempo cue
    """
    
    n_samples = data.shape[0]
    features_2d = data.reshape(n_samples, N_FREQ_BINS, N_TIME_FRAMES)
    
    # === 1. basic statistics over time ===
    feat_mean = np.mean(features_2d, axis=2)
    feat_std = np.std(features_2d, axis=2)
    
    # === 2. delta features (rate of change) ===
    delta = np.diff(features_2d, axis=2)  # shape: (n_samples, 64, 100)
    feat_delta_mean = np.mean(np.abs(delta), axis=2)
    feat_delta_std = np.std(delta, axis=2)
    
    # === 3. autocorrelation features (periodicity) ===
    feat_autocorr = np.zeros((n_samples, N_FREQ_BINS))
    for i in range(n_samples):
        for freq_bin in range(N_FREQ_BINS):
            feat_autocorr[i, freq_bin] = compute_autocorr(features_2d[i, freq_bin])
    
    # === 4. frame similarity features (rhythm patterns) ===
    # compute similarity between consecutive frames
    frame_similarity = np.zeros((n_samples, N_TIME_FRAMES - 1))
    
    for i in range(n_samples):
        spectrogram = features_2d[i]  # shape: (64, 101)
        
        for t in range(N_TIME_FRAMES - 1):
            frame_current = spectrogram[:, t]
            frame_next = spectrogram[:, t + 1]
            
            # handle zero variance
            if np.std(frame_current) == 0 or np.std(frame_next) == 0:
                frame_similarity[i, t] = 0.0
            else:
                similarity = np.corrcoef(frame_current, frame_next)[0, 1]
                frame_similarity[i, t] = 0.0 if np.isnan(similarity) else similarity
    
    # compute global statistics from frame similarity
    fds_mean = np.mean(frame_similarity, axis=1).reshape(-1, 1)
    fds_std = np.std(frame_similarity, axis=1).reshape(-1, 1)
    fds_skew = np.mean((frame_similarity - fds_mean)**3, axis=1).reshape(-1, 1)
    fds_kurt = np.mean((frame_similarity - fds_mean)**4, axis=1).reshape(-1, 1)
    
    # compute stretch index (variation in rhythm)
    stretch = np.abs(np.diff(frame_similarity, axis=1))
    stretch_mean = np.mean(stretch, axis=1).reshape(-1, 1)
    stretch_std = np.std(stretch, axis=1).reshape(-1, 1)
    
    # combine all frame similarity features
    feat_frame_similarity = np.hstack([
        fds_mean, fds_std, fds_skew, fds_kurt, 
        stretch_mean, stretch_std
    ])
    
    # === 5. concatenate all features ===
    features = np.hstack([
        feat_mean,              # 64 features
        feat_std,               # 64 features
        feat_delta_mean,        # 64 features
        feat_delta_std,         # 64 features
        feat_autocorr,          # 64 features
        feat_frame_similarity   # 6 features
    ])
    # total: 326 features
    
    return features


def build_model() -> Pipeline:
    """
    construct complete classification pipeline.
    """
    # feature extraction transformer
    extractor = FunctionTransformer(
        extract_feat,
        validate=False
    )
    
    # build 3-stage pipeline
    model = Pipeline([
        ('feature_extraction', extractor),
        ('scaler', StandardScaler()),
        ('random_forest', RandomForestClassifier(
            n_estimators=400,
            max_depth=22,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            verbose=0
        ))
    ])
    
    return model


def train_and_save_model(train_file: str, model_file: str) -> None:
    """
    complete training workflow: load data, train, and save model.
    """
    # === 1. load training data ===
    print(f"Loading training data from {train_file}...")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found: {train_file}")
    
    data = joblib.load(train_file)
    X_train = data['features']
    y_train = data['target']
    
    print(f"  Loaded {X_train.shape[0]} samples with {X_train.shape[1]} features")
    print(f"  Classes: {np.unique(y_train)} (5-class problem)")
    
    # === 2. train model ===
    print("\nBuilding and training Random Forest classifier...")
    print("  Hyperparameters: n_estimators=400, max_depth=22, min_samples_leaf=2")
    print("  Expected time: ~2 minutes\n")
    
    model = build_model()
    model.fit(X_train, y_train)
    
    # === 3. save model ===
    print(f"\nSaving model to {model_file}...")
    os.makedirs(os.path.dirname(model_file) or '.', exist_ok=True)
    joblib.dump(model, model_file)
    
    # === 4. validate model size ===
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
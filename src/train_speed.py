"""
Speed Classification Model - SVM with Data Augmentation
This script trains an SVM classifier to detect speed modifications in speech.
Speed changes both pitch and duration, making it easier to classify than tempo.

key approach:
- extract statistical features (mean, std, max, min over time)
- use data augmentation with noise to improve generalization
- train SVM with RBF kernel
"""

import sys
import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

# constants for filterbank feature dimensions
N_FREQ_BINS = 64
N_TIME_FRAMES = 101
N_FEATURES_TOTAL = N_FREQ_BINS * N_TIME_FRAMES  # 6464


def extract_feat(data: np.ndarray) -> np.ndarray:
    """
    extract features from filterbank spectrograms.
    reduces dimensionality from 6464 to 256 by computing:
        - mean over time (64)
        - std deviation over time (64)
        - max over time (64)
        - min over time (64)
    """
    
    n_samples = data.shape[0]
    features_2d = data.reshape(n_samples, N_FREQ_BINS, N_TIME_FRAMES)
    
    # extract statistics over time (axis=2)
    feat_mean = np.mean(features_2d, axis=2)
    feat_std = np.std(features_2d, axis=2)
    feat_max = np.max(features_2d, axis=2)
    feat_min = np.min(features_2d, axis=2)
    
    # concatenate all features
    features = np.hstack([feat_mean, feat_std, feat_max, feat_min])
    
    return features


def augment_data(
    X: np.ndarray, 
    y: np.ndarray, 
    n_augmentations: int = 2,
    noise_level: float = 0.015
) -> tuple[np.ndarray, np.ndarray]:
    """
    create additional training samples by adding small random noise.
    this helps model generalize better and prevents overfitting.
    i create 2 extra copies of each sample with slightly different noise.
    """
    print(f"Applying data augmentation (creating {n_augmentations} noisy copies)...")
    
    X_copies = [X]
    y_copies = [y]
    
    # create 2 noisy copies
    for i in range(n_augmentations):
        noise = np.random.normal(loc=0.0, scale=noise_level, size=X.shape)
        X_noisy = X + noise
        X_copies.append(X_noisy)
        y_copies.append(y)
    
    # concatenate all copies
    X_augmented = np.vstack(X_copies)
    y_augmented = np.hstack(y_copies)
    
    print(f"  Original samples: {X.shape[0]}")
    print(f"  Augmented samples: {X_augmented.shape[0]} "
          f"({n_augmentations}× increase)")
    
    return X_augmented, y_augmented


def build_model() -> Pipeline:
    """
    construct complete classification pipeline.
    """
    # feature extraction transformer
    extractor = FunctionTransformer(
        extract_feat,
        validate=False
    )
    
    # build 2-stage pipeline
    model = Pipeline([
        ('feature_extraction', extractor),
        ('scaler', StandardScaler()),
        ('svm', SVC(
            C=10.0,
            kernel='rbf',
            gamma='scale',
            cache_size=1000,
            random_state=42,
            verbose=False
        ))
    ])
    
    return model


def train_and_save_model(train_file: str, model_file: str) -> None:
    """
    complete training workflow: load data, augment, train, and save model.
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
    
    # === 2. extract features ===
    print("\nExtracting statistical features...")
    extractor = FunctionTransformer(extract_feat, validate=False)
    X_features = extractor.transform(X_train)
    print(f"  Feature shape: {X_features.shape} (dimensionality reduced)")
    
    # === 3. data augmentation ===
    print("\nAugmenting training data...")
    X_aug, y_aug = augment_data(X_features, y_train, 
                                 n_augmentations=2, noise_level=0.015)
    
    # === 4. train model ===
    print("\nTraining SVM classifier...")
    print("  Hyperparameters: C=10.0, kernel=rbf, gamma=scale")
    print("  Expected time: 10-15 seconds\n")
    
    # build model
    model_inner = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=10.0, kernel='rbf', gamma='scale', 
                   cache_size=1000, random_state=42))
    ])
    
    model_inner.fit(X_aug, y_aug)
    
    # wrap in final pipeline with feature extraction
    final_model = Pipeline([
        ('feature_extraction', extractor),
        ('model', model_inner)
    ])
    
    # === 5. save model ===
    print(f"\nSaving model to {model_file}...")
    os.makedirs(os.path.dirname(model_file) or '.', exist_ok=True)
    joblib.dump(final_model, model_file)
    
    # === 6. validate model size ===
    size_mb = os.path.getsize(model_file) / (1024 * 1024)
    print(f"✓ Training complete!")
    print(f"  Model size: {size_mb:.2f} MB")
    
    if size_mb > 80:
        raise ValueError(f"Model size ({size_mb:.1f} MB) exceeds 80 MB limit!!!")
    else:
        print(f"  ✓ Model size is within 80 MB limit")


def main() -> None:
    """
    command-line interface for training speed classification model.
    """
    if len(sys.argv) != 3:
        print("ERROR: Invalid number of arguments")
        print("\nUsage: python train_speed.py <TRAIN_DATA> <MODEL_FILE>")
        print("\nExample:")
        print("  python train_speed.py data/fbank_speed.train.joblib models/model.speed.joblib")
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
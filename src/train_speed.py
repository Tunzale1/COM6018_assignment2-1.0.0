"""
Speed Classification - SVM Version
To compare with Gradient Boosting
"""
import sys
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

def extract_features(data):
    """
    Extract features for SVM.
    """
    n_samples = data.shape[0]
    features_2d = data.reshape(n_samples, 64, 101)
    
    feature_list = []
    
    # Mean, std, max, min over time
    feat_mean = np.mean(features_2d, axis=2)
    feat_std = np.std(features_2d, axis=2)
    feat_max = np.max(features_2d, axis=2)
    feat_min = np.min(features_2d, axis=2)
    
    feature_list.extend([feat_mean, feat_std, feat_max, feat_min])
    
    # Total: 256 features
    features = np.hstack(feature_list)
    return features

def augment_data(X, y, noise_level=0.015):
    """
    Data augmentation
    """
    print("Applying data augmentation...")
    
    X_aug_list = [X]
    y_aug_list = [y]
    
    # Create 2 augmented copies
    for _ in range(2):
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        X_aug_list.append(X_noisy)
        y_aug_list.append(y)
    
    X_augmented = np.vstack(X_aug_list)
    y_augmented = np.hstack(y_aug_list)
    
    print(f"Original: {X.shape[0]} → Augmented: {X_augmented.shape[0]}")
    return X_augmented, y_augmented

def main():
    if len(sys.argv) != 3:
        print("Usage: python train_speed_svm.py <TRAIN_DATA> <MODEL_FILE>")
        sys.exit(1)
    
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    
    print(f"Loading training data from {train_file}...")
    data = joblib.load(train_file)
    
    X_train = data['features']
    y_train = data['target']
    
    print(f"Training samples: {X_train.shape}")
    
    # Extract features
    print("Extracting features...")
    extractor = FunctionTransformer(extract_features)
    X_features = extractor.transform(X_train)
    print(f"Feature shape: {X_features.shape}")
    
    # Augment data
    X_aug, y_aug = augment_data(X_features, y_train, noise_level=0.015)
    
    # Train SVM
    print("Training SVM classifier...")
    print("This will take 2-4 minutes...")
    
    model_inner = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            C=10.0,                # Regularization
            kernel='rbf',          # RBF kernel
            gamma='scale',         # Kernel coefficient
            cache_size=1000,       # Speed up
            random_state=42
        ))
    ])
    
    model_inner.fit(X_aug, y_aug)
    
    # Create final pipeline
    final_model = Pipeline([
        ('feature_extraction', extractor),
        ('model', model_inner)
    ])
    
    # Save
    print(f"Saving model to {model_file}...")
    joblib.dump(final_model, model_file)
    
    import os
    size_mb = os.path.getsize(model_file) / (1024 * 1024)
    print(f"✓ Training complete!")
    print(f"Model size: {size_mb:.1f} MB")
    
    if size_mb > 80:
        print("⚠️ WARNING: Model exceeds 80 MB!")

if __name__ == "__main__":
    main()
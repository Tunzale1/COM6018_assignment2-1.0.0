"""
Speed Classification - Improved Model
Beats baseline by using Random Forest instead of 1-NN
"""
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

def extract_features(data):
    """
    Extract features from filterbank data.
    For now: simple averaging over time (like baseline)
    """
    # Reshape to (n_samples, 64, 101)
    n_samples = data.shape[0]
    features_2d = data.reshape(n_samples, 64, 101)
    
    # Average over time axis (axis=2)
    features_avg = np.mean(features_2d, axis=2)
    
    # Also add standard deviation over time for extra info
    features_std = np.std(features_2d, axis=2)
    
    # Concatenate: [mean, std] = 128 features
    features = np.hstack([features_avg, features_std])
    
    return features

def main():
    if len(sys.argv) != 3:
        print("Usage: python train_speed.py <TRAIN_DATA> <MODEL_FILE>")
        sys.exit(1)
    
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    
    print(f"Loading training data from {train_file}...")
    data = joblib.load(train_file)
    
    X_train = data['features']
    y_train = data['target']
    
    print(f"Training samples: {X_train.shape}")
    
    # Create pipeline with feature extraction built-in
    print("Training Random Forest classifier...")
    model = Pipeline([
        ('feature_extraction', FunctionTransformer(extract_features)),
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=100,      # 100 trees
            max_depth=20,          # Limit depth to prevent overfitting
            random_state=42,       # Reproducibility
            n_jobs=-1              # Use all CPU cores
        ))
    ])
    
    # Train
    model.fit(X_train, y_train)
    
    # Save model
    print(f"Saving model to {model_file}...")
    joblib.dump(model, model_file)
    print("✓ Training complete!")

if __name__ == "__main__":
    main()
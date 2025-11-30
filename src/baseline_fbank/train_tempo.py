"""Script to train baseline tempo model.

Trains a model from the FBANK features. A simple dimensionality reduction
is performed by averaging each frequency channel over time to form a 64-dim
feature vector. Classification is performed with a k-NN model with k=1.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

N_CHANNELS = 64  # Must matching the number of FBANK channels


def average_frames(X: np.ndarray) -> np.ndarray:
    """Average over time frames."""
    n_samples = X.shape[0]
    return np.mean(X.reshape(n_samples, N_CHANNELS, -1), axis=2)


def train(data_file: Path, model_file: Path) -> None:
    """Train the baseline knn model"""

    if not model_file.parent.exists():
        model_file.parent.mkdir(parents=True)

    train_data = joblib.load(data_file)

    baseline_pipeline = Pipeline(
        [
            ("average_frames", FunctionTransformer(average_frames, validate=False)),
            ("knn", KNeighborsClassifier(n_neighbors=1)),
        ]
    )

    baseline_pipeline.fit(train_data["features"], train_data["target"])

    joblib.dump(baseline_pipeline, model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline tempo model")
    parser.add_argument(
        "data_file",
        type=Path,
        help="Path to training data file (joblib format)",
    )
    parser.add_argument(
        "model_file",
        type=Path,
        help="Path to save trained model (joblib format)",
    )
    args = parser.parse_args()

    train(args.data_file, args.model_file)

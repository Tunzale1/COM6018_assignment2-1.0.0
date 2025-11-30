"""Script to make the baseline speed model from signals.

Trains a model from the raw signals. The signal is transformed into a
64-channel filterbank representation. Then a simple dimensionality reduction
is performed by averaging each frequency channel over time to form a 64-dim
feature vector. Classification is performed with a k-NN model with k=1.
"""

import argparse
from pathlib import Path

import joblib
import librosa as lr
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Signal and Filterbank parameters
SAMPLE_RATE = 16000
N_CHANNELS = 64
WINDOW_LENGTH = int(0.025 * SAMPLE_RATE)  # 400
HOP_LENGTH = int(0.010 * SAMPLE_RATE)  # 160
N_FFT = 512
FREQ_MIN, FREQ_MAX = 40, 7500


def make_fbank(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Compute log-mel filterbank with shape [n_mels, T]."""

    # Power mel spectrogram
    mel_spectrogram = lr.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WINDOW_LENGTH,
        window="hann",
        center=True,
        power=2.0,
        n_mels=N_CHANNELS,
        fmin=FREQ_MIN,
        fmax=FREQ_MAX,
    )

    # Convert to log-dB with clipped dynamic range (stable, standard)
    log_mel_spectrogram = lr.power_to_db(mel_spectrogram, ref=1.0, top_db=80.0).astype(
        np.float32
    )

    return log_mel_spectrogram


def average_frames(X: np.ndarray) -> np.ndarray:
    """Average over time frames."""
    n_samples = X.shape[0]
    return np.mean(X.reshape(n_samples, 64, -1), axis=2)


def train(data_file: Path, model_file: Path) -> None:
    """Train the baseline knn model"""

    if not model_file.parent.exists():
        model_file.parent.mkdir(parents=True)

    train_data = joblib.load(data_file)

    baseline_pipeline = Pipeline(
        [
            ("make_fbank", FunctionTransformer(make_fbank, validate=False)),
            ("average_frames", FunctionTransformer(average_frames, validate=False)),
            ("knn", KNeighborsClassifier(n_neighbors=1)),
        ]
    )

    baseline_pipeline.fit(train_data["features"], train_data["target"])

    joblib.dump(baseline_pipeline, model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train baseline speed model from signals."
    )
    parser.add_argument(
        "data_file",
        type=Path,
        help="Path to training data file (joblib format)",
        nargs="?",
    )
    parser.add_argument(
        "model_file",
        type=Path,
        help="Path to save trained model (joblib format)",
        nargs="?",
    )
    args = parser.parse_args()

    train(args.data_file, args.model_file)

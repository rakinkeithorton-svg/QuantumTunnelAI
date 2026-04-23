from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HBAR = 1.055e-34
ELECTRON_MASS = 9.11e-31
EV_TO_J = 1.602e-19
TRANSMISSION_FLOOR = 1e-40
MODEL_DIR = Path(__file__).resolve().parent / "saved_model"
MODEL_PATH = MODEL_DIR / "model.npz"
METRICS_PATH = MODEL_DIR / "metrics.json"


def compute_transmission_exact(v0_ev: np.ndarray, energy_ev: np.ndarray, width_nm: np.ndarray) -> np.ndarray:
    v0 = v0_ev * EV_TO_J
    energy = energy_ev * EV_TO_J
    width = width_nm * 1e-9

    kappa = np.sqrt((2.0 * ELECTRON_MASS * np.maximum(v0 - energy, 0.0)) / HBAR**2)
    sinh_term = np.sinh(kappa * width)
    denominator = 1.0 + (v0**2 * sinh_term**2) / (4.0 * energy * np.maximum(v0 - energy, 1e-30))
    transmission = 1.0 / denominator
    transmission = np.where(energy_ev >= v0_ev, 1.0, transmission)
    return np.clip(transmission, TRANSMISSION_FLOOR, 1.0)


def build_feature_matrix(v0_ev: np.ndarray, energy_ev: np.ndarray, width_nm: np.ndarray) -> np.ndarray:
    gap_ev = np.maximum(v0_ev - energy_ev, 0.0)
    decay_proxy = np.sqrt(gap_ev + 1e-12) * width_nm
    return np.column_stack([v0_ev, energy_ev, width_nm, gap_ev, gap_ev * width_nm, decay_proxy])


def generate_dataset(sample_count: int = 20000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    v0_ev = rng.uniform(1.0, 20.0, size=sample_count)
    energy_margin = rng.uniform(0.05, 0.95, size=sample_count)
    energy_ev = np.maximum(0.05, v0_ev * energy_margin)
    width_nm = rng.uniform(0.1, 2.0, size=sample_count)

    transmission = compute_transmission_exact(v0_ev, energy_ev, width_nm)
    features = build_feature_matrix(v0_ev, energy_ev, width_nm)
    targets = np.log10(transmission).reshape(-1, 1)
    return features, targets


@dataclass
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float


def prepare_data(sample_count: int = 20000, seed: int = 42) -> DatasetSplit:
    features, targets = generate_dataset(sample_count=sample_count, seed=seed)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))
    features = features[indices]
    targets = targets[indices]

    cutoff = int(len(features) * 0.85)
    x_train_raw = features[:cutoff]
    y_train_raw = targets[:cutoff]
    x_val_raw = features[cutoff:]
    y_val_raw = targets[cutoff:]

    x_mean = x_train_raw.mean(axis=0)
    x_std = x_train_raw.std(axis=0) + 1e-8
    y_mean = float(y_train_raw.mean())
    y_std = float(y_train_raw.std() + 1e-8)

    x_train = (x_train_raw - x_mean) / x_std
    x_val = (x_val_raw - x_mean) / x_std
    y_train = (y_train_raw - y_mean) / y_std
    y_val = (y_val_raw - y_mean) / y_std

    return DatasetSplit(x_train, y_train, x_val, y_val, x_mean, x_std, y_mean, y_std)


class SimpleMLP:
    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, hidden_dim))
        self.b2 = np.zeros((1, hidden_dim))
        self.w3 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, 1))
        self.b3 = np.zeros((1, 1))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(0.0, z2)
        y_pred = a2 @ self.w3 + self.b3
        cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "y_pred": y_pred}
        return y_pred, cache

    def backward(self, cache: dict[str, np.ndarray], y_true: np.ndarray) -> dict[str, np.ndarray]:
        batch_size = y_true.shape[0]
        error = (cache["y_pred"] - y_true) * (2.0 / batch_size)

        grads: dict[str, np.ndarray] = {}
        grads["w3"] = cache["a2"].T @ error
        grads["b3"] = error.sum(axis=0, keepdims=True)

        da2 = error @ self.w3.T
        dz2 = da2 * (cache["z2"] > 0.0)
        grads["w2"] = cache["a1"].T @ dz2
        grads["b2"] = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ self.w2.T
        dz1 = da1 * (cache["z1"] > 0.0)
        grads["w1"] = cache["x"].T @ dz1
        grads["b1"] = dz1.sum(axis=0, keepdims=True)
        return grads

    def apply_adam(
        self,
        grads: dict[str, np.ndarray],
        moments: dict[str, dict[str, np.ndarray]],
        learning_rate: float,
        step: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
            param = getattr(self, name)
            grad = grads[name]
            moments["m"][name] = beta1 * moments["m"][name] + (1.0 - beta1) * grad
            moments["v"][name] = beta2 * moments["v"][name] + (1.0 - beta2) * (grad**2)

            m_hat = moments["m"][name] / (1.0 - beta1**step)
            v_hat = moments["v"][name] / (1.0 - beta2**step)
            setattr(self, name, param - learning_rate * m_hat / (np.sqrt(v_hat) + eps))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def train(sample_count: int = 20000, epochs: int = 100, hidden_dim: int = 64, seed: int = 42) -> tuple[SimpleMLP, DatasetSplit, dict[str, list[float]]]:
    data = prepare_data(sample_count=sample_count, seed=seed)
    model = SimpleMLP(input_dim=data.x_train.shape[1], hidden_dim=hidden_dim, seed=seed)
    rng = np.random.default_rng(seed)
    moments = {
        "m": {name: np.zeros_like(getattr(model, name)) for name in ("w1", "b1", "w2", "b2", "w3", "b3")},
        "v": {name: np.zeros_like(getattr(model, name)) for name in ("w1", "b1", "w2", "b2", "w3", "b3")},
    }

    history = {"train_loss": [], "val_loss": []}
    batch_size = 128
    learning_rate = 0.003
    global_step = 0

    for _epoch in range(epochs):
        indices = rng.permutation(data.x_train.shape[0])
        x_train = data.x_train[indices]
        y_train = data.y_train[indices]

        for start in range(0, x_train.shape[0], batch_size):
            stop = start + batch_size
            xb = x_train[start:stop]
            yb = y_train[start:stop]
            predictions, cache = model.forward(xb)
            grads = model.backward(cache, yb)
            global_step += 1
            model.apply_adam(grads, moments, learning_rate, global_step)

        train_pred, _ = model.forward(data.x_train)
        val_pred, _ = model.forward(data.x_val)
        history["train_loss"].append(mse(data.y_train, train_pred))
        history["val_loss"].append(mse(data.y_val, val_pred))

    return model, data, history


def unscale_predictions(predictions: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    return predictions * y_std + y_mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the QuantumTunnelAI neural approximation model.")
    parser.add_argument("--sample-count", type=int, default=20000, help="Number of synthetic training samples to generate.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Width of each hidden layer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model, data, history = train(
        sample_count=args.sample_count,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )

    val_scaled, _ = model.forward(data.x_val)
    val_log_true = unscale_predictions(data.y_val, data.y_mean, data.y_std)
    val_log_pred = unscale_predictions(val_scaled, data.y_mean, data.y_std)
    val_true = np.clip(10.0 ** val_log_true, TRANSMISSION_FLOOR, 1.0)
    val_pred = np.clip(10.0 ** val_log_pred, TRANSMISSION_FLOOR, 1.0)

    np.savez(
        MODEL_PATH,
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        w3=model.w3,
        b3=model.b3,
        x_mean=data.x_mean,
        x_std=data.x_std,
        y_mean=np.array(data.y_mean),
        y_std=np.array(data.y_std),
    )

    scatter_slice = slice(0, min(100, len(val_true)))
    scatter_points = [
        {
            "true_value": float(val_true[i, 0]),
            "predicted_value": float(val_pred[i, 0]),
        }
        for i in range(scatter_slice.stop)
    ]

    metrics = {
        "sample_count": int(data.x_train.shape[0] + data.x_val.shape[0]),
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "training_loss_final": round(history["train_loss"][-1], 6),
        "validation_loss_final": round(history["val_loss"][-1], 6),
        "validation_mae": round(float(np.mean(np.abs(val_true - val_pred))), 8),
        "validation_log_mae": round(float(np.mean(np.abs(val_log_true - val_log_pred))), 6),
        "scatter_points": scatter_points,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Saved trained model to {MODEL_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

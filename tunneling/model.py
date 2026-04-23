from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HBAR = 1.055e-34
ELECTRON_MASS = 9.11e-31
EV_TO_J = 1.602e-19
NM_TO_M = 1e-9
TRANSMISSION_FLOOR = 1e-40
MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_model"
MODEL_PATH = MODEL_DIR / "model.npz"
METRICS_PATH = MODEL_DIR / "metrics.json"


def compute_transmission_exact(barrier_height_ev: float, particle_energy_ev: float, barrier_width_nm: float) -> float:
    if particle_energy_ev >= barrier_height_ev:
        return 1.0

    v0 = barrier_height_ev * EV_TO_J
    energy = particle_energy_ev * EV_TO_J
    width = barrier_width_nm * NM_TO_M

    kappa = np.sqrt((2.0 * ELECTRON_MASS * (v0 - energy)) / HBAR**2)
    sinh_term = np.sinh(kappa * width)
    denominator = 1.0 + (v0**2 * sinh_term**2) / (4.0 * energy * (v0 - energy))
    transmission = 1.0 / denominator
    return float(np.clip(transmission, TRANSMISSION_FLOOR, 1.0))


def build_feature_matrix(
    barrier_height_ev: np.ndarray | float,
    particle_energy_ev: np.ndarray | float,
    barrier_width_nm: np.ndarray | float,
) -> np.ndarray:
    v0 = np.asarray(barrier_height_ev, dtype=np.float64)
    energy = np.asarray(particle_energy_ev, dtype=np.float64)
    width = np.asarray(barrier_width_nm, dtype=np.float64)
    gap = np.maximum(v0 - energy, 0.0)
    decay_proxy = np.sqrt(gap + 1e-12) * width
    return np.column_stack(
        [
            v0.reshape(-1),
            energy.reshape(-1),
            width.reshape(-1),
            gap.reshape(-1),
            (gap * width).reshape(-1),
            decay_proxy.reshape(-1),
        ]
    )


def compute_wave_numbers(barrier_height_ev: float, particle_energy_ev: float) -> tuple[float, float]:
    energy = particle_energy_ev * EV_TO_J
    barrier = barrier_height_ev * EV_TO_J
    k = np.sqrt(2.0 * ELECTRON_MASS * energy) / HBAR
    kappa = np.sqrt(2.0 * ELECTRON_MASS * max(barrier - energy, 0.0)) / HBAR
    return float(k), float(kappa)


def solve_rectangular_barrier(barrier_height_ev: float, particle_energy_ev: float, barrier_width_nm: float) -> dict:
    if particle_energy_ev >= barrier_height_ev:
        raise ValueError("Wavefunction solver is currently defined for the tunneling regime E < V0.")

    width_m = barrier_width_nm * NM_TO_M
    k, kappa = compute_wave_numbers(barrier_height_ev, particle_energy_ev)
    exp_kappa = np.exp(kappa * width_m)
    exp_minus_kappa = np.exp(-kappa * width_m)
    exp_ika = np.exp(1j * k * width_m)

    matrix = np.array(
        [
            [1.0, -1.0, -1.0, 0.0],
            [-1j * k, -kappa, kappa, 0.0],
            [0.0, exp_kappa, exp_minus_kappa, -exp_ika],
            [0.0, kappa * exp_kappa, -kappa * exp_minus_kappa, -1j * k * exp_ika],
        ],
        dtype=np.complex128,
    )
    rhs = np.array([-1.0, -1j * k, 0.0, 0.0], dtype=np.complex128)
    reflection, coeff_a, coeff_b, transmission = np.linalg.solve(matrix, rhs)

    return {
        "k": k,
        "kappa": kappa,
        "reflection_amplitude": reflection,
        "coeff_a": coeff_a,
        "coeff_b": coeff_b,
        "transmission_amplitude": transmission,
    }


def build_wavefunction_payload(
    barrier_height_ev: float, particle_energy_ev: float, barrier_width_nm: float, samples: int = 720
) -> dict:
    solution = solve_rectangular_barrier(barrier_height_ev, particle_energy_ev, barrier_width_nm)
    width_nm = barrier_width_nm
    left_extent_nm = max(0.7, 0.9 * width_nm)
    right_extent_nm = max(0.9, 1.6 * width_nm)
    x_nm = np.linspace(-left_extent_nm, width_nm + right_extent_nm, samples)
    x_m = x_nm * NM_TO_M
    width_m = width_nm * NM_TO_M

    reflection = solution["reflection_amplitude"]
    coeff_a = solution["coeff_a"]
    coeff_b = solution["coeff_b"]
    transmission = solution["transmission_amplitude"]
    k = solution["k"]
    kappa = solution["kappa"]

    psi = np.zeros_like(x_m, dtype=np.complex128)
    left_mask = x_nm < 0.0
    barrier_mask = (x_nm >= 0.0) & (x_nm <= width_nm)
    right_mask = x_nm > width_nm

    psi[left_mask] = np.exp(1j * k * x_m[left_mask]) + reflection * np.exp(-1j * k * x_m[left_mask])
    psi[barrier_mask] = coeff_a * np.exp(kappa * x_m[barrier_mask]) + coeff_b * np.exp(-kappa * x_m[barrier_mask])
    psi[right_mask] = transmission * np.exp(1j * k * x_m[right_mask])

    density = np.abs(psi) ** 2
    transmission_probability = compute_transmission_exact(barrier_height_ev, particle_energy_ev, barrier_width_nm)
    reflection_probability = float(np.clip(1.0 - transmission_probability, 0.0, 1.0))
    penetration_depth_nm = float((1.0 / max(kappa, 1e-30)) / NM_TO_M)
    potential_profile = np.where(barrier_mask, barrier_height_ev, 0.0)

    return {
        "x_nm": x_nm.tolist(),
        "real_wavefunction": np.real(psi).tolist(),
        "imag_wavefunction": np.imag(psi).tolist(),
        "magnitude": np.abs(psi).tolist(),
        "density": density.tolist(),
        "potential_profile_ev": potential_profile.tolist(),
        "barrier_left_nm": 0.0,
        "barrier_right_nm": float(width_nm),
        "barrier_height_ev": float(barrier_height_ev),
        "particle_energy_ev": float(particle_energy_ev),
        "transmission_probability": float(transmission_probability),
        "reflection_probability": reflection_probability,
        "penetration_depth_nm": penetration_depth_nm,
    }


def build_comparison_payload(
    model: QuantumTunnelModel | None,
    barrier_height_ev: float,
    particle_energy_ev: float,
    current_width_nm: float,
    points: int = 100,
) -> dict:
    widths = np.linspace(0.1, 2.0, points)
    exact = np.array(
        [compute_transmission_exact(barrier_height_ev, particle_energy_ev, float(width)) for width in widths],
        dtype=np.float64,
    )

    predicted = None
    if model is not None:
        raw_features = np.column_stack(
            [
                np.full(points, barrier_height_ev, dtype=np.float64),
                np.full(points, particle_energy_ev, dtype=np.float64),
                widths,
            ]
        )
        predicted = model.predict_batch(raw_features).reshape(-1)

    current_exact = compute_transmission_exact(barrier_height_ev, particle_energy_ev, current_width_nm)
    current_predicted = model.predict(barrier_height_ev, particle_energy_ev, current_width_nm) if model is not None else None
    relative_error_curve = None
    max_relative_error = None
    mean_relative_error = None
    if predicted is not None:
        relative_error_curve = np.abs(predicted - exact) / np.maximum(exact, TRANSMISSION_FLOOR)
        max_relative_error = float(np.max(relative_error_curve))
        mean_relative_error = float(np.mean(relative_error_curve))

    return {
        "width_values_nm": widths.tolist(),
        "exact_values": exact.tolist(),
        "predicted_values": predicted.tolist() if predicted is not None else [],
        "relative_error_values": relative_error_curve.tolist() if relative_error_curve is not None else [],
        "current_width_nm": float(current_width_nm),
        "current_exact": float(current_exact),
        "current_predicted": float(current_predicted) if current_predicted is not None else None,
        "max_relative_error": max_relative_error,
        "mean_relative_error": mean_relative_error,
    }


def compute_transmission_exact_grid(
    barrier_height_ev: float, particle_energy_ev: np.ndarray, barrier_width_nm: np.ndarray
) -> np.ndarray:
    energy = np.asarray(particle_energy_ev, dtype=np.float64)
    width = np.asarray(barrier_width_nm, dtype=np.float64)

    v0 = barrier_height_ev * EV_TO_J
    energy_j = energy * EV_TO_J
    width_m = width * NM_TO_M

    delta = np.maximum(v0 - energy_j, 1e-30)
    kappa = np.sqrt((2.0 * ELECTRON_MASS * delta) / HBAR**2)
    sinh_term = np.sinh(kappa * width_m)
    denominator = 1.0 + (v0**2 * sinh_term**2) / (4.0 * np.maximum(energy_j, 1e-30) * delta)
    transmission = 1.0 / denominator
    transmission = np.where(energy >= barrier_height_ev, 1.0, transmission)
    return np.clip(transmission, TRANSMISSION_FLOOR, 1.0)


@dataclass
class QuantumTunnelModel:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def load(cls) -> "QuantumTunnelModel":
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Saved model not found at {MODEL_PATH}. Run train_model.py first."
            )

        with np.load(MODEL_PATH) as data:
            return cls(
                w1=data["w1"],
                b1=data["b1"],
                w2=data["w2"],
                b2=data["b2"],
                w3=data["w3"],
                b3=data["b3"],
                x_mean=data["x_mean"],
                x_std=data["x_std"],
                y_mean=float(data["y_mean"]),
                y_std=float(data["y_std"]),
            )

    def _forward(self, features: np.ndarray) -> np.ndarray:
        hidden1 = np.maximum(0.0, features @ self.w1 + self.b1)
        hidden2 = np.maximum(0.0, hidden1 @ self.w2 + self.b2)
        return hidden2 @ self.w3 + self.b3

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape {features.shape}.")
        if features.shape[1] == 3:
            features = build_feature_matrix(features[:, 0], features[:, 1], features[:, 2])
        elif features.shape[1] != self.x_mean.shape[0]:
            raise ValueError(
                f"Expected feature width 3 or {self.x_mean.shape[0]}, got {features.shape[1]}."
            )
        normalized = (features - self.x_mean) / self.x_std
        prediction_scaled = self._forward(normalized)
        prediction_log = prediction_scaled * self.y_std + self.y_mean
        transmission = 10.0 ** prediction_log
        return np.clip(transmission, TRANSMISSION_FLOOR, 1.0)

    def predict(self, barrier_height_ev: float, particle_energy_ev: float, barrier_width_nm: float) -> float:
        features = build_feature_matrix(barrier_height_ev, particle_energy_ev, barrier_width_nm)
        transmission = self.predict_batch(features)
        return float(transmission[0, 0])


def build_surface_payload(
    model: QuantumTunnelModel | None,
    barrier_height_ev: float,
    energy_steps: int = 14,
    width_steps: int = 14,
) -> dict:
    energy_min = max(0.05, barrier_height_ev * 0.05)
    energy_max = max(energy_min + 0.05, barrier_height_ev * 0.95)
    energies = np.linspace(energy_min, energy_max, energy_steps)
    widths = np.linspace(0.1, 2.0, width_steps)

    energy_grid, width_grid = np.meshgrid(energies, widths, indexing="ij")
    exact_grid = compute_transmission_exact_grid(barrier_height_ev, energy_grid, width_grid)

    predicted_grid = None
    error_grid = None
    if model is not None:
        features = np.column_stack(
            [
                np.full(energy_grid.size, barrier_height_ev, dtype=np.float64),
                energy_grid.reshape(-1),
                width_grid.reshape(-1),
            ]
        )
        predicted_grid = model.predict_batch(
            build_feature_matrix(features[:, 0], features[:, 1], features[:, 2])
        ).reshape(energy_grid.shape)
        error_grid = np.abs(predicted_grid - exact_grid) / np.maximum(exact_grid, TRANSMISSION_FLOOR)
        error_grid = np.clip(error_grid, 0.0, 1.0)

    return {
        "barrier_height_ev": round(float(barrier_height_ev), 4),
        "energy_values": [round(float(value), 4) for value in energies],
        "width_values": [round(float(value), 4) for value in widths],
        "exact_grid": exact_grid.tolist(),
        "predicted_grid": predicted_grid.tolist() if predicted_grid is not None else [],
        "error_grid": error_grid.tolist() if error_grid is not None else [],
    }


def build_physics_payload(
    barrier_height_ev: float, particle_energy_ev: float, barrier_width_nm: float
) -> dict:
    wavefunction_payload = build_wavefunction_payload(barrier_height_ev, particle_energy_ev, barrier_width_nm, samples=480)
    return {
        "x_nm": wavefunction_payload["x_nm"],
        "potential_profile_ev": wavefunction_payload["potential_profile_ev"],
        "barrier_left_nm": wavefunction_payload["barrier_left_nm"],
        "barrier_right_nm": wavefunction_payload["barrier_right_nm"],
        "barrier_height_ev": wavefunction_payload["barrier_height_ev"],
        "particle_energy_ev": wavefunction_payload["particle_energy_ev"],
        "energy_gap_ev": float(barrier_height_ev - particle_energy_ev),
        "barrier_width_nm": float(barrier_width_nm),
        "transmission_probability": wavefunction_payload["transmission_probability"],
        "penetration_depth_nm": wavefunction_payload["penetration_depth_nm"],
    }


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())

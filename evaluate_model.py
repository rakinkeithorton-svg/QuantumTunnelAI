from __future__ import annotations

import argparse
import json

import numpy as np

from tunneling.model import QuantumTunnelModel, TRANSMISSION_FLOOR, compute_transmission_exact


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved QuantumTunnelAI model on random tunneling inputs.")
    parser.add_argument("--samples", type=int, default=200, help="Number of sampled inputs to evaluate.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    model = QuantumTunnelModel.load()
    rows = []

    for _ in range(args.samples):
        barrier_height = rng.uniform(1.0, 20.0)
        particle_energy = rng.uniform(max(0.05, 0.03 * barrier_height), 0.97 * barrier_height)
        barrier_width = rng.uniform(0.1, 2.0)
        exact = compute_transmission_exact(barrier_height, particle_energy, barrier_width)
        predicted = model.predict(barrier_height, particle_energy, barrier_width)
        relative_error = abs(predicted - exact) / max(exact, TRANSMISSION_FLOOR)
        rows.append(
            {
                "v0": float(barrier_height),
                "e": float(particle_energy),
                "a": float(barrier_width),
                "exact": float(exact),
                "predicted": float(predicted),
                "relative_error": float(relative_error),
            }
        )

    relative_errors = np.array([row["relative_error"] for row in rows], dtype=np.float64)
    exact_values = np.array([row["exact"] for row in rows], dtype=np.float64)
    predicted_values = np.array([row["predicted"] for row in rows], dtype=np.float64)

    small_bucket = [row["relative_error"] for row in rows if row["exact"] < 1e-6]
    mid_bucket = [row["relative_error"] for row in rows if 1e-6 <= row["exact"] < 1e-2]
    large_bucket = [row["relative_error"] for row in rows if row["exact"] >= 1e-2]

    summary = {
        "count": len(rows),
        "mean_relative_error": float(relative_errors.mean()),
        "median_relative_error": float(np.median(relative_errors)),
        "p90_relative_error": float(np.quantile(relative_errors, 0.9)),
        "p95_relative_error": float(np.quantile(relative_errors, 0.95)),
        "max_relative_error": float(relative_errors.max()),
        "mean_absolute_error": float(np.mean(np.abs(predicted_values - exact_values))),
        "mean_relative_error_small_T_<1e-6": float(np.mean(small_bucket)) if small_bucket else None,
        "mean_relative_error_mid_T_1e-6_to_1e-2": float(np.mean(mid_bucket)) if mid_bucket else None,
        "mean_relative_error_large_T_>=1e-2": float(np.mean(large_bucket)) if large_bucket else None,
        "worst_cases": sorted(rows, key=lambda row: row["relative_error"], reverse=True)[:10],
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

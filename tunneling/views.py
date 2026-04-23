from __future__ import annotations

from django.shortcuts import render

from .forms import TunnelingInputForm
from .model import (
    QuantumTunnelModel,
    TRANSMISSION_FLOOR,
    build_comparison_payload,
    build_physics_payload,
    build_surface_payload,
    build_wavefunction_payload,
    compute_transmission_exact,
    load_metrics,
)


def home(request):
    form = TunnelingInputForm(request.POST or None)
    prediction = None
    exact_value = None
    relative_error = None
    prediction_display = None
    exact_display = None
    relative_error_display = None
    reflection_display = None
    penetration_depth_display = None
    model_error = None
    surface_barrier_height = form["barrier_height_ev"].value() or form.fields["barrier_height_ev"].initial
    selected_barrier_height = float(form.fields["barrier_height_ev"].initial)
    selected_particle_energy = float(form.fields["particle_energy_ev"].initial)
    selected_barrier_width = float(form.fields["barrier_width_nm"].initial)

    try:
        surface_barrier_height_value = float(surface_barrier_height)
    except (TypeError, ValueError):
        surface_barrier_height_value = float(form.fields["barrier_height_ev"].initial)

    try:
        model = QuantumTunnelModel.load()
    except FileNotFoundError as exc:
        model = None
        model_error = str(exc)

    if request.method == "POST" and form.is_valid():
        barrier_height = form.cleaned_data["barrier_height_ev"]
        particle_energy = form.cleaned_data["particle_energy_ev"]
        barrier_width = form.cleaned_data["barrier_width_nm"]
        selected_barrier_height = barrier_height
        selected_particle_energy = particle_energy
        selected_barrier_width = barrier_width
        surface_barrier_height_value = barrier_height

        exact_value = compute_transmission_exact(barrier_height, particle_energy, barrier_width)
        reflection_display = f"{(1.0 - exact_value):.4e}"
        if model is not None:
            prediction = model.predict(barrier_height, particle_energy, barrier_width)
            baseline = max(exact_value, TRANSMISSION_FLOOR)
            relative_error = abs(prediction - exact_value) / baseline
            prediction_display = f"{prediction:.4e}"
            exact_display = f"{exact_value:.4e}"
            relative_error_display = f"{relative_error:.4%}"
        wavefunction_snapshot = build_wavefunction_payload(barrier_height, particle_energy, barrier_width, samples=720)
        penetration_depth_display = f"{wavefunction_snapshot['penetration_depth_nm']:.4f} nm"
    else:
        wavefunction_snapshot = build_wavefunction_payload(
            selected_barrier_height,
            selected_particle_energy,
            selected_barrier_width,
            samples=720,
        )
        exact_value = wavefunction_snapshot["transmission_probability"]
        reflection_display = f"{wavefunction_snapshot['reflection_probability']:.4e}"
        penetration_depth_display = f"{wavefunction_snapshot['penetration_depth_nm']:.4f} nm"
        if model is not None:
            prediction = model.predict(selected_barrier_height, selected_particle_energy, selected_barrier_width)
            baseline = max(exact_value, TRANSMISSION_FLOOR)
            relative_error = abs(prediction - exact_value) / baseline
            prediction_display = f"{prediction:.4e}"
            exact_display = f"{exact_value:.4e}"
            relative_error_display = f"{relative_error:.4%}"

    metrics = load_metrics()
    scatter_points = metrics.get("scatter_points", [])
    surface_payload = build_surface_payload(model, surface_barrier_height_value)
    physics_payload = build_physics_payload(
        selected_barrier_height,
        selected_particle_energy,
        selected_barrier_width,
    )
    comparison_payload = build_comparison_payload(
        model,
        selected_barrier_height,
        selected_particle_energy,
        selected_barrier_width,
    )

    context = {
        "form": form,
        "prediction": prediction,
        "exact_value": exact_value,
        "relative_error": relative_error,
        "prediction_display": prediction_display,
        "exact_display": exact_display,
        "relative_error_display": relative_error_display,
        "reflection_display": reflection_display,
        "penetration_depth_display": penetration_depth_display,
        "metrics": metrics,
        "physics_payload": physics_payload,
        "comparison_payload": comparison_payload,
        "scatter_points_json": scatter_points,
        "physics_payload_json": physics_payload,
        "wavefunction_payload_json": wavefunction_snapshot,
        "comparison_payload_json": comparison_payload,
        "surface_payload_json": surface_payload,
        "surface_barrier_height": f"{surface_barrier_height_value:.2f}",
        "selected_barrier_height": f"{selected_barrier_height:.2f}",
        "selected_particle_energy": f"{selected_particle_energy:.2f}",
        "selected_barrier_width": f"{selected_barrier_width:.2f}",
        "energy_ratio_display": f"{selected_particle_energy / selected_barrier_height:.3f}",
        "comparison_mean_error_display": (
            f"{comparison_payload['mean_relative_error']:.3%}"
            if comparison_payload["mean_relative_error"] is not None
            else "Unavailable"
        ),
        "comparison_max_error_display": (
            f"{comparison_payload['max_relative_error']:.3%}"
            if comparison_payload["max_relative_error"] is not None
            else "Unavailable"
        ),
        "model_error": model_error,
    }
    return render(request, "tunneling/index.html", context)

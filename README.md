# QuantumTunnelAI

QuantumTunnelAI is a Django web application for exploring one-dimensional quantum tunneling through a finite rectangular barrier. It combines an exact analytical solver, a small NumPy neural network, and interactive browser visualizations so users can compare physical transmission probabilities with learned predictions.

The current project focuses on stationary under-barrier tunneling where `E < V0`.

## Features

- Exact transmission probability for a finite rectangular potential barrier
- Continuity-matched wavefunction and probability-density visualization
- Neural approximation trained on synthetic tunneling samples
- Exact vs neural transmission sweep across barrier widths
- Interactive 3D transmission landscape across particle energy and barrier width
- Batch evaluation script for randomized model-quality checks
- Django UI with form validation for physically valid tunneling inputs

## Demo Workflow

In the web app, choose:

- `V0`: rectangular barrier height in electron-volts
- `E`: incident particle energy in electron-volts
- `a`: barrier width in nanometers

The app returns:

- exact transmission probability `T`
- neural model prediction
- relative prediction error
- reflection probability
- penetration depth
- wavefunction, density, sweep, and surface plots

## Tech Stack

- Python
- Django
- NumPy
- HTML, CSS, and browser canvas rendering
- SQLite for the local Django database

## Project Structure

```text
QuantumTunnelAI/
├── manage.py
├── QuantumTunnelAI/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── tunneling/
│   ├── forms.py
│   ├── model.py
│   ├── views.py
│   ├── urls.py
│   ├── templates/tunneling/index.html
│   └── static/tunneling/style.css
├── saved_model/
│   ├── model.npz
│   └── metrics.json
├── train_model.py
├── evaluate_model.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/QuantumTunnelAI.git
cd QuantumTunnelAI
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run database migrations:

```bash
python manage.py migrate
```

## Run Locally

Start the Django development server:

```bash
python manage.py runserver
```

Open the app:

```text
http://127.0.0.1:8000/
```

## Train The Model

The neural network is trained from synthetic samples generated from the exact finite-barrier transmission formula.

```bash
python train_model.py --sample-count 20000 --epochs 100 --hidden-dim 64 --seed 42
```

Training writes:

- `saved_model/model.npz`: learned weights and normalization values
- `saved_model/metrics.json`: training and validation metrics

Available options:

```text
--sample-count   Number of synthetic training samples
--epochs         Number of training epochs
--hidden-dim     Width of each hidden layer
--seed           Random seed for reproducible training
```

## Evaluate The Model

Run a randomized evaluation sweep:

```bash
python evaluate_model.py --samples 200 --seed 123
```

This reports relative error statistics, absolute error, bucketed errors by transmission scale, and worst sampled cases.

## Current Saved Model

Metrics from the included `saved_model/metrics.json`:

| Metric | Value |
| --- | ---: |
| Training samples | 20,000 |
| Epochs | 100 |
| Hidden width | 64 |
| Final training loss | 0.000041 |
| Final validation loss | 0.000043 |
| Validation MAE | 0.00144681 |
| Validation log MAE | 0.035267 |

Randomized evaluation using:

```bash
python evaluate_model.py --samples 200 --seed 123
```

| Metric | Value |
| --- | ---: |
| Mean relative error | 0.08648 |
| Median relative error | 0.06041 |
| P90 relative error | 0.18310 |
| P95 relative error | 0.25408 |
| Mean absolute error | 0.00075521 |

## Physics Scope

This version models:

- one-dimensional finite rectangular barriers
- stationary-state scattering
- under-barrier tunneling with `E < V0`
- transmission, reflection, wavefunction structure, and penetration depth

It does not yet model:

- above-barrier scattering
- time-dependent wave packets
- arbitrary potential profiles
- multi-dimensional quantum systems

## Model Notes

The neural network predicts `log10(T)` rather than raw transmission probability. This keeps training numerically stable when transmission becomes extremely small.

The model uses engineered features derived from:

- `V0`
- `E`
- `a`
- `V0 - E`
- `(V0 - E) * a`
- `sqrt(V0 - E) * a`

The exact analytical solver remains the physical reference. The neural network is used as an approximation and comparison target.

## Development Checks

Run Django's project check:

```bash
python manage.py check
```

Run a quick model evaluation:

```bash
python evaluate_model.py --samples 200 --seed 123
```

## Publishing Notes

Before deploying publicly, replace the demo Django settings with production-safe configuration:

- set `DEBUG = False`
- move `SECRET_KEY` into an environment variable
- configure `ALLOWED_HOSTS`
- use a production database if user data will be stored
- serve static files through a production-ready setup

For GitHub publishing, avoid committing local runtime files such as `.venv/`, `__pycache__/`, `.DS_Store`, and local SQLite databases unless they are intentionally part of the demo.

## Roadmap

- Add above-barrier scattering support
- Expose JSON API endpoints for simulation results
- Add arbitrary potential profiles
- Add residual/error visualization directly in the UI
- Explore physics-informed neural-network approaches

## License

Add a license file before publishing if you want others to use, modify, or distribute the project.

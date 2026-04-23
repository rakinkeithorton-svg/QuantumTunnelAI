# QuantumTunnelAI

QuantumTunnelAI is a Django-based physics and machine learning project for one-dimensional quantum tunneling through a finite rectangular barrier. The application combines an exact stationary-state solver with a neural network approximation, then visualizes both results in a browser interface built around physically meaningful plots instead of placeholder graphics.

## Current scope

The app currently models:

- A finite rectangular potential barrier
- A stationary-state tunneling regime with `E < V0`
- Exact transmission probability from the analytical under-barrier formula
- A learned neural approximation of the same mapping
- Wavefunction structure and probability density across the barrier
- Width sweeps and transmission landscapes for comparison

The current interface intentionally does **not** model:

- Above-barrier scattering mode
- Time-dependent wave packets
- Arbitrary potential shapes
- Multi-dimensional quantum systems

## Features

- Django web app with a physics-first interface
- Exact transmission solver for the tunneling regime
- NumPy neural network trained on synthetic tunneling data
- Interactive browser plots for:
  - Barrier physics
  - Wavefunction structure
  - Exact vs neural width sweep
  - 3D transmission landscape
- Batch evaluation script for measuring model quality over many random inputs

## Model design

The neural model predicts transmission from engineered features derived from:

- `V0`
- `E`
- `a`
- `V0 - E`
- `(V0 - E) * a`
- `sqrt(V0 - E) * a`

This improved the model substantially in the very small-transmission tail, where a simple raw-input model tended to saturate.

To preserve numerical stability, the model is trained on `log10(T)` rather than `T` directly.

## Latest training run

Current saved model metrics from `saved_model/metrics.json`:

- Training samples: `20,000`
- Epochs: `220`
- Hidden width: `64`
- Validation MAE: `0.00075274`
- Validation log-MAE: `0.013232`

Additional randomized batch evaluation from `evaluate_model.py --samples 200 --seed 123`:

- Mean relative error: `0.03334`
- Median relative error: `0.02574`
- P90 relative error: `0.06533`
- P95 relative error: `0.09231`

## Project structure

```text
QuantumTunnelAI/
├── manage.py
├── QuantumTunnelAI/
├── tunneling/
│   ├── forms.py
│   ├── model.py
│   ├── views.py
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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
```

## Run the app

```bash
cd /Users/rakinahmed/Documents/QuantumTunnelAI
source .venv/bin/activate
python manage.py runserver
```

Then open:

```text
http://127.0.0.1:8000/
```

## Retrain the neural model

```bash
python train_model.py --sample-count 20000 --epochs 220 --hidden-dim 64
```

Useful options:

- `--sample-count`: number of generated training samples
- `--epochs`: training epochs
- `--hidden-dim`: hidden-layer width
- `--seed`: reproducible run seed

## Evaluate the saved model

Run a randomized evaluation sweep:

```bash
python evaluate_model.py --samples 200 --seed 123
```

This reports:

- Mean and median relative error
- P90 and P95 relative error
- Mean absolute error
- Error buckets by transmission scale
- Worst-case sampled inputs

## Physics notes

- The interface enforces `E < V0` because the current wavefunction and comparison panels are built specifically for tunneling through a rectangular barrier.
- The wavefunction plot is generated from continuity-matched stationary solutions across the three barrier regions.
- The comparison sweep varies barrier width while holding `V0` and `E` fixed to the currently selected input.

## Why this project stands out

QuantumTunnelAI is not only a UI wrapper around a formula. It demonstrates:

- exact analytical physics
- neural approximation
- error analysis
- batch evaluation
- interactive scientific visualization
- a full-stack Django implementation

## Next extensions

Natural next steps for the project are:

- add above-barrier scattering with the correct oscillatory transmission formula
- expose JSON/API endpoints for simulations
- support arbitrary potential profiles
- add uncertainty or residual visualization in the UI
- explore physics-informed neural networks

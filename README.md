This branch uses 120 synthetic samples per operating state with randomized signal generation for stochastic signal evaluation. Named it '20' by mistake, but generated notebook results as is, so not touching the name at this point.

# Pelagia1

Pelagia is a novel near-shore wave-energy platform with an interpretable edge-intelligence layer for monitoring and decision support.

This repository contains the computational proof-of-concept for the IEEE RTSI 2026 paper on Pelagia. The codebase supports synthetic signal generation, feature extraction, interpretable classification, anomaly detection, streaming-style updates, and figure generation for the paper.

## Project aim

The repository demonstrates a lightweight AI-enabled monitoring and decision-support workflow for a foldable dock-mounted wave-energy platform with three leaf-shaped petals. The proof-of-concept is based on synthetic multi-signal data representing:

- petal motion
- phase relationships between petals
- voltage
- current
- power
- thermal behaviour
- operating and protection states

The analytics layer is designed to support interpretable edge intelligence for:

- normal harvesting mode
- anomaly or inspection recommendation
- adaptive operation under changing water conditions
- protective or stall-like reduced-response mode

## Repository structure

```text
Pelagia1/
├── README.md
├── requirements.txt
├── data/
│   └── synthetic_signals/
├── src/
│   ├── generate_signals.py
│   ├── extract_features.py
│   ├── train_classifier.py
│   ├── train_anomaly_model.py
│   ├── streaming_update.py
│   ├── plot_branches.py
│   └── utils.py
├── notebooks/
│   └── pelagia_demo.ipynb
├── figures/
└── results/

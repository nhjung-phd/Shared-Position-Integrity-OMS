# Experiment Design

## Main factorial
- OMS: 2, 4, 8
- Propagation delay: 0, 2, 5, 10
- Load: 0.6, 1.0, 1.6
- Replications: 80
- Policies: P0, P1C, P2S, P2D, P3
- Orders per episode: 180

## Lifecycle robustness
Fault dimensions are varied independently while partial-fill and cancellation rates are held fixed:
- duplicate probability
- terminal-report loss probability
- out-of-order probability

This avoids confounding fault severity with effective demand pressure.

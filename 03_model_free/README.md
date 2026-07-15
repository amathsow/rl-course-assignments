# Module 03 — Model-Free RL: Monte Carlo & Temporal-Difference

No model this time — learn the optimal policy purely from **sampled experience**.
This is the heart of classical RL.

## What you'll learn
- **Monte Carlo** prediction & control (learn from complete episodes).
- **TD(0)** prediction and the idea of **bootstrapping**.
- **SARSA** (on-policy) and **Q-learning** (off-policy).
- Why SARSA learns a *safe* path and Q-learning a *risky-but-optimal* one
  (the CliffWalking experiment).

## Environments used
- The course `GridWorld` — for MC/TD prediction and control.
- `CliffWalking-v1` (Gymnasium) — the SARSA-vs-Q-learning showdown.

## How to work through it
1. Open `assignment/03_model_free_assignment.ipynb` and select the
   **`Python (rl-course)`** kernel.
2. Fill in every `# TODO` / `raise NotImplementedError`.
3. Run all cells; each method should recover a sensible greedy policy.

## Tasks
1. Implement first-visit **MC prediction** and **MC control**.
2. Implement **TD(0)** prediction and compare its values to MC.
3. Implement **SARSA** and **Q-learning**; visualize both greedy policies.
4. Reproduce the **CliffWalking** SARSA-vs-Q-learning comparison and explain it.

## Prerequisites
Modules 01–02.

## Estimated time
2–3 hours (the meatiest module).

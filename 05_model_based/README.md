# Module 05 — Model-Based RL: Dyna-Q

Instead of discarding each transition after one update, **learn a model** of the
world and **plan** with it — reusing real experience to squeeze out many more
updates. This is the key to **sample efficiency**.

## What you'll learn
- The difference between **model-free** and **model-based** RL.
- Learning a tabular model $\hat P, \hat R$ from experience.
- **Dyna-Q**: interleaving direct RL, model learning, and planning.
- How planning (`n` simulated updates per real step) cuts the number of **real**
  episodes needed — the classic Dyna maze result.
- Pitfalls: stale models, changing environments, and **Dyna-Q+**.

## Environment used
- The course `GridWorld` configured as a deterministic **maze** (the classic Dyna
  testbed).

## How to work through it
1. Open `assignment/05_model_based_assignment.ipynb` and select the
   **`Python (rl-course)`** kernel.
2. Fill in every `# TODO` / `raise NotImplementedError`.
3. Run all cells; more planning steps should visibly reduce the steps-per-episode.

## Tasks
1. Complete the **Dyna-Q** direct-RL and planning updates.
2. Reproduce the "planning steps vs. episodes-to-solve" comparison.
3. Inspect the learned greedy policy and watch it navigate the maze.

## Prerequisites
Module 03 (Q-learning).

## Estimated time
1.5–2 hours.

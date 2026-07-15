# Module 04 — Function Approximation

Tables don't scale to large or continuous state spaces. Here we **approximate**
the value function and learn its parameters by semi-gradient descent — from simple
linear features all the way to a deep Q-network.

## What you'll learn
- Why tabular methods break down; the **semi-gradient** update.
- **Tile coding**: turning continuous states into sparse binary features.
- **Semi-gradient SARSA** with linear FA on MountainCar.
- **DQN**: neural-network value approximation with **experience replay** and a
  **target network**.

## Environments used
- `MountainCar-v0` — continuous state, linear FA + tile coding.
- `CartPole-v1` — DQN.

## How to work through it
1. Open `assignment/04_function_approximation_assignment.ipynb` and select the
   **`Python (rl-course)`** kernel.
2. Fill in every `# TODO` / `raise NotImplementedError`.
3. Run all cells; MountainCar should reach the flag and DQN should improve on CartPole.

## Tasks
1. Complete the **semi-gradient SARSA** update and solve MountainCar (best return
   comfortably above −200); plot the cost-to-go surface.
2. Complete the **DQN target + optimization step** and improve on CartPole.
3. **Ablation:** remove the target network (or the replay buffer) and describe how
   training destabilises. Why does each trick help?

## Prerequisites
Modules 01–03. Basic PyTorch familiarity helps for Part B (a short primer is inline).

## Note
Part B (DQN) trains a small network for ~250 episodes. On CPU this takes a couple
of minutes; increase `n_episodes` for stronger results. A GPU is used automatically
if available.

## Estimated time
2–3 hours.

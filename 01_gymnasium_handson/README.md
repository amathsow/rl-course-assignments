# Module 01 — Getting familiar with Gymnasium

The warm-up module. No learning algorithms yet — the goal is to be fully
comfortable with the **environment interface** that every later module builds on.

## What you'll learn
- The Gymnasium API: `reset()`, `step()`, observation/action **spaces**.
- The **agent–environment loop** and the difference between `terminated` and `truncated`.
- How to measure a policy with the **episode return**.
- How to **render** an environment and watch the agent move as a GIF.
- Writing a simple **rule-based policy** and comparing it to a random agent.

## Environments used
- `CartPole-v1` — continuous observations, discrete actions (the running example).
- `FrozenLake-v1` — a discrete grid world (previewing Module 02).

## How to work through it
1. Open `assignment/01_gymnasium_assignment.ipynb` and select the
   **`Python (rl-course)`** kernel.
2. Fill in every `# TODO` / `raise NotImplementedError`.
3. Run all cells top-to-bottom; your heuristic policy should clearly beat random.

## Tasks
1. Complete the random-agent episode loop.
2. Implement the pole-angle heuristic policy and beat the random baseline.
3. Record a GIF of each policy and confirm the heuristic keeps the pole up longer.

## Estimated time
45–60 minutes.

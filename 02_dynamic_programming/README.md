# Module 02 — Dynamic Programming: Policy & Value Iteration

When the **model is known** ($P$ and $R$), we can compute the optimal policy
**exactly** — no learning required. DP is the theoretical backbone of RL.

## What you'll learn
- The **Bellman expectation** and **optimality** equations.
- **Iterative policy evaluation**.
- **Policy iteration** (evaluation + greedy improvement).
- **Value iteration** and how it relates to policy iteration.
- How the **discount factor** $\gamma$ reshapes the optimal policy.

## Environment used
- The course `GridWorld` (a white-box MDP that exposes its full model as
  `env.P[s][a] -> [(prob, next_state, reward, done), ...]`).

## How to work through it
1. Open `assignment/02_dynamic_programming_assignment.ipynb` and select the
   **`Python (rl-course)`** kernel.
2. Fill in every `# TODO` / `raise NotImplementedError`.
3. Run all cells; policy iteration and value iteration should agree on $V^*$, and
   you should be able to watch the optimal policy act.

## Tasks
1. Implement `policy_evaluation`.
2. Implement `q_from_v` and `policy_iteration`.
3. Implement `value_iteration`, and confirm it matches policy iteration.
4. Explore how the optimal policy changes with the discount factor $\gamma$.

## Prerequisites
Module 01 (Gymnasium API).

## Estimated time
90 minutes.
